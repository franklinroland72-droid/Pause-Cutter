#!/usr/bin/env python3
"""
Suppression automatique des pauses dans une video de presentation
ou dans un fichier audio seul.

Usage :
    python cut.py entree.mp4 sortie.mp4
    python cut.py entree.mp4 sortie.mp3           # extraction audio montee
    python cut.py entree.mp3 sortie.mp3
    python cut.py entree.mp3 sortie.mp3 --dry-run #sans encodage

Le mode de rendu est deduit du couple entree/sortie : une sortie
d'extension audio produit toujours un fichier audio seul, meme depuis
une video ; une sortie video exige une entree contenant un flux video
reel (les pochettes attached_pic ne comptent pas).


Installation :
    Dependances : numpy, ffmpeg et ffprobe

    macOS :
    brew install ffmpeg python

    Linux (Debian, Ubuntu) :
    sudo apt update
    sudo apt install ffmpeg python3 python3-venv python3-pip

    Windows :
    winget install Gyan.FFmpeg
    winget install Python.Python.3.12

    Environnement : 
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# PARAMETRES (calibres sur l'extrait fourni : plancher de bruit ~ -60 dBFS,
# parole ~ -20 dBFS, separation nette, pauses entre 6 s et 13 s a -50 dBFS)
# ---------------------------------------------------------------------------

# --- Analyse ---
ANALYSE_SR = 16000          # Hz, frequence d'echantillonnage de l'analyse
FRAME_S = 0.02              # s, pas de l'enveloppe RMS (20 ms)
PLANCHER_ANALYSE_DB = -80.0 # l'enveloppe est bornee en bas a cette valeur :
                            # le silence numerique exact vaut -200 dB et
                            # etirerait l'histogramme au point de fausser Otsu

# "vallee"   : seuil place dans le creux de l'histogramme en dB (Otsu). A
#              utiliser des qu'un fond parle continu occupe les pauses, car
#              il n'y a alors pas de plancher de bruit a mesurer.
# "plancher" : ancienne methode, valable seulement si les pauses sont du
#              vrai silence.
# "manuel"   : SEUIL_MANUEL_DB.
METHODE_SEUIL = "vallee"
PLANCHER_PERCENTILE = 10    # percentile de l'enveloppe pris pour le plancher
MARGE_AU_DESSUS_DB = 22.0   # seuil = plancher + cette marge, methode "plancher"
SEUIL_MANUEL_DB = -25.0
SEUIL_MIN_DB = -50.0        # bornes de securite du seuil automatique
SEUIL_MAX_DB = -18.0

# --- Regles de montage ---
LISSAGE_FRAMES = 7          # median glissant sur l'enveloppe, en trames (140 ms)
HYSTERESIS_DB = 5.0         # ecart entre seuil d'entree et de sortie de parole
PAUSE_MIN_S = 0.35          # une pause plus courte n'est pas coupee
GARDE_AVANT_S = 0.05        # son conserve avant chaque reprise de parole
GARDE_APRES_S = 0.05        # son conserve apres chaque fin de parole
SEGMENT_MIN_S = 0.30        # un segment de parole plus court est rejete (bruit)

# --- Rendu ---
FONDU_MS = 30               # fondu audio aux jointures, contre les clics
CODEC_VIDEO = "libx264"     # h264_nvenc / h264_videotoolbox si GPU disponible
PRESET = "veryfast"
CRF = "20"                  # ignore par les encodeurs materiels
CODEC_AUDIO = "aac"         # piste audio du rendu video
BITRATE_AUDIO = "160k"
CODEC_MP3 = "libmp3lame"    # rendu audio seul
BITRATE_MP3 = "192k"
JOBS = max(1, (os.cpu_count() or 4) - 1)   # segments encodes en parallele

EXT_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

# ---------------------------------------------------------------------------


def sonde(chemin):
    """Une seule interrogation ffprobe, au format JSON, seule syntaxe stable
    d'une version a l'autre. Retourne (source, sample_rate, canaux, duree).
    Une pochette embarquee est un flux video de disposition attached_pic et
    ne fait donc pas basculer en source video."""
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", chemin],
        capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"ffprobe a echoue sur {chemin} :\n{p.stderr.strip()}")
    info = json.loads(p.stdout)
    flux = info.get("streams", [])

    video = [s for s in flux
             if s.get("codec_type") == "video"
             and not s.get("disposition", {}).get("attached_pic")]
    audio = [s for s in flux if s.get("codec_type") == "audio"]
    if not audio:
        sys.exit(f"aucun flux audio dans {chemin}")

    source = "video" if video else "audio"
    sr = int(audio[0].get("sample_rate") or 44100)
    ch = int(audio[0].get("channels") or 2)

    d = info.get("format", {}).get("duration") or audio[0].get("duration")
    if d is None:
        sys.exit(f"duree indeterminee pour {chemin}")
    return source, sr, ch, float(d)


def duree(chemin):
    return sonde(chemin)[3]


def enveloppe(chemin, duree_totale):
    """Retourne l'enveloppe RMS en dBFS, une valeur par FRAME_S. Le flux
    analyse est explicitement 0:a:0, celui-la meme que le montage decoupera :
    sans ce -map, ffmpeg choisirait le flux audio qu'il juge le meilleur,
    qui n'est pas forcement le premier."""
    taille = int(FRAME_S * ANALYSE_SR)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", chemin, "-map", "0:a:0",
         "-ac", "1", "-ar", str(ANALYSE_SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    blocs, reste, lus = [], b"", 0
    t0 = time.time()
    while True:
        brut = proc.stdout.read(taille * 4 * 500)
        if not brut:
            break
        lus += len(brut)
        brut = reste + brut
        n = len(brut) // (taille * 4)
        if n:
            x = np.frombuffer(brut[:n * taille * 4], dtype=np.float32)
            blocs.append(np.sqrt((x.reshape(n, taille) ** 2).mean(axis=1)))
            reste = brut[n * taille * 4:]
        else:
            reste = brut
        avance = lus / 4 / ANALYSE_SR
        barre(avance / max(duree_totale, 1e-9),
              f"analyse audio {avance/60:6.1f} min", t0)
    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace")
    proc.stderr.close()
    proc.wait()
    print()
    if proc.returncode != 0 or not blocs:
        sys.exit(f"le decodage audio de {chemin} a echoue :\n{err.strip()}")
    rms = np.concatenate(blocs)
    db = 20 * np.log10(rms + 1e-10)
    return np.clip(db, PLANCHER_ANALYSE_DB, 0.0)


def barre(fraction, texte, t0, largeur=30):
    fraction = min(max(fraction, 0.0), 1.0)
    plein = int(fraction * largeur)
    ecoule = time.time() - t0
    reste = ecoule * (1 - fraction) / fraction if fraction > 0.01 else 0
    sys.stdout.write(
        f"\r[{'#' * plein}{'-' * (largeur - plein)}] {fraction*100:5.1f}%  "
        f"{texte}  restant {reste/60:5.1f} min ")
    sys.stdout.flush()


def seuil_vallee(db):
    """Seuil place au creux entre les deux modes de l'histogramme en dB,
    par maximisation de la variance interclasse (Otsu). Le mode bas est la
    voix de fond, le mode haut la voix de premier plan ; le critere ne
    suppose aucun plancher de bruit."""
    h, bords = np.histogram(db, bins=120)
    centres = (bords[:-1] + bords[1:]) / 2
    w = h / h.sum()
    cw = np.cumsum(w)
    cm = np.cumsum(w * centres)
    inter = (cm[-1] * cw - cm) ** 2 / np.maximum(cw * (1 - cw), 1e-12)
    return float(centres[int(np.argmax(inter))])


def lissage_median(v, k):
    if k < 3:
        return v
    k += 1 - k % 2
    pad = np.pad(v, (k // 2, k // 2), mode="edge")
    return np.median(np.stack([pad[i:i + len(v)] for i in range(k)]), axis=0)


def parole_hysteresis(db, haut, bas):
    """Un etat de parole ne s'ouvre qu'au-dessus de `haut` et ne se ferme
    qu'en dessous de `bas` : les creux inter-syllabiques ne fragmentent
    plus les segments, et le fond parle ne suffit pas a en ouvrir un."""
    etat = False
    fort = np.zeros(len(db), dtype=bool)
    for i, v in enumerate(db):
        if etat:
            if v < bas:
                etat = False
        elif v > haut:
            etat = True
        fort[i] = etat
    return fort


def segments_parole(db, duree_totale):
    # Le lissage precede le calcul du seuil : il resserre la distribution,
    # un seuil calcule sur l'enveloppe brute ne tomberait pas au meme
    # endroit relatif de l'enveloppe lissee, celle qui decide reellement.
    db = lissage_median(db, LISSAGE_FRAMES)

    if METHODE_SEUIL == "vallee":
        seuil = seuil_vallee(db)
        seuil = min(max(seuil, SEUIL_MIN_DB), SEUIL_MAX_DB)
        print(f"seuil bimodal retenu {seuil:.1f} dBFS")
    elif METHODE_SEUIL == "plancher":
        plancher = float(np.percentile(db, PLANCHER_PERCENTILE))
        seuil = min(max(plancher + MARGE_AU_DESSUS_DB, SEUIL_MIN_DB), SEUIL_MAX_DB)
        print(f"plancher {plancher:.1f} dBFS, seuil retenu {seuil:.1f} dBFS")
    else:
        seuil = SEUIL_MANUEL_DB
        print(f"seuil fixe {seuil:.1f} dBFS")

    p10 = float(np.percentile(db, PLANCHER_PERCENTILE))
    if p10 > seuil - 12.0:
        print(f"avertissement : niveau bas a {p10:.1f} dBFS, soit moins de "
              f"12 dB sous le seuil ; les pauses ne sont pas du silence "
              f"(fond parle ou bruit), la separation sera imparfaite")

    fort = parole_hysteresis(db, seuil + HYSTERESIS_DB / 2,
                             seuil - HYSTERESIS_DB / 2)
    bords = np.diff(np.concatenate(([0], fort.view(np.int8), [0])))
    debuts = np.flatnonzero(bords == 1) * FRAME_S
    fins = np.flatnonzero(bords == -1) * FRAME_S
    bruts = list(zip(debuts, fins))

    # fusion des trous plus courts que la pause minimale
    fusionnes = []
    for d, f in bruts:
        if fusionnes and d - fusionnes[-1][1] < PAUSE_MIN_S:
            fusionnes[-1] = (fusionnes[-1][0], f)
        else:
            fusionnes.append((d, f))

    # rejet des micro-segments, puis marges de garde
    gardes = []
    for d, f in fusionnes:
        if f - d < SEGMENT_MIN_S:
            continue
        d = max(0.0, d - GARDE_AVANT_S)
        f = min(duree_totale, f + GARDE_APRES_S)
        if gardes and d <= gardes[-1][1]:
            gardes[-1] = (gardes[-1][0], f)
        else:
            gardes.append((d, f))
    return gardes, seuil


def encode(args):
    """Rend un segment. L'audio sort toujours en wav PCM : c'est sans perte,
    et surtout un segment PCM n'a ni delai d'encodeur ni bourrage final,
    contrairement a l'AAC, dont les quelques millisecondes ajoutees a chaque
    bout s'additionneraient sur des centaines de segments et desynchro-
    niseraient la fin du montage. La video sort sans piste audio ; l'une et
    l'autre ne sont reunies qu'apres concatenation, en un seul encodage AAC
    couvrant toute la duree."""
    i, (d, f), source, dossier, mode, sr, ch = args
    fondu = FONDU_MS / 1000.0
    duree_seg = f - d
    af = (f"afade=t=in:st=0:d={fondu},"
          f"afade=t=out:st={max(0.0, duree_seg - fondu):.3f}:d={fondu}")
    # -ss puis -t, tous deux options d'entree : la portee de -to place la
    # depend de la version de ffmpeg, celle de -t est univoque.
    tete = ["ffmpeg", "-v", "error", "-y",
            "-ss", f"{d:.3f}", "-t", f"{duree_seg:.3f}", "-i", source]

    wav = os.path.join(dossier, f"seg{i:06d}.wav")
    sortie_wav = ["-map", "0:a:0", "-af", af,
                  "-c:a", "pcm_s16le", "-ar", str(sr), "-ac", str(ch), wav]

    if mode != "video":
        subprocess.run(tete + sortie_wav, check=True, capture_output=True)
        return None, wav

    mp4 = os.path.join(dossier, f"seg{i:06d}.mp4")
    sortie_mp4 = ["-map", "0:v:0", "-an",
                  "-c:v", CODEC_VIDEO, "-preset", PRESET]
    if CODEC_VIDEO == "libx264":
        sortie_mp4 += ["-crf", CRF]
    sortie_mp4 += ["-pix_fmt", "yuv420p", "-force_key_frames", "expr:eq(n,0)",
                   "-video_track_timescale", "90000",
                   "-avoid_negative_ts", "make_zero", mp4]
    subprocess.run(tete + sortie_mp4 + sortie_wav,
                   check=True, capture_output=True)
    return mp4, wav


def ecrire_liste(chemins, chemin_liste):
    with open(chemin_liste, "w") as fh:
        for c in chemins:
            fh.write(f"file '{c}'\n")
    return chemin_liste


def concatene(liste, sortie, options):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
           "-i", liste] + options + [sortie]
    subprocess.run(cmd, check=True)


def options_audio(sortie):
    ext = os.path.splitext(sortie)[1].lower()
    if ext == ".wav":
        return ["-c:a", "pcm_s16le"]
    if ext == ".mp3":
        return ["-c:a", CODEC_MP3, "-b:a", BITRATE_MP3]
    if ext in (".m4a", ".aac"):
        return ["-c:a", CODEC_AUDIO, "-b:a", BITRATE_AUDIO]
    if ext == ".flac":
        return ["-c:a", "flac"]
    return ["-b:a", BITRATE_MP3]     # codec deduit du conteneur


def muxe(video, audio, sortie):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video, "-i", audio,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", CODEC_AUDIO, "-b:a", BITRATE_AUDIO, "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", sortie], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entree")
    ap.add_argument("sortie")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse seule, aucun encodage")
    a = ap.parse_args()

    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            sys.exit(f"{exe} introuvable dans le PATH")

    if not os.path.isfile(a.entree):
        sys.exit(f"fichier introuvable : {a.entree}")

    source, sr, ch, total = sonde(a.entree)
    ext_sortie = os.path.splitext(a.sortie)[1].lower()
    if ext_sortie in EXT_AUDIO:
        mode = "audio"
    elif source == "audio":
        sys.exit(f"l'entree ne contient pas de video : sortie {ext_sortie} "
                 f"impossible, choisir une extension audio {sorted(EXT_AUDIO)}")
    else:
        mode = "video"

    detail = f"{sr} Hz, {ch} canaux" if source == "audio" else "video + audio"
    print(f"source : {a.entree}  ({total/3600:.2f} h)  {detail}"
          f"  ->  rendu {mode}")

    db = enveloppe(a.entree, total)
    segs, _ = segments_parole(db, total)
    if not segs:
        sys.exit("aucun segment de parole detecte, revoir le seuil")

    garde = sum(f - d for d, f in segs)
    print(f"{len(segs)} segments conserves, {garde/60:.1f} min sur "
          f"{total/60:.1f} min ({100*garde/total:.1f} %), "
          f"{(total-garde)/60:.1f} min de pauses supprimees")
    if a.dry_run:
        for d, f in segs[:40]:
            print(f"  {d:9.2f} -> {f:9.2f}   ({f-d:5.2f} s)")
        return

    dossier = tempfile.mkdtemp(prefix="pauses_")
    try:
        taches = [(i, s, a.entree, dossier, mode, sr, ch)
                  for i, s in enumerate(segs)]
        rendus = [None] * len(taches)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=JOBS) as ex:
            futs = {ex.submit(encode, t): t[0] for t in taches}
            faits = 0
            for fut in concurrent.futures.as_completed(futs):
                rendus[futs[fut]] = fut.result()
                faits += 1
                barre(faits / len(taches),
                      f"encodage {faits}/{len(taches)} segments", t0)
        print()

        wavs = [w for _, w in rendus]
        print("concatenation...")
        if mode == "audio":
            concatene(ecrire_liste(wavs, os.path.join(dossier, "audio.txt")),
                      a.sortie, options_audio(a.sortie))
        else:
            piste = os.path.join(dossier, "piste.wav")
            concatene(ecrire_liste(wavs, os.path.join(dossier, "audio.txt")),
                      piste, ["-c:a", "pcm_s16le"])
            muet = os.path.join(dossier, "muet.mp4")
            concatene(ecrire_liste([v for v, _ in rendus],
                                   os.path.join(dossier, "video.txt")),
                      muet, ["-c", "copy"])
            muxe(muet, piste, a.sortie)
    finally:
        shutil.rmtree(dossier, ignore_errors=True)

    print(f"ecrit : {a.sortie}  ({duree(a.sortie)/60:.1f} min)")


if __name__ == "__main__":
    main()