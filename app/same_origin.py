"""Same-origin confidence: do the stems, master, and Logic project all come
from the same session?

Implements the layered rubric:
  A. Audio DNA -- the stems must perceptually reconstruct the master
     (coherence), and the project's Media/ takes must be findable inside
     the stems/master (containment matching with a full-length slide).
  B. Superset / authorship-shape evidence -- real sessions contain MORE
     than the release: unused takes, project file backups, recording
     timestamps spread over multiple dates.
  C. Metadata consistency -- sample rate, declared duration, file-name
     cross references. Logic's untouched defaults (120 BPM / C major) are
     detected and excluded rather than scored.

Every check yields match / contradiction / not_evaluable. Only evaluable
checks carry weight (weights renormalize), and only contradictions or
red-flag signatures subtract -- absence of evidence abstains, it never
condemns. Red-flag caps handle the two theft signatures (project media
exactly equals the stems with nothing else; zero audio-DNA overlap).

Nothing here proves authorship. The output is a triage aid: it measures
whether this material plausibly came from one session, for a human
adjudicator to weigh alongside custody and priority evidence.
"""

import plistlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import audio, coherence, fingerprint, hashing, logic_inspect

MAX_MEDIA_FILES = 12
MEDIA_TRIM_S = 45
STEM_TRIM_S = 120
MIN_MEDIA_SECONDS = 3.0
CONTAIN_MATCH = 0.65       # containment similarity that counts as "found"
CONTAIN_NOISE = 0.55       # below this a media file is "unique to project"
MIN_OVERLAP_FRAMES = 24    # ~3 s of fingerprint overlap minimum

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def containment_similarity(needle: list[int], haystack: list[int]) -> float:
    """Best bit-similarity of `needle` slid across every position of
    `haystack` (order-agnostic: shorter print is the needle)."""
    if not needle or not haystack:
        return 0.0
    nd = np.asarray(needle, dtype=np.uint64)
    hs = np.asarray(haystack, dtype=np.uint64)
    if len(nd) > len(hs):
        nd, hs = hs, nd
    n, m = len(nd), len(hs)
    best = 0.0
    for off in range(0, m - MIN_OVERLAP_FRAMES + 1):
        ov = min(n, m - off)
        if ov < MIN_OVERLAP_FRAMES:
            break
        x = np.bitwise_xor(hs[off:off + ov], nd[:ov]).astype(np.uint32)
        errs = int(_POP[x.view(np.uint8)].sum())
        best = max(best, 1.0 - errs / (32.0 * ov))
    return round(best, 4)


def _safe_extract(zip_path: Path, dest: Path) -> list[Path]:
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            target = (dest / zi.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi) as src, open(target, "wb") as f:
                shutil.copyfileobj(src, f)
            out.append(target)
    return out


def _find_metadata(files: list[Path]) -> dict | None:
    candidates = [f for f in files if f.name == "MetaData.plist"
                  and "Project File Backups" not in str(f)]
    for f in sorted(candidates, key=lambda p: str(p)):
        try:
            return plistlib.loads(f.read_bytes())
        except Exception:
            continue
    return None


def _decode_fp(src: Path, workdir: Path, name: str, trim_s: int) -> tuple[Path, list[int]]:
    norm = workdir / f"{name}.wav"
    audio.decode_to_normalized_wav(src, norm)
    fp, dur = fingerprint.compute_fingerprint(norm)
    frames = int(trim_s / 0.124)
    return norm, fp[:frames]


def analyze(project_zip: bytes, stems: list[tuple[str, bytes]],
            master: tuple[str, bytes]) -> dict:
    """Full same-origin analysis. Inputs are (filename, bytes) pairs."""
    tmp = Path(tempfile.mkdtemp(prefix="same_origin_"))
    try:
        return _analyze(tmp, project_zip, stems, master)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _analyze(tmp: Path, project_zip: bytes, stems, master) -> dict:
    checks: list[dict] = []
    flags: list[str] = []

    def check(name, weight, result, note, value=None):
        checks.append({"check": name, "weight": weight, "result": result,
                       "note": note, "value": value})

    # ---- unpack project -------------------------------------------------
    zpath = tmp / "project.zip"
    zpath.write_bytes(project_zip)
    try:
        files = _safe_extract(zpath, tmp / "proj")
    except zipfile.BadZipFile:
        return {"error": "project upload is not a valid zip"}

    metadata = _find_metadata(files)
    backups = [f for f in files if "Project File Backups" in str(f)]
    media_files = []
    for f in files:
        if "/Media/" not in str(f) and "/Audio Files/" not in str(f):
            continue
        head = f.read_bytes()[:12] if f.stat().st_size >= 12 else b""
        if head[:4] in (b"RIFF", b"FORM", b"caff") or f.suffix.lower() in (
                ".wav", ".aif", ".aiff", ".caf", ".mp3", ".m4a"):
            media_files.append(f)
    media_files = sorted(media_files, key=lambda p: -p.stat().st_size)[:MAX_MEDIA_FILES]

    # ---- decode + fingerprint everything --------------------------------
    master_norm, master_fp = _decode_fp(_write(tmp, "master", master), tmp,
                                        "master_norm", STEM_TRIM_S)
    stem_norms, stem_fps, stem_shas = [], [], set()
    for i, s in enumerate(stems):
        p = _write(tmp, f"stem_{i}", s)
        stem_shas.add(hashing.sha256_file(p))
        norm, fp = _decode_fp(p, tmp, f"stem_norm_{i}", STEM_TRIM_S)
        stem_norms.append(norm)
        stem_fps.append(fp)

    media = []  # {name, sha, fp or None}
    for i, f in enumerate(media_files):
        entry = {"name": f.name, "sha": hashing.sha256_file(f), "fp": None}
        try:
            norm = tmp / f"media_norm_{i}.wav"
            audio.decode_to_normalized_wav(f, norm)
            fp, dur = fingerprint.compute_fingerprint(norm)
            if dur >= MIN_MEDIA_SECONDS and fp:
                entry["fp"] = fp[:int(MEDIA_TRIM_S / 0.124)]
        except RuntimeError:
            pass
        media.append(entry)
    usable_media = [m for m in media if m["fp"]]

    # ---- A. audio DNA ---------------------------------------------------
    coh = coherence.check_coherence(master_norm, stem_norms, tmp)
    check("A1 stems reconstruct master (coherence)", 20,
          "match" if coh.verified else
          ("contradiction" if coh.confidence < 0.6 else "not_evaluable"),
          f"mixdown-vs-master similarity {coh.confidence:.3f}"
          + (f"; flags: {coh.flags}" if coh.flags else ""),
          round(coh.confidence, 4))

    media_results = []
    if usable_media:
        for m in usable_media:
            if m["sha"] in stem_shas:
                m["best_stem"] = 1.0
            else:
                m["best_stem"] = max(containment_similarity(m["fp"], sf)
                                     for sf in stem_fps)
            m["best_master"] = containment_similarity(m["fp"], master_fp)
            media_results.append({"file": m["name"],
                                  "best_in_stems": m["best_stem"],
                                  "best_in_master": m["best_master"]})
        in_stems = sum(1 for m in usable_media if m["best_stem"] >= CONTAIN_MATCH)
        in_master = sum(1 for m in usable_media if m["best_master"] >= CONTAIN_MATCH)
        frac_stems = in_stems / len(usable_media)
        frac_master = in_master / len(usable_media)
        check("A2 project takes found in stems", 15,
              "match" if frac_stems >= 0.5 else
              ("contradiction" if frac_stems == 0 else "not_evaluable"),
              f"{in_stems}/{len(usable_media)} media files matched a stem "
              f"(containment >= {CONTAIN_MATCH})", round(frac_stems, 3))
        stems_covered = sum(
            1 for sf in stem_fps
            if any(containment_similarity(m["fp"], sf) >= CONTAIN_MATCH
                   for m in usable_media))
        frac_cov = stems_covered / len(stem_fps) if stem_fps else 0
        check("A3 stems explained by project media", 10,
              "match" if frac_cov >= 0.5 else
              ("contradiction" if frac_cov == 0 else "not_evaluable"),
              f"{stems_covered}/{len(stem_fps)} stems contain at least one "
              "project take", round(frac_cov, 3))
        check("A4 project takes audible in master", 10,
              "match" if frac_master >= 0.4 else
              ("contradiction" if frac_master == 0 else "not_evaluable"),
              f"{in_master}/{len(usable_media)} media files matched the master",
              round(frac_master, 3))
    else:
        note = ("project has no usable recorded audio (all-MIDI/virtual-"
                "instrument session, cleaned project, or takes too short) -- "
                "audio-DNA checks abstain")
        for name, w in (("A2 project takes found in stems", 15),
                        ("A3 stems explained by project media", 10),
                        ("A4 project takes audible in master", 10)):
            check(name, w, "not_evaluable", note)

    # ---- B. superset evidence -------------------------------------------
    unique = [m for m in usable_media
              if m["best_stem"] < CONTAIN_NOISE and m["best_master"] < CONTAIN_NOISE]
    unused_declared = bool(metadata and metadata.get("UnusedAudioFiles"))
    if usable_media:
        check("B1 project holds material absent from release", 10,
              "match" if (unique or unused_declared) else "not_evaluable",
              f"{len(unique)} media files unique to the project"
              + ("; MetaData lists UnusedAudioFiles" if unused_declared else ""),
              len(unique))
    else:
        check("B1 project holds material absent from release", 10,
              "match" if unused_declared else "not_evaluable",
              "MetaData lists UnusedAudioFiles" if unused_declared
              else "no usable media to assess")

    check("B2 edit history (project file backups)", 7,
          "match" if backups else "not_evaluable",
          f"{len(backups)} backup files inside the package" if backups
          else "no Project File Backups folder (absence is not suspicious)")

    bext_dates = set()
    for f in media_files:
        try:
            wc = logic_inspect.parse_wav_chunks(f.read_bytes())
            d = (wc.get("bext") or {}).get("origination_date")
            if d:
                bext_dates.add(d)
        except Exception:
            pass
    check("B3 recording sessions spread over time", 8,
          "match" if len(bext_dates) >= 2 else "not_evaluable",
          f"bext recording dates found: {sorted(bext_dates) or 'none'}",
          len(bext_dates))

    # ---- C. metadata consistency ----------------------------------------
    proj_sr = metadata.get("SampleRate") if metadata else None
    master_probe_sr = _wav_sr(master_norm)  # normalized -- compare declared only
    if proj_sr:
        check("C1 sample rate declared", 4,
              "match" if proj_sr in (44100, 48000) else "not_evaluable",
              f"project declares {proj_sr} Hz (uploads are normalized to 44.1k "
              "before comparison; 48k sessions mastered to 44.1k are normal)",
              proj_sr)
    else:
        check("C1 sample rate declared", 4, "not_evaluable", "no MetaData.plist")

    bpm = metadata.get("BeatsPerMinute") if metadata else None
    key = metadata.get("SongKey") if metadata else None
    if bpm == 120.0 and key == "C":
        check("C2 tempo/key declaration", 0, "not_evaluable",
              "project shows Logic's untouched defaults (120 BPM, C) -- "
              "excluded from scoring as likely never set")
    elif bpm or key:
        check("C2 tempo/key declaration", 0, "not_evaluable",
              f"declared {bpm} BPM / key {key} (measurement not implemented "
              "in this POC; informational only)")
    else:
        check("C2 tempo/key declaration", 0, "not_evaluable", "not declared")

    song_end = metadata.get("SongEndTime") if metadata else None
    master_dur = len(master_fp) * 0.124
    if song_end and master_dur:
        ratio = master_dur / float(song_end) if song_end else 0
        check("C3 project length vs master length", 3,
              "match" if 0.5 <= ratio <= 1.5 else "not_evaluable",
              f"master ~{master_dur:.0f}s vs SongEndTime {song_end} "
              "(end markers are unreliable; gross mismatch only abstains)",
              round(ratio, 2))
    else:
        check("C3 project length vs master length", 3, "not_evaluable",
              "SongEndTime not present")

    declared = {Path(a).name.lower()
                for a in (metadata.get("AudioFiles") or [])} if metadata else set()
    actual = {f.name.lower() for f in media_files}
    if declared and actual:
        overlap = len(declared & actual) / max(len(declared), 1)
        check("C4 declared audio files exist in Media/", 3,
              "match" if overlap >= 0.5 else "contradiction" if overlap == 0
              else "not_evaluable",
              f"{len(declared & actual)}/{len(declared)} declared files present",
              round(overlap, 3))
    else:
        check("C4 declared audio files exist in Media/", 3, "not_evaluable",
              "no declared AudioFiles list or no media")

    # ---- red-flag signatures --------------------------------------------
    cap = 100.0
    media_shas = {m["sha"] for m in media}
    if media and media_shas and media_shas.issubset(stem_shas) and not backups \
            and len(bext_dates) <= 1:
        flags.append(
            "project media is exactly the uploaded stems with no extra takes, "
            "no backups, no session spread -- consistent with a project "
            "reconstructed around existing stems (also a legitimate "
            "record-then-mix workflow; human review needed)")
        cap = min(cap, 40.0)
    if usable_media and all(m["best_stem"] < 0.5 and m["best_master"] < 0.5
                            for m in usable_media):
        flags.append("zero audio-DNA overlap between project media and the "
                     "uploaded stems/master")
        cap = min(cap, 20.0)
    if coh.flags:
        flags.extend(coh.flags)
        cap = min(cap, 50.0)

    # ---- score ----------------------------------------------------------
    evaluable = [c for c in checks if c["result"] != "not_evaluable" and c["weight"]]
    total_w = sum(c["weight"] for c in evaluable)
    got_w = sum(c["weight"] for c in evaluable if c["result"] == "match")
    contradictions = [c for c in checks if c["result"] == "contradiction"]
    raw = 100.0 * got_w / total_w if total_w else 0.0
    score = round(min(raw, cap), 1)
    band = ("strong" if score >= 75 else "moderate" if score >= 50
            else "weak" if score >= 25 else "contradicted")
    if contradictions and band in ("strong", "moderate"):
        band = "weak"

    return {
        "score": score,
        "band": band,
        "cap_applied": cap if cap < 100 else None,
        "checks": checks,
        "red_flags": flags,
        "contradictions": [c["check"] for c in contradictions],
        "media_matching": media_results,
        "profile": ("audio session" if usable_media else
                    "no usable recorded media (MIDI/cleaned profile)"),
        "coherence": {"verified": coh.verified,
                      "confidence": round(coh.confidence, 4),
                      "flags": coh.flags},
        "notes": [
            "score = matched weight / evaluable weight; absent evidence "
            "abstains rather than subtracts; red-flag signatures cap the score",
            "this measures same-session plausibility, not authorship",
        ],
    }


def _write(tmp: Path, name: str, item: tuple[str, bytes]) -> Path:
    p = tmp / f"{name}_{Path(item[0]).name or 'file'}"
    p.write_bytes(item[1])
    return p


def _wav_sr(path: Path) -> int | None:
    m = re.search(rb"fmt .{6}(.{4})", path.read_bytes()[:200], re.S)
    return int.from_bytes(m.group(1)[:4], "little") if m else None
