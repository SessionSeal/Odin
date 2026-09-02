"""FastAPI app: upload, public record, evidence, verify.

Run with:  uvicorn app.main:app --reload
"""

import io
import json
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import (audio, fingerprint, hashing, logic_inspect, pg,
               same_origin, simulate, verify, watermark)

# A re-encoded copy of a sealed master must score at least this to re-link.
VERIFY_THRESHOLD = 0.80

REQUIRED_BINARIES = {
    "ffmpeg": "brew install ffmpeg   (or https://ffmpeg.org/download.html)",
    "fpcalc": "brew install chromaprint   (or https://acoustid.org/chromaprint)",
    "c2patool": "brew install c2patool   (or https://github.com/contentauth/c2pa-rs/tree/main/cli)",
    "openssl": "brew install openssl   (usually preinstalled)",
    "audiowmark": "build from source -- see benchmarks/README.md",
}


def check_prerequisites() -> None:
    missing = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        print("Missing required external tools:", file=sys.stderr)
        for name in missing:
            print(f"  {name}: install with -> {REQUIRED_BINARIES[name]}", file=sys.stderr)
        print("Install them and restart the service.", file=sys.stderr)
        sys.exit(1)


check_prerequisites()

app = FastAPI(
    title="SessionSeal Odin (verify)",
    description=(
        "Tamper-evident, timestamped records of a master, its stems, and its "
        "project file. Records prove custody, integrity, coherence, and "
        "priority. They do not prove authorship."
    ),
)

# The web app calls this API directly (the Next.js dev proxy chokes on
# multi-hundred-MB multipart bodies, e.g. real .logicx packages).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/logic/inspect")
async def inspect_logic_project(project: UploadFile = File(...)):
    """POC: exhaustively inspect a zipped .logicx package.

    Parses everything parseable (plists incl. NSKeyedArchiver, media chunk
    metadata, images, embedded strings in opaque binaries) and returns a
    report that accounts for every file in the zip. Ephemeral: nothing is
    stored and no provenance record is created.
    """
    data = await project.read()
    try:
        return logic_inspect.inspect_logicx_zip(data, project.filename or "upload.zip")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not inspect zip: {e}")


@app.post("/logic/inspect-files")
async def inspect_logic_project_folder(
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(...),
):
    """POC: inspect a .logicx package picked as a *folder* in the browser.

    A .logicx is a directory, so the web app sends every file inside it along
    with its package-relative path. They are repacked into an in-memory zip
    and run through the same inspector as the zip upload. Ephemeral.
    """
    if len(files) != len(paths):
        raise HTTPException(status_code=422, detail="files and paths must pair up")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f, rel in zip(files, paths):
            rel = rel.replace("\\", "/").lstrip("/")
            if not rel or ".." in rel.split("/"):
                raise HTTPException(status_code=422, detail=f"unsafe path: {rel}")
            zf.writestr(rel, await f.read())
    name = (paths[0].split("/", 1)[0] if paths else "package") + " (folder upload)"
    try:
        return logic_inspect.inspect_logicx_zip(buf.getvalue(), name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not inspect package: {e}")


def _fingerprint_original(data: bytes, filename: str, tmp: Path) -> tuple[dict, Path, list[int]]:
    """Shared analysis: save upload, fingerprint + probe + loudness."""
    raw = tmp / (Path(filename or "audio").name or "audio")
    raw.write_bytes(data)
    norm = audio.decode_to_normalized_wav(raw, tmp / "orig_norm.wav")
    fp_raw, duration = fingerprint.compute_fingerprint(norm)
    compressed = fingerprint.compute_compressed_fingerprint(norm)
    report = {
        "filename": raw.name,
        "size_bytes": len(data),
        "sha256": hashing.sha256_file(raw),
        "format": simulate.probe(raw),
        "loudness": simulate.measure_loudness(raw),
        "fingerprint": {
            "algorithm": "chromaprint",
            "analyzed_seconds": duration,
            "frames": len(fp_raw),
            "compressed": compressed,
        },
    }
    return report, raw, fp_raw


@app.post("/fingerprint/analyze")
async def fingerprint_analyze(audio_file: UploadFile = File(...)):
    """Chromaprint fingerprint + format + EBU R128 loudness of an upload."""
    data = await audio_file.read()
    with tempfile.TemporaryDirectory() as td:
        try:
            report, _, _ = _fingerprint_original(data, audio_file.filename, Path(td))
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
    return report


@app.post("/fingerprint/simulate")
async def fingerprint_simulate(
    audio_file: UploadFile = File(...),
    codec: str = Form("vorbis"),        # vorbis | aac | mp3 | opus | none
    bitrate: int = Form(160),           # kbps
    loudnorm: str = Form("none"),       # none | -11 | -14 | -16 | -23 (LUFS)
    sample_rate: str = Form("keep"),    # keep | 44100 | 48000
    generations: int = Form(1),         # total encode passes (1 = single encode)
):
    """Apply a platform-style processing chain, then compare fingerprints.

    Fingerprints the original, runs the selected codec/loudness/resample
    chain via ffmpeg, fingerprints the result, and reports similarity
    against the /verify re-link threshold.
    """
    if codec not in simulate.VALID_CODECS:
        raise HTTPException(status_code=422, detail=f"codec must be one of {sorted(simulate.VALID_CODECS)}")
    bitrate = max(32, min(int(bitrate), 512))
    try:
        loudnorm_target = None if loudnorm == "none" else float(loudnorm)
        sr = None if sample_rate == "keep" else int(sample_rate)
    except ValueError:
        raise HTTPException(status_code=422, detail="bad loudnorm or sample_rate value")
    if loudnorm_target is not None and not -30.0 <= loudnorm_target <= -5.0:
        raise HTTPException(status_code=422, detail="loudnorm target must be between -30 and -5 LUFS")
    if sr is not None and sr not in (44100, 48000):
        raise HTTPException(status_code=422, detail="sample_rate must be keep, 44100 or 48000")
    generations = max(1, min(int(generations), 10))

    data = await audio_file.read()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            original, raw, orig_fp = _fingerprint_original(data, audio_file.filename, tmp)
            processed_path, steps, commands = simulate.apply_chain(
                raw, tmp,
                codec=codec, bitrate_kbps=bitrate,
                loudnorm_target=loudnorm_target, sample_rate=sr,
                generations=generations,
            )
            proc_norm = audio.decode_to_normalized_wav(processed_path, tmp / "proc_norm.wav")
            proc_fp, proc_dur = fingerprint.compute_fingerprint(proc_norm)
            comparison = fingerprint.compare_detailed(orig_fp, proc_fp)

            # Keep the processed file so the user can download it. Pruned
            # after 2 hours; this is a lab scratch area, not evidence storage.
            LAB_DOWNLOADS.mkdir(parents=True, exist_ok=True)
            now = time.time()
            for old in LAB_DOWNLOADS.glob("*"):
                if now - old.stat().st_mtime > 2 * 3600:
                    old.unlink(missing_ok=True)
            stored = LAB_DOWNLOADS / f"{uuid.uuid4().hex}{processed_path.suffix}"
            shutil.copy2(processed_path, stored)
            base = Path(audio_file.filename or "audio").stem or "audio"
            download = {
                "url": f"/fingerprint/download/{stored.name}",
                "filename": (
                    f"{base}_{codec}_{bitrate}k_x{generations}{processed_path.suffix}"
                    if codec != "none" else f"{base}_processed{processed_path.suffix}"
                ),
                "size_bytes": stored.stat().st_size,
            }
            processed = {
                "size_bytes": processed_path.stat().st_size,
                "size_change_pct": round(
                    100 * (processed_path.stat().st_size - len(data)) / len(data), 1
                ),
                "format": simulate.probe(processed_path),
                "loudness": simulate.measure_loudness(processed_path),
                "fingerprint": {
                    "algorithm": "chromaprint",
                    "analyzed_seconds": proc_dur,
                    "frames": len(proc_fp),
                    "compressed": fingerprint.compute_compressed_fingerprint(proc_norm),
                },
            }
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return {
        "original": original,
        "chain": {"steps": steps, "ffmpeg_commands": commands},
        "processed": processed,
        "download": download,
        "comparison": {
            **comparison,
            "verify_threshold": VERIFY_THRESHOLD,
            "would_relink": comparison["similarity"] >= VERIFY_THRESHOLD,
        },
    }


LAB_DOWNLOADS = Path(__file__).resolve().parent.parent / "storage" / "fingerprint_lab"
_LAB_MEDIA_TYPES = {".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4",
                    ".mp3": "audio/mpeg", ".wav": "audio/wav"}


@app.post("/watermark/embed")
async def watermark_embed(
    audio_file: UploadFile = File(...),
    watermark_id: str = Form(...),
):
    """Embed a UUID as an inaudible echo-hiding watermark (one frame / 10 s).

    Returns a download link for the watermarked WAV plus an immediate
    self-check extraction so you can see the mark went in.
    """
    try:
        watermark_id = str(uuid.UUID(watermark_id.strip()))
    except ValueError:
        raise HTTPException(status_code=422, detail="watermark_id must be a valid UUID")
    data = await audio_file.read()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = tmp / (Path(audio_file.filename or "audio").name or "audio")
        raw.write_bytes(data)
        try:
            norm = audio.decode_to_normalized_wav(raw, tmp / "norm.wav")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        x, sr = sf.read(norm, dtype="float64", always_2d=True)
        y, blocks = watermark.embed(x, watermark_id)
        if blocks == 0:
            raise HTTPException(
                status_code=422,
                detail="audio must be at least 10 seconds long to fit one watermark block",
            )
        out = tmp / "watermarked.wav"
        sf.write(out, y, sr, subtype="PCM_16")
        self_check = watermark.extract(y)

        LAB_DOWNLOADS.mkdir(parents=True, exist_ok=True)
        stored = LAB_DOWNLOADS / f"{uuid.uuid4().hex}.wav"
        shutil.copy2(out, stored)
        base = Path(audio_file.filename or "audio").stem or "audio"

    return {
        "id": watermark_id,
        "blocks_embedded": blocks,
        "duration_seconds": round(x.shape[0] / sr, 2),
        "self_check": {
            "found": self_check["found"],
            "id_matches": self_check.get("id") == watermark_id,
            "confidence": self_check.get("confidence"),
        },
        "download": {
            "url": f"/fingerprint/download/{stored.name}",
            "filename": f"{base}_watermarked.wav",
            "size_bytes": stored.stat().st_size,
        },
    }


@app.post("/watermark/extract")
async def watermark_extract(audio_file: UploadFile = File(...)):
    """Blind-extract a watermark ID from any audio format."""
    data = await audio_file.read()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = tmp / (Path(audio_file.filename or "audio").name or "audio")
        raw.write_bytes(data)
        try:
            norm = audio.decode_to_normalized_wav(raw, tmp / "norm.wav")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        x, _ = sf.read(norm, dtype="float64", always_2d=True)
    return watermark.extract(x)


async def _project_zip_from_upload(
    project: UploadFile | None,
    project_files: list[UploadFile] | None,
    project_paths: list[str] | None,
) -> tuple[bytes, str]:
    """Accept a .logicx either as a zip file or as folder-picked files+paths."""
    if project is not None:
        return await project.read(), project.filename or "project.zip"
    if project_files and project_paths:
        if len(project_files) != len(project_paths):
            raise HTTPException(status_code=422, detail="project files/paths must pair up")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f, rel in zip(project_files, project_paths):
                rel = rel.replace("\\", "/").lstrip("/")
                if not rel or ".." in rel.split("/"):
                    raise HTTPException(status_code=422, detail=f"unsafe path: {rel}")
                zf.writestr(rel, await f.read())
        name = (project_paths[0].split("/", 1)[0] or "project") + ".zip"
        return buf.getvalue(), name
    raise HTTPException(status_code=422,
                        detail="provide the .logicx as a zip or as folder files+paths")


@app.post("/check/stem-master")
async def check_stem_master(
    master: UploadFile = File(...),
    stems: list[UploadFile] = File(...),
):
    """Standalone stems<->master perceptual coherence check. Ephemeral."""
    if len(stems) < 2:
        raise HTTPException(status_code=422, detail="upload at least two stems")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            m_raw = tmp / (Path(master.filename or "master").name or "master")
            m_raw.write_bytes(await master.read())
            m_norm = audio.decode_to_normalized_wav(m_raw, tmp / "m.wav")
            stem_norms = []
            for i, s in enumerate(stems):
                p = tmp / f"s{i}_{Path(s.filename or 'stem').name}"
                p.write_bytes(await s.read())
                stem_norms.append(audio.decode_to_normalized_wav(p, tmp / f"sn{i}.wav"))
            from . import coherence as coherence_mod
            coh = coherence_mod.check_coherence(m_norm, stem_norms, tmp)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
    return {
        "match_rate": round(coh.confidence, 4),
        "verified": coh.verified,
        "threshold": coherence_mod.COHERENCE_THRESHOLD,
        "method": coh.method,
        "flags": coh.flags,
        "statement": ("the stems perceptually reconstruct this master"
                      if coh.verified else
                      "the stems do NOT verifiably reconstruct this master"),
    }


@app.post("/check/logic-master")
async def check_logic_master(
    master: UploadFile = File(...),
    stems: list[UploadFile] = File(...),
    project: UploadFile | None = File(None),
    project_files: list[UploadFile] | None = File(None),
    project_paths: list[str] | None = Form(None),
):
    """Standalone same-origin analysis: logicx vs stems vs master. Ephemeral."""
    if len(stems) < 2:
        raise HTTPException(status_code=422, detail="upload at least two stems")
    project_zip, _ = await _project_zip_from_upload(project, project_files, project_paths)
    master_data = (master.filename or "master.wav", await master.read())
    stem_data = [(s.filename or f"stem_{i}.wav", await s.read())
                 for i, s in enumerate(stems)]
    try:
        report = same_origin.analyze(project_zip, stem_data, master_data)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if "error" in report:
        raise HTTPException(status_code=422, detail=report["error"])
    return report


@app.post("/product/link")
async def product_link(request: Request, audio_file: UploadFile = File(...)):
    """Link any audio copy back to a registered record.

    Tries audiowmark payload (exact), chromaprint similarity (fuzzy), and
    embedded C2PA manifest (lossless copies only).
    """
    data = await audio_file.read()
    try:
        return verify.link(
            data, audio_file.filename or "audio",
            ip=request.headers.get("x-forwarded-for",
                                   request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/product/records/{record_id}/manifest")
def product_record_manifest(record_id: str):
    """Signed C2PA manifest definition for a sealed record.

    TODO(auth): OPEN IN THE POC. In production this must be gated — the
    manifest reveals stem/session SHA-256s and verification internals, and
    an open endpoint lets anyone with a record id pull them. Tracked in
    SECURITY-TODO.md.
    """
    try:
        row = pg.record_manifest(record_id)
    except Exception:
        raise HTTPException(status_code=422, detail="bad record id")
    if row is None:
        raise HTTPException(status_code=404, detail="no such record")
    return {
        "record_id": str(row["id"]),
        "artist": row["artist_name"],
        "sealed_at_utc": row["sealed_at"].isoformat(timespec="seconds"),
        "cert_subject": row["cert_subject"],
        "self_attested": bool(row["signer_self_attested"]),
        "manifest": row["manifest"],
    }


# In-memory register of waterprint runs (POC lab state, newest 40 kept).
_WATERPRINT_RUNS: dict[str, dict] = {}


def _chain_params(codec, bitrate, loudnorm, sample_rate, generations):
    """Validate/normalize the shared platform-chain form fields."""
    if codec not in simulate.VALID_CODECS:
        raise HTTPException(status_code=422, detail=f"codec must be one of {sorted(simulate.VALID_CODECS)}")
    try:
        target = None if loudnorm == "none" else float(loudnorm)
        sr = None if sample_rate == "keep" else int(sample_rate)
    except ValueError:
        raise HTTPException(status_code=422, detail="bad loudnorm or sample_rate value")
    if target is not None and not -30.0 <= target <= -5.0:
        raise HTTPException(status_code=422, detail="loudnorm target must be between -30 and -5 LUFS")
    if sr is not None and sr not in (44100, 48000):
        raise HTTPException(status_code=422, detail="sample_rate must be keep, 44100 or 48000")
    return max(32, min(int(bitrate), 512)), target, sr, max(1, min(int(generations), 10))


def _store_download(src: Path, filename: str) -> dict:
    LAB_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    stored = LAB_DOWNLOADS / f"{uuid.uuid4().hex}{src.suffix}"
    shutil.copy2(src, stored)
    return {"url": f"/fingerprint/download/{stored.name}", "filename": filename,
            "size_bytes": stored.stat().st_size}


@app.post("/waterprint/run")
async def waterprint_run(
    audio_file: UploadFile = File(...),
    waterprint_id: str = Form(...),
    codec: str = Form("vorbis"),
    bitrate: int = Form(160),
    loudnorm: str = Form("-14"),
    sample_rate: str = Form("44100"),
    generations: int = Form(1),
):
    """Combined pipeline: embed watermark -> register fingerprint ->
    platform-compress -> recover both from the compressed copy.

    Returns a run_id; /waterprint/check verifies any other copy against
    this run's registered fingerprint + ID.
    """
    try:
        waterprint_id = str(uuid.UUID(waterprint_id.strip()))
    except ValueError:
        raise HTTPException(status_code=422, detail="waterprint_id must be a valid UUID")
    bitrate, loudnorm_target, sr, generations = _chain_params(
        codec, bitrate, loudnorm, sample_rate, generations)

    data = await audio_file.read()
    base = Path(audio_file.filename or "audio").stem or "audio"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = tmp / (Path(audio_file.filename or "audio").name or "audio")
        raw.write_bytes(data)
        try:
            norm = audio.decode_to_normalized_wav(raw, tmp / "norm.wav")
            # 1. Embed the watermark.
            x, srate = sf.read(norm, dtype="float64", always_2d=True)
            y, blocks = watermark.embed(x, waterprint_id)
            if blocks == 0:
                raise HTTPException(
                    status_code=422,
                    detail="audio must be at least 10 seconds long for the watermark")
            wm_wav = tmp / "watermarked.wav"
            sf.write(wm_wav, y, srate, subtype="PCM_16")

            # 2. Register the fingerprint of the watermarked master.
            fp_registered, fp_duration = fingerprint.compute_fingerprint(wm_wav)

            # 3. Platform-compress the watermarked file.
            processed_path, steps, commands = simulate.apply_chain(
                wm_wav, tmp, codec=codec, bitrate_kbps=bitrate,
                loudnorm_target=loudnorm_target, sample_rate=sr,
                generations=generations)

            # 4. Recover both from the compressed copy.
            proc_norm = audio.decode_to_normalized_wav(processed_path, tmp / "proc_norm.wav")
            fp_processed, _ = fingerprint.compute_fingerprint(proc_norm)
            fp_cmp = fingerprint.compare_detailed(fp_registered, fp_processed)
            xp, _ = sf.read(proc_norm, dtype="float64", always_2d=True)
            wm_result = watermark.extract(xp)

            wm_download = _store_download(wm_wav, f"{base}_waterprinted.wav")
            proc_download = _store_download(
                processed_path,
                f"{base}_{codec}_{bitrate}k_x{generations}{processed_path.suffix}"
                if codec != "none" else f"{base}_processed{processed_path.suffix}")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

    run_id = uuid.uuid4().hex
    _WATERPRINT_RUNS[run_id] = {"id": waterprint_id, "fp": fp_registered,
                                "created": time.time()}
    while len(_WATERPRINT_RUNS) > 40:
        del _WATERPRINT_RUNS[min(_WATERPRINT_RUNS, key=lambda k: _WATERPRINT_RUNS[k]["created"])]

    wm_ok = wm_result["found"] and wm_result["id"] == waterprint_id
    fp_ok = fp_cmp["similarity"] >= VERIFY_THRESHOLD
    return {
        "run_id": run_id,
        "id": waterprint_id,
        "registered": {
            "watermark_blocks": blocks,
            "fingerprint_frames": len(fp_registered),
            "fingerprint_seconds": fp_duration,
            "download": wm_download,
        },
        "chain": {"steps": steps, "ffmpeg_commands": commands},
        "processed": {"download": proc_download},
        "recovery": {
            "watermark": {
                "recovered": wm_ok,
                "extracted_id": wm_result.get("id"),
                "confidence": wm_result.get("confidence"),
                "source": wm_result.get("source"),
                "blocks_crc_valid": wm_result.get("blocks_crc_valid"),
                "blocks_analyzed": wm_result.get("blocks_analyzed"),
            },
            "fingerprint": {
                "recovered": fp_ok,
                "similarity": fp_cmp["similarity"],
                "threshold": VERIFY_THRESHOLD,
                "frame_errors": fp_cmp["frame_errors"],
            },
            "verdict": ("both recovered" if wm_ok and fp_ok
                        else "fingerprint only" if fp_ok
                        else "watermark only" if wm_ok
                        else "neither recovered"),
        },
    }


@app.post("/waterprint/check")
async def waterprint_check(
    run_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """Verify any audio copy against a waterprint run: blind watermark
    extraction plus fingerprint similarity against the registered print."""
    run = _WATERPRINT_RUNS.get(run_id.strip())
    if run is None:
        raise HTTPException(status_code=404, detail="unknown or expired run_id (runs are kept in memory)")
    data = await audio_file.read()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = tmp / (Path(audio_file.filename or "audio").name or "audio")
        raw.write_bytes(data)
        try:
            norm = audio.decode_to_normalized_wav(raw, tmp / "norm.wav")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        fp, _ = fingerprint.compute_fingerprint(norm)
        x, _ = sf.read(norm, dtype="float64", always_2d=True)
        wm_result = watermark.extract(x)

    fp_cmp = fingerprint.compare_detailed(run["fp"], fp)
    wm_ok = wm_result["found"] and wm_result["id"] == run["id"]
    fp_ok = fp_cmp["similarity"] >= VERIFY_THRESHOLD
    return {
        "run_id": run_id,
        "expected_id": run["id"],
        "watermark": {
            "recovered": wm_ok,
            "extracted_id": wm_result.get("id"),
            "id_matches": wm_result.get("id") == run["id"],
            "confidence": wm_result.get("confidence"),
            "blocks_crc_valid": wm_result.get("blocks_crc_valid"),
            "blocks_analyzed": wm_result.get("blocks_analyzed"),
        },
        "fingerprint": {
            "recovered": fp_ok,
            "similarity": fp_cmp["similarity"],
            "threshold": VERIFY_THRESHOLD,
            "frame_errors": fp_cmp["frame_errors"],
        },
        "verdict": ("both recovered" if wm_ok and fp_ok
                    else "fingerprint only" if fp_ok
                    else "watermark only" if wm_ok
                    else "neither recovered"),
    }


@app.get("/fingerprint/download/{name}")
def fingerprint_download(name: str):
    """Download a processed file produced by /fingerprint/simulate (kept ~2h)."""
    if not re.fullmatch(r"[0-9a-f]{32}\.(ogg|m4a|mp3|opus|wav)", name):
        raise HTTPException(status_code=404, detail="unknown file")
    path = LAB_DOWNLOADS / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="expired or unknown file")
    return FileResponse(path, media_type=_LAB_MEDIA_TYPES[path.suffix])
