"""Streaming-platform processing simulation for the /fingerprint lab.

Applies the transforms a distribution platform applies to uploaded masters
-- lossy codecs (Ogg Vorbis like Spotify, AAC like Apple Music/YouTube,
MP3, Opus), EBU R128 loudness normalization (e.g. -14 LUFS), sample-rate
conversion, optional generational re-encoding -- then the caller measures
how well the chromaprint fingerprint survives the chain.

Every ffmpeg command used is returned in the report, so the simulation is
transparent and reproducible. Note: some platforms apply loudness
normalization as playback gain rather than re-rendering the file;
simulating it as a render step is the conservative (harsher) test.
"""

import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

VALID_CODECS = {"vorbis", "aac", "mp3", "opus", "none"}
CODEC_EXT = {"vorbis": ".ogg", "aac": ".m4a", "mp3": ".mp3", "opus": ".opus"}


@lru_cache(maxsize=1)
def _encoder_list() -> str:
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout


def _aac_encoder() -> str:
    # aac_at is Apple's AudioToolbox encoder -- the closest thing to what
    # Apple Music actually uses; fall back to ffmpeg's native aac.
    return "aac_at" if " aac_at " in _encoder_list() else "aac"


def _have_oggenc() -> bool:
    return shutil.which("oggenc") is not None


def _codec_args(codec: str, bitrate_kbps: int) -> list[str]:
    if codec == "vorbis":
        # ffmpeg's native vorbis encoder is gated behind -strict experimental
        # and is notably worse than libvorbis; only used if oggenc is absent.
        return ["-c:a", "vorbis", "-strict", "-2", "-b:a", f"{bitrate_kbps}k"]
    if codec == "aac":
        return ["-c:a", _aac_encoder(), "-b:a", f"{bitrate_kbps}k"]
    if codec == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k"]
    if codec == "opus":
        return ["-c:a", "libopus", "-b:a", f"{bitrate_kbps}k"]
    raise ValueError(f"unknown codec: {codec}")


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg step failed: {res.stderr.strip()[-400:]}")


def probe(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json",
         str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return {"error": res.stderr.strip()[:300]}
    data = json.loads(res.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    return {
        "container": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "duration_seconds": round(float(fmt["duration"]), 3) if fmt.get("duration") else None,
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
    }


def measure_loudness(path: Path) -> dict | None:
    """EBU R128 measurement using loudnorm's analysis pass (JSON on stderr)."""
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr, re.S)
    if not m:
        return None
    d = json.loads(m.group())

    def num(key):
        try:
            v = float(d[key])
            return None if v in (float("inf"), float("-inf")) else round(v, 2)
        except (KeyError, ValueError):
            return None

    return {
        "integrated_lufs": num("input_i"),
        "true_peak_dbtp": num("input_tp"),
        "loudness_range_lu": num("input_lra"),
    }


def apply_chain(
    src: Path,
    workdir: Path,
    *,
    codec: str,
    bitrate_kbps: int,
    loudnorm_target: float | None,
    sample_rate: int | None,
    generations: int = 1,
) -> tuple[Path, list[dict], list[str]]:
    """Run the selected platform chain. Returns (output, steps, commands)."""
    steps: list[dict] = []
    commands: list[str] = []

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]

    if loudnorm_target is not None:
        cmd += ["-af", f"loudnorm=I={loudnorm_target}:TP=-1.0:LRA=11"]
        steps.append({
            "step": "loudness normalization",
            "detail": f"EBU R128 loudnorm to {loudnorm_target} LUFS, true peak "
                      "-1.0 dBTP (single pass)",
        })

    sr = sample_rate
    if codec == "opus" and sr != 48000:
        sr = 48000  # libopus only encodes at 48 kHz family rates
        steps.append({"step": "resample", "detail": "48000 Hz (required by Opus)"})
    elif sr:
        steps.append({"step": "resample", "detail": f"{sr} Hz"})
    if sr:
        cmd += ["-ar", str(sr)]
    elif loudnorm_target is not None:
        # loudnorm resamples to 192 kHz internally; pin back to the source rate
        orig_sr = probe(src).get("sample_rate")
        if orig_sr:
            cmd += ["-ar", str(orig_sr)]

    use_oggenc = codec == "vorbis" and _have_oggenc()

    if codec == "none":
        out = workdir / "processed.wav"
        cmd += ["-c:a", "pcm_s16le", str(out)]
        _run(cmd)
        commands.append(" ".join(cmd))
    elif use_oggenc:
        # Real libvorbis via oggenc (what Spotify-grade Vorbis actually is);
        # ffmpeg applies loudnorm/resample first, oggenc encodes.
        stage = workdir / "stage.wav"
        cmd += ["-c:a", "pcm_s16le", str(stage)]
        _run(cmd)
        commands.append(" ".join(cmd))
        out = workdir / "processed.ogg"
        enc_cmd = ["oggenc", "-Q", "-b", str(bitrate_kbps), "-o", str(out), str(stage)]
        _run(enc_cmd)
        commands.append(" ".join(enc_cmd))
        steps.append({"step": "lossy encode",
                      "detail": f"vorbis @ {bitrate_kbps} kbps (libvorbis via oggenc)"})
    else:
        out = workdir / f"processed{CODEC_EXT[codec]}"
        label = f"{codec} @ {bitrate_kbps} kbps"
        if codec == "aac" and _aac_encoder() == "aac_at":
            label += " (Apple AudioToolbox encoder)"
        steps.append({"step": "lossy encode", "detail": label})
        cmd += _codec_args(codec, bitrate_kbps) + [str(out)]
        _run(cmd)
        commands.append(" ".join(cmd))

    if codec != "none" and generations > 1:
        current = out
        for g in range(2, generations + 1):
            nxt = workdir / f"processed_gen{g}{CODEC_EXT[codec]}"
            if use_oggenc:
                stage_g = workdir / f"stage_gen{g}.wav"
                dec = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(current),
                       "-c:a", "pcm_s16le", str(stage_g)]
                _run(dec)
                enc = ["oggenc", "-Q", "-b", str(bitrate_kbps), "-o", str(nxt), str(stage_g)]
                _run(enc)
                commands += [" ".join(dec), " ".join(enc)]
            else:
                cmd_g = (["ffmpeg", "-y", "-loglevel", "error", "-i", str(current)]
                         + _codec_args(codec, bitrate_kbps) + [str(nxt)])
                _run(cmd_g)
                commands.append(" ".join(cmd_g))
            current = nxt
        steps.append({
            "step": "generational re-encode",
            "detail": f"decoded and re-encoded {codec} @ {bitrate_kbps} kbps "
                      f"{generations - 1} more times ({generations} total encode "
                      "passes — simulates repeated rip-and-reupload)",
        })
        out = current

    if not steps:
        steps.append({"step": "none", "detail": "no processing selected; file re-muxed as WAV"})
    return out, steps, commands
