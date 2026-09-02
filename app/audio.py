"""ffmpeg decode helper.

Everything downstream (mixdown, fingerprinting) operates on one normalized
form -- 44100 Hz, stereo, 16-bit PCM WAV -- so comparisons are apples to
apples regardless of what format was uploaded.
"""

import subprocess
from pathlib import Path

SAMPLE_RATE = 44100
CHANNELS = 2


def decode_to_normalized_wav(src: Path, dst: Path) -> Path:
    """Decode any audio file to normalized PCM WAV. Raises on failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {src.name}: {result.stderr.strip()}")
    return dst
