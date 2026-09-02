"""fpcalc (Chromaprint) wrapper and fingerprint similarity.

The perceptual fingerprint summarizes what audio *sounds like*, so it
survives re-encoding that changes every byte. It is used two ways:
comparing the stem mixdown against the master (coherence), and re-linking
a re-encoded copy back to its record (verify).

fpcalc -raw -json outputs {"duration": float, "fingerprint": [uint32, ...]}
where each 32-bit int encodes ~0.12 s of chroma features. Similarity between
two fingerprints is 1 minus the normalized bit error rate over the best
alignment found by a small offset search.
"""

import json
import subprocess
from pathlib import Path

# Restrict fpcalc analysis window. Long enough for full tracks in this MVP.
MAX_ANALYSIS_SECONDS = 600

# How far (in fingerprint frames, ~0.12 s each) the alignment search slides.
MAX_OFFSET = 80

# Minimum overlapping frames required to call a comparison meaningful.
MIN_OVERLAP = 16


def compute_fingerprint(path: Path) -> tuple[list[int], float]:
    """Return (raw fingerprint ints, duration in seconds) for an audio file."""
    result = subprocess.run(
        ["fpcalc", "-raw", "-json", "-length", str(MAX_ANALYSIS_SECONDS), str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fpcalc failed on {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    return data["fingerprint"], float(data["duration"])


def compute_compressed_fingerprint(path: Path) -> str:
    """Return Chromaprint's compact base64 fingerprint string (for the manifest)."""
    result = subprocess.run(
        ["fpcalc", "-json", "-length", str(MAX_ANALYSIS_SECONDS), str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fpcalc failed on {path.name}: {result.stderr.strip()}")
    return json.loads(result.stdout)["fingerprint"]


def compare_detailed(fp_a: list[int], fp_b: list[int], buckets: int = 160) -> dict:
    """Similarity plus a per-frame error profile at the best alignment.

    frame_errors is downsampled to <= `buckets` values in [0, 1] (fraction
    of differing bits per ~0.12 s frame) for visualization.
    """
    empty = {"similarity": 0.0, "offset_frames": 0, "frames_compared": 0,
             "frame_errors": []}
    if not fp_a or not fp_b:
        return empty
    best, best_off = 0.0, None
    for offset in range(-MAX_OFFSET, MAX_OFFSET + 1):
        a, b = (fp_a[offset:], fp_b) if offset >= 0 else (fp_a, fp_b[-offset:])
        n = min(len(a), len(b))
        if n < MIN_OVERLAP:
            continue
        errors = sum((x ^ y).bit_count() for x, y in zip(a[:n], b[:n]))
        score = 1.0 - errors / (32.0 * n)
        if score > best:
            best, best_off = score, offset
    if best_off is None:
        return empty
    a, b = (fp_a[best_off:], fp_b) if best_off >= 0 else (fp_a, fp_b[-best_off:])
    n = min(len(a), len(b))
    errs = [(x ^ y).bit_count() / 32.0 for x, y in zip(a[:n], b[:n])]
    step = max(1, -(-len(errs) // buckets))  # ceil division
    downsampled = [
        round(sum(errs[i:i + step]) / len(errs[i:i + step]), 4)
        for i in range(0, len(errs), step)
    ]
    return {"similarity": round(best, 4), "offset_frames": best_off,
            "frames_compared": n, "frame_errors": downsampled}


def similarity(fp_a: list[int], fp_b: list[int]) -> float:
    """Similarity in [0, 1]: 1 - normalized bit error rate at the best offset."""
    if not fp_a or not fp_b:
        return 0.0
    best = 0.0
    for offset in range(-MAX_OFFSET, MAX_OFFSET + 1):
        if offset >= 0:
            a, b = fp_a[offset:], fp_b
        else:
            a, b = fp_a, fp_b[-offset:]
        n = min(len(a), len(b))
        if n < MIN_OVERLAP:
            continue
        errors = sum((x ^ y).bit_count() for x, y in zip(a[:n], b[:n]))
        score = 1.0 - errors / (32.0 * n)
        if score > best:
            best = score
    return best
