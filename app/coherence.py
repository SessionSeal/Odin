"""Coherence check: do the uploaded stems verifiably make the master?

Mix the stems back down (sum, align, peak-normalize) and compare the
mixdown to the master *by sound* (perceptual fingerprint), because
mastering means the bytes never match exactly. This is the guard that
stops the platform from sealing unrelated or stolen audio next to a master.

Anti-gaming guards:
  - a stem that is near-identical to the master alone suggests the master
    was re-uploaded disguised as a stem;
  - stems that are near-identical to each other suggest padding with
    duplicates rather than real ingredients.

If confidence is below threshold the pipeline still runs, but the record
states coherence was NOT verified and never implies the stems make the
master.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from . import fingerprint

# Mixdown-vs-master similarity at or above this counts as verified.
COHERENCE_THRESHOLD = 0.85
# A stem this similar to the master looks like the master in disguise.
STEM_VS_MASTER_LIMIT = 0.95
# Two stems this similar to each other look like duplicates.
STEM_DUPLICATE_LIMIT = 0.98

METHOD = "stem-mixdown chromaprint similarity (1 - bit error rate, offset search)"


@dataclass
class CoherenceResult:
    verified: bool
    confidence: float
    method: str = METHOD
    flags: list[str] = field(default_factory=list)


def mixdown(stem_wavs: list[Path], out_path: Path) -> Path:
    """Sum normalized stem WAVs into one peak-normalized mixdown WAV."""
    arrays = []
    rate = None
    for p in stem_wavs:
        data, sr = sf.read(p, dtype="float64", always_2d=True)
        if rate is None:
            rate = sr
        elif sr != rate:
            raise RuntimeError(f"stem {p.name} sample rate {sr} != {rate}")
        arrays.append(data)

    length = max(a.shape[0] for a in arrays)
    total = np.zeros((length, arrays[0].shape[1]), dtype=np.float64)
    for a in arrays:
        total[: a.shape[0], :] += a

    peak = np.max(np.abs(total))
    if peak > 0:
        total = total / peak * 0.891  # about -1 dBFS of headroom
    sf.write(out_path, total, rate, subtype="PCM_16")
    return out_path


def check_coherence(
    master_wav: Path, stem_wavs: list[Path], workdir: Path
) -> CoherenceResult:
    """Compare the stem mixdown to the master by perceptual fingerprint."""
    mix_path = mixdown(stem_wavs, workdir / "mixdown.wav")

    master_fp, _ = fingerprint.compute_fingerprint(master_wav)
    mix_fp, _ = fingerprint.compute_fingerprint(mix_path)
    stem_fps = [fingerprint.compute_fingerprint(p)[0] for p in stem_wavs]

    confidence = fingerprint.similarity(mix_fp, master_fp)

    flags: list[str] = []
    for i, sfp in enumerate(stem_fps):
        if fingerprint.similarity(sfp, master_fp) >= STEM_VS_MASTER_LIMIT:
            flags.append(
                f"stem {i} is near-identical to the master "
                "(possible master uploaded disguised as a stem)"
            )
    for i in range(len(stem_fps)):
        for j in range(i + 1, len(stem_fps)):
            if fingerprint.similarity(stem_fps[i], stem_fps[j]) >= STEM_DUPLICATE_LIMIT:
                flags.append(f"stems {i} and {j} are near-identical to each other")

    verified = confidence >= COHERENCE_THRESHOLD and not flags
    return CoherenceResult(verified=verified, confidence=confidence, flags=flags)
