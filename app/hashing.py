"""SHA-256 hashing: the sealed byte-level commitments.

A hash lets anyone later confirm a produced file is byte-for-byte identical
to what was sealed at upload time.
"""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
