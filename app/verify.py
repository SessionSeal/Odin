"""Odin — verification: link any audio copy back to a sealed record.

Tries all three mechanisms and applies strict resolution rules:
a watermark hit must be corroborated by the record's own fingerprint
(otherwise it's reported as a suspected copy attack), else fingerprint
similarity, else not linked. Read-only against Postgres.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import audio, fingerprint, pg

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
WATERMARK_KEY = Path(os.environ.get(
    "WATERMARK_KEY_PATH", _SERVICE_ROOT / "signing" / "watermark.key"))

FP_LINK_THRESHOLD = 0.80
AW_SCORE_THRESHOLD = 0.7
# A watermark hit must be corroborated by the linked record's own fingerprint.
# Unrelated audio scores ~0.67 against any chromaprint (baseline noise);
# genuine re-encoded copies score ~0.99. Below this, a decoded payload is
# treated as a transplanted (copy-attack) watermark, not a link.
WM_CORROBORATION_THRESHOLD = 0.75


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def aw_extract(wav: Path) -> tuple[str | None, float]:
    r = _run(["audiowmark", "get", "--key", str(WATERMARK_KEY), str(wav)])
    best_hex, best_score = None, -1.0
    for line in r.stdout.splitlines():
        m = re.match(r"pattern\s+\S+\s+([0-9a-f]{32})\s+([\d.]+)", line.strip())
        if m and float(m.group(2)) > best_score:
            best_hex, best_score = m.group(1), float(m.group(2))
    return best_hex, best_score


def record_public(row) -> dict:
    return {
        "record_id": str(row["id"]),
        "artist": row["artist_name"],
        "registered_at_utc": row["sealed_at"].isoformat(timespec="seconds"),
        "coherence": {"verified": bool(row["coherence_verified"]),
                      "confidence": float(row["coherence_confidence"])},
        "same_origin": {"score": float(row["sameorigin_score"]),
                        "band": (row["sameorigin_band"] or "").lower()},
        "signer": {"cert_subject": row["cert_subject"], "self_attested": True},
        "manifest_url": row.get("manifest_public_url"),
        "proves": "custody, integrity, coherence, priority",
        "does_not_prove": "authorship",
    }


def link(data: bytes, filename: str, source: str = "WEB",
         ip: str | None = None, user_agent: str | None = None) -> dict:
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = tmp / (Path(filename or "audio").name or "audio")
        raw.write_bytes(data)
        norm = audio.decode_to_normalized_wav(raw, tmp / "norm.wav")

        wm_hex, wm_score = aw_extract(norm)
        query_fp, _ = fingerprint.compute_fingerprint(norm)

        c2pa = {"manifest_present": False}
        r = _run(["c2patool", str(raw)])
        if r.returncode == 0 and r.stdout.strip().startswith("{"):
            try:
                rep = json.loads(r.stdout)
                failures = (rep.get("validation_results", {})
                            .get("activeManifest", {}).get("failure", []))
                c2pa = {
                    "manifest_present": True,
                    "validation_state": rep.get("validation_state"),
                    "failures": [f.get("code") for f in failures],
                    "note": "signingCredential.untrusted is expected "
                            "(self-attested signer)",
                }
            except json.JSONDecodeError:
                pass
        if not c2pa["manifest_present"]:
            c2pa["note"] = ("no embedded manifest -- platform re-encoding "
                            "strips it; that is why the remote record plus "
                            "watermark/fingerprint recovery exist")

    rows = pg.sealed_records()

    wm_record = None
    if wm_hex and wm_score >= AW_SCORE_THRESHOLD:
        wm_record = next((r for r in rows if r["watermark_payload"] == wm_hex), None)

    best_row, best_sim = None, 0.0
    for row in rows:
        sim = fingerprint.similarity(query_fp, row["fingerprint_raw"])
        if sim > best_sim:
            best_row, best_sim = row, sim
    fp_record = best_row if best_sim >= FP_LINK_THRESHOLD else None

    # Corroborate a watermark hit against that record's own fingerprint:
    # a valid payload inside audio that sounds nothing like the record is a
    # transplanted watermark (copy attack), not a link.
    copy_attack_suspected = False
    wm_corroboration = None
    if wm_record is not None:
        wm_corroboration = fingerprint.similarity(
            query_fp, wm_record["fingerprint_raw"])
        if wm_corroboration < WM_CORROBORATION_THRESHOLD:
            copy_attack_suspected = True
            wm_record = None

    linked = wm_record or fp_record

    mechanisms_out = {
        "watermark": {
            "detected": wm_record is not None or copy_attack_suspected,
            "payload_hex": wm_hex,
            "score": round(wm_score, 2) if wm_score > 0 else None,
            "matched_registered_record": wm_record is not None,
            "fingerprint_corroboration":
                round(wm_corroboration, 4)
                if wm_corroboration is not None else None,
            "note": None if wm_record else
                    "payload matches a registered record, but the audio "
                    "does not match that record's fingerprint -- "
                    "consistent with a copied/transplanted watermark; "
                    "not linked" if copy_attack_suspected else
                    "no registered record carries this payload"
                    if wm_hex and wm_score >= AW_SCORE_THRESHOLD else
                    "no confident watermark pattern found",
        },
        "fingerprint": {
            "best_similarity": round(best_sim, 4),
            "threshold": FP_LINK_THRESHOLD,
            "matched": fp_record is not None,
        },
        "c2pa": c2pa,
    }

    # Append-only audit log — a failed insert must never fail verification.
    try:
        pg.insert_verification(
            source=source,
            upload_filename=Path(filename or "audio").name,
            upload_sha256=hashlib.sha256(data).hexdigest(),
            linked=linked is not None,
            linked_via=("WATERMARK" if wm_record
                        else "FINGERPRINT" if fp_record else None),
            matched_record_id=linked["id"] if linked is not None else None,
            copy_attack_suspected=copy_attack_suspected,
            watermark_found=bool(wm_hex and wm_score >= AW_SCORE_THRESHOLD),
            watermark_payload=wm_hex,
            watermark_score=round(wm_score, 3) if wm_score > 0 else None,
            watermark_corroboration=wm_corroboration,
            fingerprint_best_similarity=round(best_sim, 4),
            c2pa_manifest_present=c2pa.get("manifest_present"),
            c2pa_validation_state=c2pa.get("validation_state"),
            mechanisms=mechanisms_out,
            records_scanned=len(rows),
            duration_ms=int((time.monotonic() - t0) * 1000),
            ip=ip, user_agent=user_agent)
    except Exception as e:
        print(f"[odin] verification log failed: {e}", flush=True)

    return {
        "linked": linked is not None,
        "linked_via": ("watermark (exact id, fingerprint-corroborated)"
                       if wm_record
                       else "fingerprint similarity" if fp_record else None),
        "record": record_public(linked) if linked is not None else None,
        "copy_attack_suspected": copy_attack_suspected,
        "mechanisms": mechanisms_out,
        "registered_records": len(rows),
    }
