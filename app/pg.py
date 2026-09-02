"""Postgres access shared by the SessionSeal services.

This module is duplicated (identical) in heimdall, odin, and thor — the
services are separate repos and the file is small; keep edits in sync.
Connection comes from the service's .env (DATABASE_URL). Until auth
exists, records are owned by a seed user (poc@sessionseal.local).
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.environ["DATABASE_URL"]
POC_USER_EMAIL = "poc@sessionseal.local"

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        # check= validates connections at checkout, so a DB restart costs a
        # reconnect instead of one failed request per stale pooled conn.
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5,
                               kwargs={"row_factory": dict_row}, open=True,
                               check=ConnectionPool.check_connection)
    return _pool


def poc_user_id() -> str:
    """Get-or-create the seed user that owns records until auth exists."""
    with pool().connection() as conn:
        row = conn.execute(
            """insert into users (email, name, signup_source)
               values (%s, 'POC seed user', 'OTHER')
               on conflict (email) do update set updated_at = now()
               returning id""",
            (POC_USER_EMAIL,)).fetchone()
        return str(row["id"])


def pack_fingerprint(fp: list[int]) -> bytes:
    """Chromaprint raw values -> packed uint32 little-endian."""
    return np.asarray(fp, dtype=np.int64).astype(np.uint32).tobytes()


def unpack_fingerprint(data: bytes) -> list[int]:
    return np.frombuffer(data, dtype="<u4").tolist()


# ---------------------------------------------------------------------------
# intake (heimdall)
# ---------------------------------------------------------------------------

def create_record_and_job(record_id: str, artist: str, payload: dict) -> None:
    """QUEUED record + SEAL job in one transaction. The record id is minted
    by the caller because it doubles as the watermark payload."""
    with pool().connection() as conn:
        conn.execute(
            """insert into records (id, user_id, artist_name, status,
                                    watermark_payload)
               values (%s, %s, %s, 'QUEUED', %s)""",
            (record_id, poc_user_id(), artist, record_id.replace("-", "")))
        conn.execute(
            """insert into jobs (kind, status, record_id, user_id, payload)
               values ('SEAL', 'QUEUED', %s, %s, %s)""",
            (record_id, poc_user_id(), Jsonb(payload)))


def job_for_record(record_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute(
            """select id, status, attempts, error, result
                 from jobs where record_id = %s and kind = 'SEAL'
                order by queued_at desc limit 1""",
            (record_id,)).fetchone()


# ---------------------------------------------------------------------------
# worker (thor)
# ---------------------------------------------------------------------------

def claim_next_job(worker_id: str) -> dict | None:
    """Atomically claim the oldest QUEUED SEAL job (SKIP LOCKED)."""
    with pool().connection() as conn:
        job = conn.execute(
            """update jobs
                  set status = 'PROCESSING', attempts = attempts + 1,
                      started_at = now(), worker_id = %s
                where id = (select id from jobs
                             where status = 'QUEUED' and kind = 'SEAL'
                             order by queued_at
                             for update skip locked
                             limit 1)
               returning *""",
            (worker_id,)).fetchone()
        if job:
            conn.execute(
                "update records set status = 'PROCESSING' where id = %s",
                (job["record_id"],))
        return job


def complete_job(job_id, result: dict) -> None:
    with pool().connection() as conn:
        conn.execute(
            """update jobs set status = 'SUCCEEDED', finished_at = now(),
                      result = %s where id = %s""",
            (Jsonb(result), job_id))


def fail_job(job_id, record_id, error: str, retry: bool) -> None:
    with pool().connection() as conn:
        if retry:
            conn.execute(
                """update jobs set status = 'QUEUED', error = %s
                    where id = %s""", (error, job_id))
            conn.execute(
                "update records set status = 'QUEUED' where id = %s",
                (record_id,))
        else:
            conn.execute(
                """update jobs set status = 'FAILED', finished_at = now(),
                          error = %s where id = %s""", (error, job_id))
            conn.execute(
                """update records set status = 'FAILED', error = %s
                    where id = %s""", (error, record_id))


def update_record_sealed(*, record_id: str, sealed_at: str,
                         selfcheck: float, fp_raw: list[int],
                         fp_seconds: float, coherence: dict, sameorigin: dict,
                         master_sha: str, release_master_sha: str,
                         project_sha: str, manifest: dict,
                         cert_subject: str, signed_asset_path: str,
                         manifest_s3_key: str | None = None,
                         manifest_public_url: str | None = None) -> None:
    with pool().connection() as conn:
        conn.execute(
            """update records set
                 status = 'SEALED', daw = 'LOGIC_PRO',
                 watermark_selfcheck = %s,
                 fingerprint = %s, fingerprint_seconds = %s,
                 coherence_verified = %s, coherence_confidence = %s,
                 sameorigin_score = %s, sameorigin_band = %s,
                 sameorigin_report = %s, red_flags = %s,
                 master_sha256 = %s, release_master_sha256 = %s,
                 project_sha256 = %s, manifest = %s, cert_subject = %s,
                 manifest_s3_key = %s, manifest_public_url = %s,
                 sealed_at = %s, meta = meta || %s
               where id = %s""",
            (selfcheck, pack_fingerprint(fp_raw), fp_seconds,
             coherence["verified"], coherence["confidence"],
             sameorigin["score"], sameorigin["band"].upper(),
             Jsonb(sameorigin), Jsonb(sameorigin.get("red_flags", [])),
             master_sha, release_master_sha, project_sha,
             Jsonb(manifest), cert_subject,
             manifest_s3_key, manifest_public_url, sealed_at,
             Jsonb({"signed_asset_path": signed_asset_path}), record_id))


def insert_asset(*, owner_user_id: str | None, record_id: str, kind: str,
                 bucket: str, key: str, position: int = 0,
                 status: str = "ATTACHED", original_filename: str | None = None,
                 content_type: str | None = None, size_bytes: int | None = None,
                 sha256: str | None = None) -> None:
    with pool().connection() as conn:
        conn.execute(
            """insert into assets (owner_user_id, record_id, kind, position,
                                   status, s3_bucket, s3_key,
                                   original_filename, content_type,
                                   size_bytes, sha256)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               on conflict (s3_bucket, s3_key) do nothing""",
            (owner_user_id, record_id, kind, position, status, bucket, key,
             original_filename, content_type, size_bytes, sha256))


def mark_record_assets_attached(record_id: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            """update assets set status = 'ATTACHED'
                where record_id = %s and status = 'UPLOADED'""",
            (record_id,))


# ---------------------------------------------------------------------------
# verify (odin) — read-only
# ---------------------------------------------------------------------------

def sealed_records() -> list[dict]:
    """All sealed records with unpacked fingerprints, for link's scan."""
    with pool().connection() as conn:
        rows = conn.execute(
            """select id, artist_name, sealed_at, watermark_payload,
                      fingerprint, coherence_verified, coherence_confidence,
                      sameorigin_score, sameorigin_band, cert_subject,
                      manifest_public_url
                 from records
                where status = 'SEALED' and deleted_at is null""").fetchall()
    for r in rows:
        r["id"] = str(r["id"])
        r["fingerprint_raw"] = unpack_fingerprint(r["fingerprint"]) if r["fingerprint"] else []
    return rows


def record_manifest(record_id: str) -> dict | None:
    with pool().connection() as conn:
        row = conn.execute(
            """select id, artist_name, sealed_at, cert_subject,
                      signer_self_attested, manifest
                 from records
                where id = %s and deleted_at is null""",
            (record_id,)).fetchone()
    return row


def insert_verification(*, source: str, upload_filename: str | None,
                        upload_sha256: str | None, linked: bool,
                        linked_via: str | None, matched_record_id: str | None,
                        copy_attack_suspected: bool, watermark_found: bool,
                        watermark_payload: str | None,
                        watermark_score: float | None,
                        watermark_corroboration: float | None,
                        fingerprint_best_similarity: float | None,
                        c2pa_manifest_present: bool | None,
                        c2pa_validation_state: str | None,
                        mechanisms: dict, records_scanned: int,
                        duration_ms: int, ip: str | None,
                        user_agent: str | None) -> None:
    """Append-only log of a /product/link attempt (used by odin)."""
    with pool().connection() as conn:
        conn.execute(
            """insert into verifications
                 (source, upload_filename, upload_sha256, linked, linked_via,
                  matched_record_id, copy_attack_suspected, watermark_found,
                  watermark_payload, watermark_score, watermark_corroboration,
                  fingerprint_best_similarity, c2pa_manifest_present,
                  c2pa_validation_state, mechanisms, records_scanned,
                  duration_ms, ip, user_agent)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s)""",
            (source, upload_filename, upload_sha256, linked, linked_via,
             matched_record_id, copy_attack_suspected, watermark_found,
             watermark_payload, watermark_score, watermark_corroboration,
             fingerprint_best_similarity, c2pa_manifest_present,
             c2pa_validation_state, Jsonb(mechanisms), records_scanned,
             duration_ms, ip, user_agent))
