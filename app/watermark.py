"""Invisible audio watermarking via time-spread echo hiding (POC).

Payload: a UUID (128 bits) framed as [16-bit sync][128-bit uuid][16-bit
CRC-16/CCITT] = 160 bits, embedded once per 10-second block. Each bit
occupies ~56 ms; a bit is written by adding a quiet *time-spread* echo:
instead of one echo tap, the echo is spread over a 192-chip pseudo-noise
(+-1) sequence starting at one of two base delays (bit 0 -> ~3 ms, bit 1
-> ~9 ms), with the opposite-delay spread echo subtracted (bipolar
embedding doubles detector separation). Total added energy sits around
-30 dB relative to the signal -- inaudible as anything but a whisper of
diffuse coloration.

Why time-spread: plain echo hiding fails on music because note pitch
periods create cepstral peaks exactly where single-echo detection looks.
Despreading the cepstrum against the PN sequence decorrelates pitch
structure and adds ~sqrt(192) of processing gain. The PN seed acts as a
detection key: without it, despreading yields noise.

Extraction is blind: per ~56 ms segment, Hann-window -> log spectrum ->
real cepstrum -> correlate the two delay regions against the PN sequence;
the sign of the difference is the bit, its magnitude the confidence.
Codec padding shifts audio, so a two-stage start-offset search aligns the
grid. Frames failing CRC go through chase decoding (flip subsets of the
lowest-confidence bits until sync+CRC validate), and multi-block files
additionally soft-vote every bit across blocks.

Robust to: lossy codecs (Vorbis/AAC/MP3/Opus, incl. repeated encodes),
gain changes / loudness normalization (gain lands in cepstrum bin 0),
resampling round-trips (extraction re-normalizes to 44.1 kHz).
Destroyed by: time-stretching, pitch-shifting, heavy overdubbing.
POC honesty: robustness is empirical, not guaranteed.
"""

import itertools
import uuid as uuid_mod

import numpy as np

SR = 44100
BLOCK = 10 * SR                 # one full payload frame every 10 s
MARGIN = SR // 2                # unmarked guard zone at block start
N_BITS = 160                    # 16 sync + 128 payload + 16 crc
BIT_LEN = (BLOCK - 2 * MARGIN) // N_BITS   # 2480 samples ~ 56 ms
ALPHA = 0.022                   # per-chip spread-echo amplitude
PN_LEN = 192                    # chips in the spread sequence
B0, B1 = 140, 384               # base delays (samples) for bit 0 / bit 1
PN_SEED = 0xC0FFEE              # detection key
RAMP = 200
SYNC = 0xB705
FFT_N = 4096
COARSE_STEP, FINE_STEP = 80, 10
MAX_OFFSET = 3 * BIT_LEN        # covers mp3/aac/opus codec padding
CHASE_WEAKEST = 26              # chase decoding: pool of least-confident bits
CHASE_MAX_FLIPS = 5

_PN = np.random.default_rng(PN_SEED).choice([-1.0, 1.0], size=PN_LEN)
_WIN = np.hanning(BIT_LEN)


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _bits_to_int(bits) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def frame_bits(id_str: str) -> list[int]:
    u = uuid_mod.UUID(id_str)
    return (_int_to_bits(SYNC, 16)
            + _int_to_bits(int.from_bytes(u.bytes, "big"), 128)
            + _int_to_bits(_crc16(u.bytes), 16))


# ---------------------------------------------------------------------------
# embedding

def _spread_signal(sig: np.ndarray, base_delay: int) -> np.ndarray:
    """FFT-convolve `sig` with the PN spread-echo kernel at `base_delay`."""
    klen = base_delay + PN_LEN
    kernel = np.zeros(klen)
    kernel[base_delay:] = _PN
    nfft = 1 << int(np.ceil(np.log2(len(sig) + klen)))
    out = np.fft.irfft(np.fft.rfft(sig, nfft) * np.fft.rfft(kernel, nfft), nfft)
    return out[:len(sig)]


def embed(x: np.ndarray, id_str: str) -> tuple[np.ndarray, int]:
    """Embed the ID into audio (float array [n, ch] at 44.1 kHz).

    Returns (watermarked audio, number of complete blocks embedded).
    """
    bits = frame_bits(id_str)
    n, ch = x.shape
    y = x.copy()

    spread0 = np.stack([_spread_signal(x[:, c], B0) for c in range(ch)], axis=1)
    spread1 = np.stack([_spread_signal(x[:, c], B1) for c in range(ch)], axis=1)

    env = np.ones(BIT_LEN)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, RAMP))
    env[:RAMP] = ramp
    env[-RAMP:] = ramp[::-1]
    env = env[:, None]

    blocks = 0
    for block_start in range(0, n, BLOCK):
        base = block_start + MARGIN
        if base + N_BITS * BIT_LEN > n:
            break
        for i, bit in enumerate(bits):
            s = base + i * BIT_LEN
            e = s + BIT_LEN
            pos, neg = (spread1, spread0) if bit else (spread0, spread1)
            y[s:e] += ALPHA * env * (pos[s:e] - neg[s:e])
        blocks += 1

    peak = np.max(np.abs(y))
    if peak > 0.999:
        y *= 0.999 / peak
    return y, blocks


# ---------------------------------------------------------------------------
# extraction

def _bit_scores(mono: np.ndarray, base: int) -> np.ndarray | None:
    """Despread echo score per bit segment; positive means bit 1."""
    if base < 0 or base + N_BITS * BIT_LEN > len(mono):
        return None
    segs = np.stack([mono[base + i * BIT_LEN: base + (i + 1) * BIT_LEN]
                     for i in range(N_BITS)])
    spec = np.abs(np.fft.rfft(segs * _WIN, n=FFT_N, axis=1))
    ceps = np.fft.irfft(np.log(spec + 1e-9), axis=1)
    # Fractional-sample shifts (resampling, loudnorm's internal 192 kHz trip)
    # smear each chip across neighboring quefrency bins; a light low-pass
    # recaptures that energy before despreading.
    smoothed = ceps.copy()
    smoothed[:, 1:-1] += 0.5 * (ceps[:, :-2] + ceps[:, 2:])
    smoothed *= 0.5
    return smoothed[:, B1:B1 + PN_LEN] @ _PN - smoothed[:, B0:B0 + PN_LEN] @ _PN


def _try_frame(bits) -> tuple[bool, int | None]:
    if _bits_to_int(bits[:16]) != SYNC:
        return False, None
    payload = _bits_to_int(bits[16:144])
    if _crc16(payload.to_bytes(16, "big")) != _bits_to_int(bits[144:]):
        return False, None
    return True, payload


# CRC-16 is affine over GF(2): crc(a ^ b) = crc(a) ^ crc(b) ^ crc(0).
# Precomputing the linear part of each payload bit's effect lets a chase
# trial validate with a handful of XORs instead of a full CRC pass.
_CRC_ZERO = _crc16(bytes(16))
_CRC_UNIT_LIN = [_crc16((1 << (127 - i)).to_bytes(16, "big")) ^ _CRC_ZERO
                 for i in range(128)]


def _chase_decode(scores: np.ndarray, max_flips: int = CHASE_MAX_FLIPS) -> tuple[int | None, int]:
    """Hard-decide, then flip subsets of the weakest bits until CRC passes.

    Returns (payload or None, number of flips used).
    """
    hard = (scores > 0).astype(int)
    sync_val = _bits_to_int(hard[:16])
    payload = _bits_to_int(hard[16:144])
    crc_val = _bits_to_int(hard[144:])
    base_lin = _crc16(payload.to_bytes(16, "big")) ^ _CRC_ZERO  # linear part

    if sync_val == SYNC and base_lin ^ _CRC_ZERO == crc_val:
        return payload, 0

    weakest = np.argsort(np.abs(scores))[:CHASE_WEAKEST].tolist()
    for r in range(1, max_flips + 1):
        for combo in itertools.combinations(weakest, r):
            s, p_mask, p_lin, c = sync_val, 0, base_lin, crc_val
            for idx in combo:
                if idx < 16:
                    s ^= 1 << (15 - idx)
                elif idx < 144:
                    p_mask ^= 1 << (143 - idx)
                    p_lin ^= _CRC_UNIT_LIN[idx - 16]
                else:
                    c ^= 1 << (159 - idx)
            if s == SYNC and p_lin ^ _CRC_ZERO == c:
                return payload ^ p_mask, r
    return None, 0


def _sync_hits(scores: np.ndarray) -> int:
    hard = (scores > 0).astype(int)
    return int(np.sum(hard[:16] == np.array(_int_to_bits(SYNC, 16))))


def extract(x: np.ndarray) -> dict:
    """Blind-extract a watermark ID from audio ([n] or [n, ch] at 44.1 kHz)."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = len(mono)
    n_blocks = n // BLOCK
    if n_blocks == 0:
        return {"found": False, "reason": "audio shorter than one 10 s watermark block",
                "blocks_analyzed": 0}

    # Stage 1: locate the constant codec-padding offset on the first block.
    # The sync/strength metric ranks candidates, but ranking alone can pick a
    # wrong basin -- so the real test is: does a frame actually decode there?
    # Try chase decoding on the top-ranked candidates and lock the first
    # offset that validates.
    def score_at(off):
        sc = _bit_scores(mono, MARGIN + off)
        if sc is None:
            return None
        return (_sync_hits(sc) / 16.0 + float(np.mean(np.abs(sc))), off, sc)

    candidates = [c for off in range(0, MAX_OFFSET, COARSE_STEP)
                  if (c := score_at(off)) is not None]
    if candidates:
        top_coarse = max(candidates)[1]
        for off in range(max(0, top_coarse - COARSE_STEP),
                         top_coarse + COARSE_STEP, FINE_STEP):
            if off % COARSE_STEP and (c := score_at(off)) is not None:
                candidates.append(c)
    candidates.sort(reverse=True, key=lambda c: c[0])

    best_off = candidates[0][1] if candidates else 0
    for _, off, sc in candidates[:10]:
        payload, _flips = _chase_decode(sc, max_flips=4)
        if payload is not None:
            best_off = off
            break

    # Stage 2: score every block at that offset.
    block_reports = []
    vote = np.zeros(N_BITS)
    payloads = []
    for b in range(n_blocks):
        scores = _bit_scores(mono, b * BLOCK + MARGIN + best_off)
        if scores is None:
            continue
        vote += scores
        payload, flips = _chase_decode(scores)
        if payload is not None:
            payloads.append(payload)
        block_reports.append({
            "block": b,
            "crc_valid": payload is not None,
            "chase_flips_used": flips,
            "sync_bits_matched": _sync_hits(scores),
            "mean_echo_strength": round(float(np.mean(np.abs(scores))), 4),
        })

    voted_payload, voted_flips = _chase_decode(vote)

    payload_int, source = None, None
    if payloads:
        payload_int = max(set(payloads), key=payloads.count)
        source = "per-block extraction (CRC-validated)"
    elif voted_payload is not None:
        payload_int = voted_payload
        source = f"cross-block soft voting (chase flips: {voted_flips})"

    n_valid = sum(1 for r in block_reports if r["crc_valid"])
    if payload_int is None:
        confidence = 0.0
    elif n_valid:
        confidence = round(n_valid / n_blocks, 3)
    else:
        confidence = 0.5  # voting-only recovery: real but weaker evidence

    return {
        "found": payload_int is not None,
        "id": str(uuid_mod.UUID(int=payload_int)) if payload_int is not None else None,
        "confidence": confidence,
        "source": source,
        "blocks_analyzed": n_blocks,
        "blocks_crc_valid": n_valid,
        "start_offset_samples": best_off,
        "blocks": block_reports,
    }
