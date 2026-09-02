"""Exhaustive best-effort inspector for zipped Logic Pro .logicx packages.

Extracts every piece of information that is practically parseable:

  - the zip itself: full entry list with sizes, timestamps, CRCs;
  - every property list (XML or binary), with NSKeyedArchiver graphs
    resolved into plain nested structures;
  - every media file: ffprobe format/stream data plus manual chunk-level
    decoding of WAV (fmt/bext/iXML/LIST-INFO/cue/labl/smpl/acid),
    AIFF (COMM/NAME/AUTH/ANNO/MARK/INST) and CAF (desc/info/chan/...);
  - every image, returned inline as a data URL with dimensions;
  - text/XML/JSON files, with content previews;
  - opaque binaries (ProjectData and friends): size, hash, entropy,
    header bytes, and all embedded ASCII/UTF-16 strings -- including a
    separate list of filesystem paths found inside, which often reveal
    original file locations.

The interior ProjectData format is proprietary and undocumented, so it
cannot be *decoded*; string mining is the honest limit of what is
parseable there, and the report says so. A `coverage` section accounts
for how every single file in the zip was handled, so nothing parseable
is silently skipped.

Inspection is ephemeral: nothing is stored and no DB rows are created.
"""

import base64
import hashlib
import math
import os
import plistlib
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, date
from pathlib import Path

MAX_UNIQUE_ASCII_STRINGS = 4000
MAX_UNIQUE_UTF16_STRINGS = 1500
MAX_TEXT_PREVIEW = 20000
MAX_IMAGE_DATA_URL_BYTES = 3 * 1024 * 1024
MAX_INLINE_BYTES = 1024  # cap for raw bytes surfaced from plists

ASCII_STRING_RE = re.compile(rb"[\x20-\x7e]{4,}")
UTF16_STRING_RE = re.compile(rb"(?:[\x20-\x7e\xa0-\xff]\x00){4,}")
PATHISH_RE = re.compile(r"(?:^|[^\w])(/(?:[\w .+&#@()\[\]'-]+/)+[\w .+&#@()\[\]'-]+)")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".heic"}
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".aifc", ".caf", ".mp3", ".m4a", ".aac",
              ".flac", ".ogg", ".opus", ".wma", ".sd2"}
TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".html", ".rtf"}


# ---------------------------------------------------------------------------
# generic helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return round(-sum((c / n) * math.log2(c / n) for c in counts.values()), 4)


def sanitize(obj, depth=0):
    """Make any parsed structure JSON-serializable."""
    if depth > 60:
        return "<max depth reached>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        out = {"__type__": "bytes", "length": len(obj),
               "hex_preview": obj[:32].hex()}
        if len(obj) <= MAX_INLINE_BYTES:
            out["base64"] = base64.b64encode(obj).decode()
        return out
    if isinstance(obj, plistlib.UID):
        return {"__type__": "uid", "value": obj.data}
    if isinstance(obj, dict):
        return {str(k): sanitize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v, depth + 1) for v in obj]
    return repr(obj)


# ---------------------------------------------------------------------------
# plists (including NSKeyedArchiver graphs)

def _resolve_nska(objects, obj, seen):
    if isinstance(obj, plistlib.UID):
        idx = obj.data
        if idx in seen:
            return f"<cyclic ref #{idx}>"
        if not (0 <= idx < len(objects)):
            return f"<dangling uid #{idx}>"
        return _resolve_nska(objects, objects[idx], seen | {idx})
    if isinstance(obj, str):
        return None if obj == "$null" else obj
    if isinstance(obj, dict):
        if "NS.keys" in obj and "NS.objects" in obj:
            keys = [_resolve_nska(objects, k, seen) for k in obj["NS.keys"]]
            vals = [_resolve_nska(objects, v, seen) for v in obj["NS.objects"]]
            out = {str(k): v for k, v in zip(keys, vals)}
            cls = _class_name(objects, obj, seen)
            if cls:
                out["__class__"] = cls
            return out
        if "NS.objects" in obj:
            out = [_resolve_nska(objects, v, seen) for v in obj["NS.objects"]]
            return out
        if "NS.string" in obj:
            return _resolve_nska(objects, obj["NS.string"], seen)
        if "NS.data" in obj:
            return _resolve_nska(objects, obj["NS.data"], seen)
        if "NS.time" in obj:
            t = _resolve_nska(objects, obj["NS.time"], seen)
            try:  # NSDate epoch is 2001-01-01
                return datetime.fromtimestamp(978307200 + float(t)).isoformat() + " (NSDate)"
            except (TypeError, ValueError):
                return t
        out = {}
        for k, v in obj.items():
            if k == "$class":
                cls = _class_name(objects, obj, seen)
                if cls:
                    out["__class__"] = cls
                continue
            if k in ("$classname", "$classes"):
                continue
            out[str(k)] = _resolve_nska(objects, v, seen)
        return out
    if isinstance(obj, (list, tuple)):
        return [_resolve_nska(objects, v, seen) for v in obj]
    return obj


def _class_name(objects, obj, seen):
    ref = obj.get("$class")
    if ref is None:
        return None
    cls = objects[ref.data] if isinstance(ref, plistlib.UID) and ref.data < len(objects) else ref
    if isinstance(cls, dict):
        return cls.get("$classname")
    return None


def parse_plist(data: bytes) -> dict:
    fmt = "binary" if data[:8] == b"bplist00" else "xml"
    parsed = plistlib.loads(data)
    out = {"format": fmt, "parsed": sanitize(parsed)}
    if isinstance(parsed, dict) and parsed.get("$archiver") == "NSKeyedArchiver":
        try:
            objects = parsed.get("$objects", [])
            top = parsed.get("$top", {})
            resolved = {str(k): _resolve_nska(objects, v, frozenset())
                        for k, v in top.items()}
            out["nskeyedarchiver_resolved"] = sanitize(resolved)
        except Exception as e:  # keep the raw form; resolution is best-effort
            out["nskeyedarchiver_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# binary analysis (ProjectData and other opaque blobs)

def extract_strings(data: bytes) -> dict:
    ascii_counts = Counter(m.group().decode("ascii")
                           for m in ASCII_STRING_RE.finditer(data))
    utf16_counts = Counter(m.group().decode("utf-16-le")
                           for m in UTF16_STRING_RE.finditer(data))
    # UTF-16 matches also hit alternating-byte ASCII noise inside real data;
    # drop UTF-16 strings that duplicate an ASCII hit.
    for s in list(utf16_counts):
        if s in ascii_counts:
            del utf16_counts[s]

    def top(counter, cap):
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return ([{"s": s, "n": n} for s, n in items[:cap]], len(items) > cap)

    ascii_list, ascii_trunc = top(ascii_counts, MAX_UNIQUE_ASCII_STRINGS)
    utf16_list, utf16_trunc = top(utf16_counts, MAX_UNIQUE_UTF16_STRINGS)

    paths = sorted({m.group(1)
                    for s in list(ascii_counts) + list(utf16_counts)
                    for m in PATHISH_RE.finditer(s)})
    return {
        "ascii": ascii_list,
        "ascii_unique_total": len(ascii_counts),
        "ascii_truncated": ascii_trunc,
        "utf16": utf16_list,
        "utf16_unique_total": len(utf16_counts),
        "utf16_truncated": utf16_trunc,
        "paths_found": paths[:500],
    }


def analyze_binary(data: bytes, note: str) -> dict:
    return {
        "size": len(data),
        "sha256": sha256_bytes(data),
        "entropy_bits_per_byte": shannon_entropy(data),
        "header_hex": data[:32].hex(),
        "note": note,
        "strings": extract_strings(data),
    }


# ---------------------------------------------------------------------------
# media: ffprobe + manual chunk parsing

def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()[:500]}
    import json as _json
    return _json.loads(result.stdout)


def _cstr(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def parse_wav_chunks(data: bytes) -> dict:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {"error": "not a RIFF/WAVE file"}
    out = {"chunks": [], "fmt": None, "bext": None, "ixml": None,
           "info_tags": {}, "cues": [], "cue_labels": {}, "smpl": None,
           "acid": None}
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        body = data[pos + 8:pos + 8 + size]
        out["chunks"].append({"id": cid.decode("ascii", errors="replace"),
                              "size": size, "offset": pos})
        try:
            if cid == b"fmt " and len(body) >= 16:
                fmt_tag, ch, sr, br, ba, bits = struct.unpack("<HHIIHH", body[:16])
                out["fmt"] = {"format_tag": fmt_tag, "channels": ch,
                              "sample_rate": sr, "byte_rate": br,
                              "block_align": ba, "bits_per_sample": bits}
            elif cid == b"bext" and len(body) >= 348:
                tl, th = struct.unpack("<II", body[338:346])
                out["bext"] = {
                    "description": _cstr(body[:256]),
                    "originator": _cstr(body[256:288]),
                    "originator_reference": _cstr(body[288:320]),
                    "origination_date": _cstr(body[320:330]),
                    "origination_time": _cstr(body[330:338]),
                    "time_reference_samples": (th << 32) | tl,
                    "version": struct.unpack("<H", body[346:348])[0],
                    "coding_history": _cstr(body[602:]) if len(body) > 602 else "",
                }
            elif cid == b"iXML":
                out["ixml"] = body.decode("utf-8", errors="replace").strip("\x00 \n")
            elif cid == b"LIST" and len(body) >= 4:
                list_type = body[:4]
                sub = 4
                while sub + 8 <= len(body):
                    sid = body[sub:sub + 4]
                    (ssize,) = struct.unpack("<I", body[sub + 4:sub + 8])
                    sval = body[sub + 8:sub + 8 + ssize]
                    if list_type == b"INFO":
                        out["info_tags"][sid.decode("ascii", errors="replace")] = _cstr(sval)
                    elif list_type == b"adtl" and sid in (b"labl", b"note") and len(sval) >= 4:
                        (cue_id,) = struct.unpack("<I", sval[:4])
                        out["cue_labels"][str(cue_id)] = _cstr(sval[4:])
                    sub += 8 + ssize + (ssize & 1)
            elif cid == b"cue " and len(body) >= 4:
                (n,) = struct.unpack("<I", body[:4])
                for i in range(min(n, 256)):
                    rec = body[4 + i * 24:4 + (i + 1) * 24]
                    if len(rec) < 24:
                        break
                    cid_, pos_, _chunk, _cstart, _bstart, off = struct.unpack("<II4sIII", rec)
                    out["cues"].append({"id": cid_, "position": pos_,
                                        "sample_offset": off})
            elif cid == b"smpl" and len(body) >= 36:
                vals = struct.unpack("<9I", body[:36])
                loops = []
                for i in range(min(vals[7], 64)):
                    rec = body[36 + i * 24:36 + (i + 1) * 24]
                    if len(rec) < 24:
                        break
                    l = struct.unpack("<6I", rec)
                    loops.append({"cue_id": l[0], "type": l[1],
                                  "start": l[2], "end": l[3],
                                  "play_count": l[5]})
                out["smpl"] = {"midi_unity_note": vals[3],
                               "sample_period_ns": vals[2], "loops": loops}
            elif cid == b"acid" and len(body) >= 24:
                flags, root = struct.unpack("<IH", body[:6])
                num_beats = struct.unpack("<I", body[12:16])[0]
                denom, num = struct.unpack("<HH", body[16:20])
                tempo = struct.unpack("<f", body[20:24])[0]
                out["acid"] = {"flags": flags, "root_note": root,
                               "num_beats": num_beats,
                               "meter": f"{num}/{denom}",
                               "tempo_bpm": round(tempo, 3)}
        except (struct.error, IndexError) as e:
            out["chunks"][-1]["parse_error"] = str(e)
        pos += 8 + size + (size & 1)
    return {k: v for k, v in out.items() if v not in (None, [], {}, "")}


def _read_ext_float80(b: bytes) -> float:
    (exp,) = struct.unpack(">H", b[:2])
    (mant,) = struct.unpack(">Q", b[2:10])
    sign = -1.0 if exp & 0x8000 else 1.0
    exp &= 0x7FFF
    if exp == 0 and mant == 0:
        return 0.0
    return sign * mant * 2.0 ** (exp - 16383 - 63)


def parse_aiff_chunks(data: bytes) -> dict:
    if data[:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        return {"error": "not an AIFF/AIFC file"}
    out = {"variant": data[8:12].decode(), "chunks": [], "comm": None,
           "text_chunks": {}, "markers": [], "instrument": None}
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack(">I", data[pos + 4:pos + 8])
        body = data[pos + 8:pos + 8 + size]
        out["chunks"].append({"id": cid.decode("ascii", errors="replace"),
                              "size": size, "offset": pos})
        try:
            if cid == b"COMM" and len(body) >= 18:
                ch, frames, bits = struct.unpack(">hIh", body[:8])
                rate = _read_ext_float80(body[8:18])
                comm = {"channels": ch, "num_frames": frames,
                        "bits_per_sample": bits, "sample_rate": rate}
                if out["variant"] == "AIFC" and len(body) >= 22:
                    comm["compression"] = body[18:22].decode("ascii", errors="replace")
                out["comm"] = comm
            elif cid in (b"NAME", b"AUTH", b"ANNO", b"(c) "):
                out["text_chunks"][cid.decode().strip()] = body.decode(
                    "ascii", errors="replace")
            elif cid == b"MARK" and len(body) >= 2:
                (n,) = struct.unpack(">H", body[:2])
                p = 2
                for _ in range(min(n, 256)):
                    if p + 7 > len(body):
                        break
                    mid, mpos = struct.unpack(">hI", body[p:p + 6])
                    slen = body[p + 6]
                    name = body[p + 7:p + 7 + slen].decode("ascii", errors="replace")
                    out["markers"].append({"id": mid, "position": mpos, "name": name})
                    p += 7 + slen + ((slen + 1) & 1)
            elif cid == b"INST" and len(body) >= 20:
                base, detune, low, high = struct.unpack(">bbbb", body[:4])
                out["instrument"] = {"base_note": base, "detune": detune,
                                     "low_note": low, "high_note": high}
        except (struct.error, IndexError) as e:
            out["chunks"][-1]["parse_error"] = str(e)
        pos += 8 + size + (size & 1)
    return {k: v for k, v in out.items() if v not in (None, [], {}, "")}


def parse_caf_chunks(data: bytes) -> dict:
    if data[:4] != b"caff":
        return {"error": "not a CAF file"}
    out = {"version": struct.unpack(">H", data[4:6])[0], "chunks": [],
           "desc": None, "info": {}}
    pos = 8
    while pos + 12 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack(">q", data[pos + 4:pos + 12])
        if size < 0:  # -1 means "rest of file" (only legal for trailing data)
            size = len(data) - pos - 12
        body = data[pos + 12:pos + 12 + size]
        out["chunks"].append({"id": cid.decode("ascii", errors="replace"),
                              "size": size, "offset": pos})
        try:
            if cid == b"desc" and len(body) >= 32:
                rate = struct.unpack(">d", body[:8])[0]
                fmt = body[8:12].decode("ascii", errors="replace")
                flags, bpp, fpp, ch, bits = struct.unpack(">IIIII", body[12:32])
                out["desc"] = {"sample_rate": rate, "format_id": fmt,
                               "format_flags": flags, "bytes_per_packet": bpp,
                               "frames_per_packet": fpp, "channels": ch,
                               "bits_per_channel": bits}
            elif cid == b"info" and len(body) >= 4:
                (n,) = struct.unpack(">I", body[:4])
                parts = body[4:].split(b"\x00")
                kv = [p.decode("utf-8", errors="replace") for p in parts if p]
                for i in range(0, min(len(kv) - 1, n * 2), 2):
                    out["info"][kv[i]] = kv[i + 1]
        except (struct.error, IndexError) as e:
            out["chunks"][-1]["parse_error"] = str(e)
        pos += 12 + size
    return {k: v for k, v in out.items() if v not in (None, [], {}, "")}


def parse_media(path: Path, data: bytes) -> dict:
    out = {"ffprobe": ffprobe(path)}
    if data[:4] == b"RIFF":
        out["wav_chunks"] = parse_wav_chunks(data)
    elif data[:4] == b"FORM":
        out["aiff_chunks"] = parse_aiff_chunks(data)
    elif data[:4] == b"caff":
        out["caf_chunks"] = parse_caf_chunks(data)
    return out


# ---------------------------------------------------------------------------
# main inspection

def _classify(name: str, head: bytes) -> str:
    """Decide how to parse a file: magic bytes first, extension second."""
    ext = Path(name).suffix.lower()
    if head[:8] == b"bplist00":
        return "plist"
    if head[:5] == b"<?xml" and b"plist" in head[:200]:
        return "plist"
    if head[:4] in (b"RIFF", b"FORM", b"caff"):
        return "media"
    if head[:8] == b"\x89PNG\r\n\x1a\n" or head[:3] == b"\xff\xd8\xff" or \
            head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if ext == ".plist":
        return "plist"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "media"
    if Path(name).name == "ProjectData":
        return "projectdata"
    if ext in TEXT_EXTS or head[:5] == b"<?xml":
        return "text"
    # binary if it contains NULs early on, else treat as text
    if b"\x00" in head:
        return "binary"
    return "text" if head else "empty"


def inspect_logicx_zip(data: bytes, filename: str) -> dict:
    if data[:2] != b"PK":
        raise ValueError(
            "Upload must be a zip archive. A .logicx is a folder -- right-click "
            "it in Finder and choose Compress, then upload the resulting .zip."
        )

    tmp = Path(tempfile.mkdtemp(prefix="logicx_inspect_"))
    try:
        zip_path = tmp / "upload.zip"
        zip_path.write_bytes(data)
        with zipfile.ZipFile(zip_path) as zf:
            return _inspect(zf, tmp / "x", data, filename)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _inspect(zf: zipfile.ZipFile, extract_root: Path, zip_bytes: bytes,
             filename: str) -> dict:
    entries = []
    skipped = []
    extract_root = extract_root.resolve()
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        target = (extract_root / zi.filename).resolve()
        if not str(target).startswith(str(extract_root) + os.sep):
            skipped.append({"path": zi.filename,
                            "reason": "unsafe path (zip-slip), not extracted"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        y, mo, d, h, mi, s = zi.date_time
        entries.append({
            "path": zi.filename,
            "size": zi.file_size,
            "compressed_size": zi.compress_size,
            "modified": f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}",
            "crc32": f"{zi.CRC:08x}",
        })

    # package root: single top-level directory (ideally *.logicx)
    tops = {e["path"].split("/", 1)[0] for e in entries}
    package_root = tops.pop() if len(tops) == 1 else None

    # group Alternatives/NNN
    alternatives: dict[str, list[str]] = {}
    for e in entries:
        m = re.search(r"(?:^|/)Alternatives/(\d+)/", e["path"])
        if m:
            alternatives.setdefault(m.group(1), []).append(e["path"])

    plists, media, images, projectdata, texts, binaries = [], [], [], [], [], []
    unparsed, coverage = [], []

    for e in entries:
        rel = e["path"]
        fpath = extract_root / rel
        content = fpath.read_bytes()
        kind = _classify(rel, content[:512])
        handled = kind
        try:
            if kind == "plist":
                plists.append({"path": rel, "size": e["size"], **parse_plist(content)})
            elif kind == "media":
                media.append({"path": rel, "size": e["size"],
                              "sha256": sha256_bytes(content),
                              **sanitize(parse_media(fpath, content))})
            elif kind == "image":
                probe = ffprobe(fpath)
                stream = (probe.get("streams") or [{}])[0]
                img = {"path": rel, "size": e["size"],
                       "width": stream.get("width"),
                       "height": stream.get("height"),
                       "codec": stream.get("codec_name")}
                if len(content) <= MAX_IMAGE_DATA_URL_BYTES:
                    mime = {"png": "image/png", "mjpeg": "image/jpeg",
                            "gif": "image/gif"}.get(stream.get("codec_name"),
                                                    "application/octet-stream")
                    img["data_url"] = (f"data:{mime};base64,"
                                       + base64.b64encode(content).decode())
                else:
                    img["note"] = "image too large to inline"
                images.append(img)
            elif kind == "projectdata":
                projectdata.append({
                    "path": rel,
                    **analyze_binary(content, (
                        "Logic's ProjectData format is proprietary and "
                        "undocumented; it cannot be decoded. Embedded strings "
                        "(track/plugin/marker names, file paths) below are the "
                        "parseable limit."
                    )),
                })
            elif kind == "text":
                texts.append({"path": rel, "size": e["size"],
                              "preview": content.decode("utf-8", errors="replace")[:MAX_TEXT_PREVIEW],
                              "truncated": len(content) > MAX_TEXT_PREVIEW})
            elif kind == "empty":
                handled = "empty file"
            else:  # opaque binary: mine it for strings anyway
                binaries.append({"path": rel,
                                 **analyze_binary(content, "unrecognized binary format")})
                handled = "binary (string-mined)"
        except Exception as exc:
            handled = f"error: {exc}"
            unparsed.append({"path": rel, "reason": str(exc)})
        coverage.append({"path": rel, "size": e["size"], "handled_as": handled})

    return {
        "zip": {
            "filename": filename,
            "size": len(zip_bytes),
            "sha256": sha256_bytes(zip_bytes),
            "file_count": len(entries),
            "entries": entries,
            "skipped_unsafe": skipped,
        },
        "package": {
            "root": package_root,
            "looks_like_logicx": bool(package_root and package_root.endswith(".logicx"))
                                  or any("Alternatives/" in e["path"] for e in entries),
            "alternatives": {k: sorted(v) for k, v in sorted(alternatives.items())},
        },
        "plists": plists,
        "media": media,
        "images": images,
        "project_data": projectdata,
        "text_files": texts,
        "other_binaries": binaries,
        "coverage": coverage,
        "unparsed": unparsed,
        "limits_note": (
            "ProjectData (and any unrecognized binary) is analyzed by string "
            "mining only -- the format is proprietary and undocumented, so full "
            "decoding is not possible. Everything else found in the package is "
            "parsed; the coverage table accounts for every file."
        ),
    }
