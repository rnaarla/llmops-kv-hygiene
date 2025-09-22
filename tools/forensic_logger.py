from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import hmac
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union


class ForensicLogger:
    """Append-only, hash-chained JSONL logger.

    Extracted from cache_tracer for modularity and clearer coverage boundaries.
    Defensive IO/permission fallbacks are intentionally marked with pragma to avoid
    brittle tests that simulate OS faults; integrity is validated via verify_all tests.
    """

    def __init__(self, log_path: Union[str, Path], *, max_bytes: int = 5_000_000, hmac_secret: Optional[bytes] = None) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():  # pragma: no branch
            self.path.touch()
            try:
                os.chmod(self.path, 0o600)
            except Exception:  # pragma: no cover - permission hardening fallback
                pass
        self._lock = threading.Lock()
        self._prev_hash = self._load_last_hash()
        self._max_bytes = max_bytes
        self._hmac_key = hmac_secret or os.environ.get("FORENSIC_HMAC_SECRET", "").encode("utf-8") or None

    @staticmethod
    def _canonicalize(record: Mapping[str, Any]) -> str:
        return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _load_last_hash(self) -> str:
        if not self.path.exists():  # pragma: no branch
            return "GENESIS"
        try:
            with self.path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(0, size - 4096)
                f.seek(start)
                tail = f.read().splitlines()
                for line in reversed(tail):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        return obj.get("curr_hash", "GENESIS")
                    except Exception:  # pragma: no cover - ignores malformed tail line
                        continue
        except Exception:  # pragma: no cover - IO failure fallback
            pass
        return "GENESIS"

    def _load_last_hash_from(self, path: Union[str, Path]) -> str:
        p = Path(path)
        if not p.exists():  # pragma: no branch
            return "GENESIS"
        try:
            with p.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(0, size - 4096)
                f.seek(start)
                tail = f.read().splitlines()
                for line in reversed(tail):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        return obj.get("curr_hash", "GENESIS")
                    except Exception:  # pragma: no cover
                        continue
        except Exception:  # pragma: no cover - IO failure fallback
            pass
        return "GENESIS"

    def append(self, record: MutableMapping[str, Any]) -> str:
        with self._lock:
            record.setdefault("schema", 1)
            record.setdefault("ts", time.time())
            record.setdefault("trace_id", str(uuid.uuid4()))
            record["prev_hash"] = self._prev_hash
            canonical = self._canonicalize(record)
            curr_hash = hashlib.sha256((self._prev_hash + canonical).encode("utf-8")).hexdigest()
            record["curr_hash"] = curr_hash
            if self._hmac_key:
                record["hmac"] = hmac.new(self._hmac_key, (self._prev_hash + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
            line = json.dumps(record, ensure_ascii=False)
            if self.path.exists() and self.path.stat().st_size + len(line) + 1 > self._max_bytes:
                rotated = self.path.with_name(self.path.stem + f"-{int(time.time())}.log")
                self.path.rename(rotated)
                prev_file_last_hash = self._load_last_hash_from(rotated)
                self._prev_hash = "GENESIS"
                rotate_record = {
                    "schema": 1,
                    "event_type": "rotate",
                    "ts": time.time(),
                    "trace_id": str(uuid.uuid4()),
                    "prev_hash": self._prev_hash,
                    "prev_file": str(rotated.name),
                    "prev_file_last_hash": prev_file_last_hash,
                }
                canonical_rotate = self._canonicalize(rotate_record)
                rotate_hash = hashlib.sha256((self._prev_hash + canonical_rotate).encode("utf-8")).hexdigest()
                rotate_record["curr_hash"] = rotate_hash
                if self._hmac_key:
                    rotate_record["hmac"] = hmac.new(self._hmac_key, (self._prev_hash + canonical_rotate).encode("utf-8"), hashlib.sha256).hexdigest()
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rotate_record, ensure_ascii=False) + "\n")
                try:
                    os.chmod(self.path, 0o600)
                except Exception:  # pragma: no cover
                    pass
                self._prev_hash = rotate_hash
                record.pop("curr_hash", None)
                record.pop("hmac", None)
                record["prev_hash"] = self._prev_hash
                canonical = self._canonicalize(record)
                curr_hash = hashlib.sha256((self._prev_hash + canonical).encode("utf-8")).hexdigest()
                record["curr_hash"] = curr_hash
                if self._hmac_key:
                    record["hmac"] = hmac.new(self._hmac_key, (self._prev_hash + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
                line = json.dumps(record, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._prev_hash = curr_hash
            return curr_hash

    @staticmethod
    def verify_chain(path: Union[str, Path], *, hmac_secret: Optional[bytes] = None) -> Dict[str, Any]:
        prev = "GENESIS"
        ok = True
        count = 0
        bad_index: Optional[int] = None
        key = hmac_secret or os.environ.get("FORENSIC_HMAC_SECRET", "").encode("utf-8") or None
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                curr = obj.get("curr_hash")
                tmp = dict(obj)
                tmp.pop("curr_hash", None)
                provided_hmac = tmp.pop("hmac", None)
                canonical = json.dumps(tmp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                calc = hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()
                if curr != calc:
                    ok = False
                    bad_index = i
                    break
                if key is not None and provided_hmac:
                    calc_hmac = hmac.new(key, (prev + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
                    if provided_hmac != calc_hmac:
                        ok = False
                        bad_index = i
                        break
                prev = curr
                count += 1
        return {"ok": ok, "lines": count, "first_bad_line": bad_index}

    @staticmethod
    def verify_all(path: Union[str, Path]) -> Dict[str, Any]:
        base = Path(path)
        directory = base.parent
        stem = base.stem
        rotated = sorted(directory.glob(f"{stem}-*.log"), key=lambda p: p.name)
        files = rotated + [base]
        results: List[Dict[str, Any]] = []
        ok = True

        def _last_hash(p: Path) -> str:
            try:
                with p.open("rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    start = max(0, size - 4096)
                    f.seek(start)
                    tail = f.read().splitlines()
                    for line in reversed(tail):
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        ch = obj.get("curr_hash")
                        if ch:
                            return ch
            except Exception:  # pragma: no cover
                pass
            return "GENESIS"

        def _first_rotate_record(p: Path) -> Optional[Dict[str, Any]]:
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("event_type") == "rotate":
                            return obj
                        break
            except Exception:  # pragma: no cover
                return None
            return None

        for p in files:
            res = ForensicLogger.verify_chain(p)
            results.append({"file": str(p.name), **res})
            if not res.get("ok", False):
                ok = False

        for i in range(1, len(files)):
            prev = files[i - 1]
            curr = files[i]
            rotate = _first_rotate_record(curr)
            if rotate is None:
                if i > 0:
                    ok = False
                    results.append({"file": str(curr.name), "ok": False, "error": "missing rotate record"})
                continue
            expected_name = prev.name
            expected_hash = _last_hash(prev)
            r_prev_file = rotate.get("prev_file")
            r_prev_hash = rotate.get("prev_file_last_hash")
            if (r_prev_file == curr.name and r_prev_hash == expected_hash):  # benign self reference
                pass
            elif r_prev_file != expected_name or r_prev_hash != expected_hash:
                ok = False
                results.append({
                    "file": str(curr.name),
                    "ok": False,
                    "error": "rotation linkage mismatch",
                    "expected_prev_file": expected_name,
                    "expected_prev_last_hash": expected_hash,
                    "rotate_prev_file": r_prev_file,
                    "rotate_prev_last_hash": r_prev_hash,
                })

        return {"ok": ok, "files": results}
