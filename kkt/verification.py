"""One fail-closed ledger for every numerical check and every unverified claim."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import tempfile
import numpy as np


@dataclass(frozen=True)
class Check:
    label: str
    status: str
    detail: str


class Ledger:
    def __init__(self):
        self.records: list[Check] = []

    def _add(self, label, status, detail):
        if any(r.label == label for r in self.records):
            raise ValueError(f"duplicate check id: {label}")
        self.records.append(Check(label, status, detail))

    def close(self, label, computed, expected, *, rtol, atol=0.0):
        if not (math.isfinite(rtol) and math.isfinite(atol)
                and 0 <= rtol < 1 and atol >= 0):
            raise ValueError("declare finite 0 <= rtol < 1 and atol >= 0")
        a, b = np.asarray(computed, float), np.asarray(expected, float)
        if a.size == 0 or b.size == 0:
            self._add(label, "FAIL", "empty operand")
            return
        if a.shape != b.shape:
            self._add(label, "FAIL", f"shape mismatch {a.shape} != {b.shape}")
            return
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            self._add(label, "FAIL", "non-finite operand")
            return
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(a-b)
            limit = atol + rtol*np.abs(b)
        good = bool(np.all(np.isfinite(difference)) and np.all(difference <= limit))
        self._add(label, "PASS" if good else "FAIL",
                  f"rtol={rtol:g}, atol={atol:g}; max absolute error={np.max(difference):.6g}")

    def truth(self, label, condition, detail=""):
        self._add(label, "PASS" if bool(condition) else "FAIL", detail)

    def not_verified(self, label, detail):
        self._add(label, "NOT_VERIFIED", detail)

    def report(self):
        counts = {s: sum(r.status == s for r in self.records)
                  for s in ("PASS", "FAIL", "NOT_VERIFIED")}
        return {"scope": "implemented numerical checks, not validation of all paper claims",
                "counts": counts, "checks": [asdict(r) for r in self.records]}

    def exit_code(self, *, require_all_claims=False):
        if any(r.status == "FAIL" for r in self.records):
            return 1
        if require_all_claims and any(r.status == "NOT_VERIFIED" for r in self.records):
            return 2
        return 0


def write_json(path, document):
    """Atomic JSON; disallow NaN/Infinity; leave no temp file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as f:
            temporary = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)