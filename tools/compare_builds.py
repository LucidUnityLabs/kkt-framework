#!/usr/bin/env python3
"""Rehash complete artifacts for two runs in the SAME reference environment."""
import argparse
import hashlib
import json
from pathlib import Path


def verified_inventory(root):
    ledger = json.loads((root / "artifacts.sha256.json").read_text())
    if not isinstance(ledger, dict) or not ledger:
        raise ValueError(f"{root}: artifact ledger must be a nonempty object")
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "results").rglob("*")) if p.is_file()}
    if set(actual) != set(ledger):
        missing = sorted(set(ledger) - set(actual))
        extra = sorted(set(actual) - set(ledger))
        raise ValueError(f"{root}: artifact inventory mismatch; missing={missing}; extra={extra}")
    changed = sorted(key for key in actual if actual[key] != ledger[key])
    if changed:
        raise ValueError(f"{root}: artifact hash mismatch: {', '.join(changed)}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    try:
        a = verified_inventory(args.first)
        b = verified_inventory(args.second)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if a != b:
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise SystemExit("artifact mismatch: " + ", ".join(changed))
    print(f"{len(a)} artifacts are byte-identical")


if __name__ == "__main__":
    main()
