#!/usr/bin/env python3
"""Exact artifact check for two runs in the SAME pinned reference environment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    a = json.loads((args.first/"artifacts.sha256.json").read_text())
    b = json.loads((args.second/"artifacts.sha256.json").read_text())
    if a != b:
        changed = sorted(k for k in set(a)|set(b) if a.get(k) != b.get(k))
        raise SystemExit("artifact mismatch: "+", ".join(changed))
    print(f"{len(a)} artifacts are byte-identical")


if __name__ == "__main__":
    main()