#!/usr/bin/env python3
"""Explicit input recording and verification; never update inputs during a build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SPARC_GIT_BLOB = "8bd0fe8cf42fa0f5adf2f1a83d675abbf20ebb26"


def checked_path(relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("manifest paths must be repository-relative without '..'")
    path = ROOT/p
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("input symlink escapes repository")
    return path


def verify_manifest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not document.get("files"):
        raise ValueError("invalid or empty input manifest")
    for relative, expected in document["files"].items():
        data = checked_path(relative).read_bytes()
        if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"input hash/size mismatch: {relative}")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "verify"])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--manifest", type=Path, default=ROOT/"data"/"inputs.lock.json")
    parser.add_argument("--original-sparc", action="store_true",
                        help="additionally authenticate the shipped SPARC bytes by their known Git blob ID")
    args = parser.parse_args()
    if args.mode == "verify":
        verify_manifest(args.manifest)
        print("Input manifest verified")
        return
    if args.manifest.exists() or not args.files:
        parser.error("record needs named files and a new manifest path; updates must be explicit")
    entries = {}
    for relative in args.files:
        path = checked_path(relative)
        data = path.read_bytes()
        if args.original_sparc and relative == "data/rotation_curves.tsv":
            header = f"blob {len(data)}\0".encode("ascii")
            actual = hashlib.sha1(header+data).hexdigest()  # Git object ID, NOT a SHA256 substitute
            if actual != ORIGINAL_SPARC_GIT_BLOB:
                raise ValueError("SPARC bytes do not match the audited original Git blob")
        entries[relative] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"schema_version":1, "files":entries},
                                        sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print("Recorded inputs; add source URLs, licenses, query/column choices and preprocessing to data/SOURCES.md")


if __name__ == "__main__":
    main()