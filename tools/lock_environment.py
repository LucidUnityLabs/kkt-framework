#!/usr/bin/env python3
"""ONE-TIME maintainer operation: download pinned wheels, then generate hashes.

Run on the target CPython/platform. Do not run during a normal build or silently
regenerate the lock in CI. Review and commit the lock; archive the wheelhouse.
No hashes are invented here: every hash comes from an actual downloaded wheel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, default=ROOT/"vendor"/"wheels")
    parser.add_argument("--lock", type=Path, default=ROOT/"requirements.lock")
    args = parser.parse_args()
    if platform.python_version() != "3.13.5":
        parser.error("reference lock target is CPython 3.13.5; review a separate lock for other targets")
    if args.lock.exists() or (args.wheelhouse.exists() and any(args.wheelhouse.iterdir())):
        parser.error("refusing to overwrite a lock or mix wheels; use a new empty target for an update")
    args.wheelhouse.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                    "--dest", str(args.wheelhouse), "--requirement", str(ROOT/"requirements.in")], check=True)
    entries = {}
    inventory = []
    for path in sorted(args.wheelhouse.iterdir()):
        if path.suffix != ".whl":
            raise RuntimeError(f"non-wheel artifact: {path.name}")
        parts = path.name[:-4].split("-")
        if len(parts) < 5:
            raise RuntimeError(f"malformed wheel name: {path.name}")
        name, version = parts[0].lower().replace("_", "-"), parts[1]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in entries and entries[name][0] != version:
            raise RuntimeError(f"multiple versions resolved for {name}")
        entries.setdefault(name, (version, []))[1].append(digest)
        inventory.append({"file": path.name, "sha256": digest, "size": path.stat().st_size})
    if not entries:
        raise RuntimeError("empty wheelhouse")
    text = "# Target-specific complete wheel lock. Install with --require-hashes.\n"
    for name, (version, hashes) in sorted(entries.items()):
        text += f"{name}=={version}" + "".join(f" \\\n    --hash=sha256:{h}" for h in sorted(hashes)) + "\n"
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(text, encoding="utf-8", newline="\n")
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "requirements_in_sha256": hashlib.sha256((ROOT/"requirements.in").read_bytes()).hexdigest(),
                "lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(), "wheels": inventory}
    args.lock.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
    print(f"Wrote {args.lock}; retain the wheelhouse and manifest with the release.")


if __name__ == "__main__":
    main()