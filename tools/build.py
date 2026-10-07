#!/usr/bin/env python3
"""Build/test numerical reports without refreshing data or dependency locks.

Numerics requires no catalogue. Diagnostic additionally runs SPARC/morphology/
2D fits. All additionally requires the curated BTFR product. A missing required
input is an error, not a successful skip. This does not certify the physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from lock_inputs import verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["numerics", "diagnostic", "all"], default="numerics")
    parser.add_argument("--out", type=Path, default=ROOT/"build"/"run")
    parser.add_argument("--disk-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty; never mix stale and newly built artifacts")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg", TZ="UTC",
               MPLCONFIGDIR=str(out/"mplconfig"))
    env["LC_ALL"] = "C.UTF-8"  # reference target: Linux; review locale for other targets
    def run(*command):
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    if args.suite != "numerics":
        manifest = verify_manifest(ROOT/"data"/"inputs.lock.json")
        needed = {"data/rotation_curves.tsv"}
        if args.suite == "all":
            needed.add("data/btfr_observations.json")
        if not needed <= set(manifest["files"]):
            raise ValueError(f"required inputs are absent from lock: {sorted(needed-set(manifest['files']))}")
    run("-m", "pytest", "-q")
    for command in ("verify", "solar", "redshift", "dimensional"):
        run("-m", "kkt", command, "--out", str(out/"results"), "--plot")
    if args.suite != "numerics":
        common = ["--out", str(out/"results"), "--plot"]
        if args.disk_only:
            common.append("--disk-only")
        for command in ("sparc", "morphology", "ds"):
            run("-m", "kkt", command, *common)
    if args.suite == "all":
        run("-m", "kkt", "btfr", "--out", str(out/"results"), "--plot")
    artifacts = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((out/"results").rglob("*")) if p.is_file()}
    (out/"artifacts.sha256.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True)+"\n")
    print(f"Built {len(artifacts)} artifacts for suite={args.suite}; no model-validation claim implied")


if __name__ == "__main__":
    main()