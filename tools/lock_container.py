#!/usr/bin/env python3
"""One-time optional archival-container lock; requires Docker and network access.

A reviewed content digest is resolved from real registry metadata. No made-up
image digest is embedded in this audit. The actual container build is offline.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.13.5-slim-bookworm")
    args = parser.parse_args()
    target = ROOT/"Dockerfile.locked"
    if target.exists() or (ROOT/"container-image.lock").exists():
        parser.error("refusing to overwrite a reviewed image lock")
    if not (ROOT/"requirements.lock").exists() or not (ROOT/"vendor"/"wheels").exists():
        parser.error("generate and retain the dependency lock and wheelhouse first")
    subprocess.run(["docker", "pull", "--platform=linux/amd64", args.image], check=True)
    response = subprocess.run(["docker", "image", "inspect", args.image,
                               "--format", "{{json .RepoDigests}}"], check=True, text=True,
                              capture_output=True)
    digests = json.loads(response.stdout)
    if not digests or not re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", digests[0]):
        raise ValueError("registry did not provide a valid immutable image reference")
    image = digests[0]
    dockerfile = f'''FROM {image}
ENV PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /work
COPY vendor/wheels /deps/wheels
COPY requirements.lock /deps/requirements.lock
RUN python -m pip install --no-index --find-links=/deps/wheels --only-binary=:all: --require-hashes -r /deps/requirements.lock && python -m pip check
COPY . /work
CMD ["python", "tools/build.py", "--suite", "numerics", "--out", "/tmp/kkt-build"]
'''
    target.write_text(dockerfile, encoding="utf-8")
    (ROOT/"container-image.lock").write_text(image+"\n", encoding="utf-8")
    print("Review/commit the generated locks; build with --platform=linux/amd64 --network=none --pull=false.")


if __name__ == "__main__":
    main()