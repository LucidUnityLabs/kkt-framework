#!/usr/bin/env python3
"""Compatibility entry point; importing this file does not run an analysis."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kkt.commands import main
    raise SystemExit(main(["morphology", *sys.argv[1:]]))
