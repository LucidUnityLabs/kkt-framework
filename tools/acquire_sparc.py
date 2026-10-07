#!/usr/bin/env python3
"""Explicit one-time acquisition, NOT part of a normal build.

Fetch a complete named-column table rather than VizieR's default projection.
Retain the original legacy TSV separately before replacing it. Catalogue
identity, record counts and schema must be reviewed before accepting a change.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt.data import load_sparc, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    candidate = output.with_suffix(".candidate.tsv")
    source_record = output.with_suffix(".source.json")
    if any(p.exists() for p in (output, candidate, source_record)):
        parser.error("refusing to overwrite an input or source record")
    columns = "Name,Dist,Rad,Vobs,e_Vobs,Vgas,Vdisk,Vbulge,SBdisk"
    query = urlencode({"-source":"J/AJ/152/157/table2", "-out.max":"10000", "-out":columns})
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"+query
    request = Request(url, headers={"User-Agent":"kkt-framework-reproducible-acquisition/1"})
    with urlopen(request, timeout=60) as response:
        raw = response.read(10_000_001)
    if len(raw)>10_000_000:
        raise ValueError("unexpectedly large catalogue response")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(raw)
    try:
        _, metadata = load_sparc(candidate, min_points=3, allow_disk_only=False)
        if metadata["galaxies"] != 175 or metadata["points"] != 3391:
            raise ValueError("catalogue count differs from the documented release; inspect before accepting")
        candidate.replace(output)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    source_record.write_text(json.dumps({"url":url,
        "retrieved_utc":datetime.now(timezone.utc).isoformat(), "columns":columns.split(","),
        "sha256":sha256_file(output), "catalogue":"J/AJ/152/157/table2",
        "citation":"Lelli, McGaugh & Schombert (2016), AJ 152, 157",
        "rights":"retain CDS/VizieR and original catalogue terms; MIT code license does not replace them"},
        indent=2,sort_keys=True)+"\n", encoding="utf-8")
    print("Acquired and schema-checked input; review source record, then record a new input manifest.")


if __name__ == "__main__":
    main()