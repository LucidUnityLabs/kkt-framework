"""Strict, header-driven SPARC ingestion; no model-dependent row deletion."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import numpy as np
from .core import KPC


@dataclass(frozen=True)
class Galaxy:
    name: str
    row_ids: tuple[str, ...]
    radius_m: np.ndarray
    velocity_m_s: np.ndarray
    velocity_error_m_s: np.ndarray
    gas_acceleration: np.ndarray
    disk_acceleration: np.ndarray
    bulge_acceleration: np.ndarray
    sb_disk: np.ndarray

    def __post_init__(self):
        n = len(self.row_ids)
        if n == 0 or len(set(self.row_ids)) != n:
            raise ValueError("galaxy requires nonempty unique row IDs")
        for key in ("radius_m", "velocity_m_s", "velocity_error_m_s",
                    "gas_acceleration", "disk_acceleration", "bulge_acceleration", "sb_disk"):
            array = np.array(getattr(self, key), dtype=float, copy=True)
            if array.shape != (n,) or not np.all(np.isfinite(array)):
                raise ValueError(f"invalid {key} for {self.name}")
            array.setflags(write=False)
            object.__setattr__(self, key, array)
        if any(np.any(getattr(self, k) <= 0) for k in
               ("radius_m", "velocity_m_s", "velocity_error_m_s")):
            raise ValueError("radius, observed velocity and velocity uncertainty must be positive")
        if any(np.any(getattr(self, k) < 0) for k in
               ("disk_acceleration", "bulge_acceleration", "sb_disk")):
            raise ValueError("stellar accelerations and surface brightness must be nonnegative")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sparc(path, *, min_points=3, allow_disk_only=False):
    """Read VizieR TSV, checking its units row and all selected observations.

    Missing Vbulge is an error unless the caller explicitly selects a
    disk+gas-only *diagnostic*. SBdisk is not a velocity or bulge column.
    """
    if not isinstance(min_points, int) or min_points < 2:
        raise ValueError("min_points must be an integer >= 2")
    required = ("Name", "Dist", "Rad", "Vobs", "e_Vobs", "Vgas", "Vdisk", "SBdisk")
    expected_units = {"Dist": "Mpc", "Rad": "kpc", "Vobs": "km/s",
                      "e_Vobs": "km/s", "Vgas": "km/s", "Vdisk": "km/s",
                      "Vbulge": "km/s", "SBdisk": "Lsun/pc2"}
    header = None
    units_seen = False
    separator_seen = False
    rows: dict[str, list] = {}
    seen_observations = set()
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = [v.strip() for v in next(csv.reader([line], delimiter="\t"))]
            if header is None:
                if fields[0] != "Name" or len(fields) != len(set(fields)):
                    raise ValueError(f"{source}:{line_number}: missing/duplicate TSV header")
                header = fields
                missing = set(required) - set(header)
                if missing:
                    raise ValueError(f"missing columns: {sorted(missing)}")
                if "Vbulge" not in header and not allow_disk_only:
                    raise ValueError("Vbulge missing; restore full mass models or explicitly use --disk-only")
                continue
            if len(fields) != len(header):
                raise ValueError(f"{source}:{line_number}: wrong field count")
            record = dict(zip(header, fields, strict=True))
            if not units_seen:
                for key, unit in expected_units.items():
                    if key in record and record[key] != unit:
                        raise ValueError(f"{source}:{line_number}: {key} must have unit {unit!r}")
                units_seen = True
                continue
            if not separator_seen:
                if not all(v and set(v) <= {"-"} for v in fields):
                    raise ValueError(f"{source}:{line_number}: missing TSV separator")
                separator_seen = True
                continue
            name = record["Name"]
            if not name:
                raise ValueError(f"{source}:{line_number}: empty galaxy identifier")
            try:
                values = {k: float(record[k]) for k in expected_units if k in record}
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: malformed numeric field") from exc
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"{source}:{line_number}: non-finite numeric field")
            if any(values[k] <= 0 for k in ("Dist", "Rad", "Vobs", "e_Vobs")):
                raise ValueError(f"{source}:{line_number}: invalid positive measurement")
            if any(values.get(k, 0) < 0 for k in ("Vdisk", "Vbulge", "SBdisk")):
                raise ValueError(f"{source}:{line_number}: negative stellar component or brightness")
            key = (name, values["Rad"])
            if key in seen_observations:
                raise ValueError(f"duplicate galaxy/radius: {key}; resolve before fitting")
            seen_observations.add(key)
            radius = values["Rad"]*KPC
            gas = math.copysign((values["Vgas"]*1e3)**2, values["Vgas"])/radius
            disk = (values["Vdisk"]*1e3)**2/radius
            bulge = (values.get("Vbulge", 0.0)*1e3)**2/radius
            rows.setdefault(name, []).append((radius, f"{name}:{line_number}",
                values["Vobs"]*1e3, values["e_Vobs"]*1e3, gas, disk, bulge,
                values["SBdisk"], values["Dist"]))
    if not separator_seen or not rows:
        raise ValueError("no valid SPARC observations")
    galaxies, exclusions = [], []
    for name, points in sorted(rows.items()):
        if len({p[8] for p in points}) != 1:
            raise ValueError(f"inconsistent distance within galaxy {name}")
        points.sort(key=lambda p: p[0])
        if len(points) < min_points:
            exclusions.append({"name": name, "reason": "min_points", "n": len(points)})
            continue
        galaxies.append(Galaxy(name, tuple(p[1] for p in points),
            *[np.array([p[index] for p in points]) for index in (0, 2, 3, 4, 5, 6, 7)]))
    if not galaxies:
        raise ValueError("selection leaves no galaxies")
    metadata = {"data_sha256": sha256_file(source), "source_name": source.name,
                "min_points": min_points, "galaxies": len(galaxies),
                "points": sum(len(g.row_ids) for g in galaxies), "excluded": exclusions,
                "baryonic_model": "disk+gas-only diagnostic" if "Vbulge" not in header
                                     else "gas+disk+bulge; declared shared-ML prescription"}
    return galaxies, metadata