"""Replacement command-line reports. Every scientific limitation is explicit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from scipy.stats import chi2 as chi2_distribution

from . import core as c
from .data import load_sparc, sha256_file
from .fitting import (Policy, bounded_minimum, fit_sample, minimum_2d,
                      paired_delta, profile_intervals)
from .statistics import evolution_fit, forecast_sample_size
from .verification import Ledger, write_json

ROOT = Path(__file__).resolve().parents[1]


def provenance():
    sources = sorted(p for p in [*ROOT.glob("kkt/*.py"), *ROOT.glob("scripts/*.py"),
                                 *ROOT.glob("tools/*.py"), *ROOT.glob("tests/*.py"),
                                 ROOT/"requirements.in", ROOT/"pytest.ini",
                                 ROOT/".python-version"] if p.is_file())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                text=True, capture_output=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                   cwd=ROOT, check=True, text=True, capture_output=True,
                                   timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "matplotlib")},
            "source_commit": commit, "worktree_dirty": dirty,
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}}


def _plotter():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _save_figure(plt, figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "kkt-audit-reference"})
    plt.close(figure)


def verification_report(args):
    ledger = Ledger()
    T = c.HBAR*c.H0/(2*math.pi*c.K_B)
    E1 = 2*math.pi*c.HBAR*c.H0
    ledger.close("a0-thermal-ratio", (c.C*c.K_B*T/c.HBAR)/c.A0, 1.0, rtol=3e-14)
    ledger.close("mode-energy-temperature-ratio", E1/(c.K_B*T), (2*math.pi)**2, rtol=3e-14)
    ledger.close("mode-frequency-H0-ratio", (E1/c.H_PLANCK)/c.H0, 1.0, rtol=3e-14)
    ledger.close("occupation-vs-independent-exp", c.occupation(),
                 1/math.expm1((2*math.pi)**2), rtol=3e-14)
    golden = (math.sqrt(5)-1)/2
    for label, actual, expected in [
        ("mu(1)", c.mu_kk(1), golden), ("nu(1)", c.nu_kk(1), math.sqrt(2)),
        ("nu(phi)", c.nu_kk(golden), (math.sqrt(5)+1)/2),
        ("mu(sqrt2)", c.mu_kk(math.sqrt(2)), 1/math.sqrt(2)),
        ("mu(0)", c.mu_kk(0), 0.0),
    ]:
        ledger.close(label, actual, expected, rtol=3e-14)
    for y in np.logspace(-16, 16, 17):
        # A separately formed relation for x rather than calling acceleration().
        x = math.sqrt(y)*math.sqrt(y+1)
        ledger.close(f"mu-nu-inverse/y={y:g}", c.mu_kk(x)*c.nu_kk(y), 1.0, rtol=1e-12)
        ledger.close(f"quadrature/y={y:g}", c.nu_kk(y)**2/(1+1/y), 1.0, rtol=3e-14)
    y = 1e-4
    scaled = c.nu_kk(y)*math.sqrt(y)
    series = 1+y/2-y*y/8
    ledger.truth("small-y-series-positive-linear-term", abs(scaled-series) < y**3/10)
    for x in (1e-9, 1e-5, .01, .1, .5, 1., 2., 10.):
        integral, error = c.F_aqual_integral(x)
        ledger.close(f"F-integral/x={x:g}", c.F_aqual(x)/integral, 1.0, rtol=1e-10)
        ledger.truth(f"quadrature-error/x={x:g}", error <= 1e-10*abs(integral),
                     "estimated quadrature error, not an interval proof")
        ledger.close(f"momentum-inverse/x={x:g}", c.x_from_momentum(c.momentum(x))/x,
                     1.0, rtol=1e-12)
    ledger.close("E(0)-normalized", c.E_z(0), 1.0, rtol=0)
    ledger.close("E(0)-with-radiation", c.E_z(0, omega_r=9e-5), 1.0, rtol=0)
    ledger.close("BTFR-ratio-is-E-not-E-squared", c.a0_z(2)/c.A0, c.E_z(2), rtol=3e-14)
    ledger.close("generalized-half-is-gauge", c.acceleration(c.A0, model="generalized", beta=.5)/c.A0,
                 2., rtol=1e-13)
    ledger.truth("generalized-half-is-not-simple-MOND",
                 abs(c.acceleration(c.A0, model="generalized", beta=.5)
                     /c.acceleration(c.A0, model="mond")-1) > .1)
    for label, reason in {
        "Cassini": "PPN metric and light-time likelihood not specified; acceleration ratio is not gamma",
        "LLR": "requires a consistent Earth-Moon-Sun/external-field calculation and ranging likelihood",
        "binary-pulsar": "Newtonian force fraction is not a radiation/strong-field timing prediction",
        "Klein-bottle-reduction": "a Klein bottle has two internal dimensions; 1D interval ansatz is not a KB",
        "Hole-B-closure": "thermal conversion and radius are assumptions, not a derived coupling",
        "JT-Narain-birefringence": "conditional identities do not establish boundary dynamics or physical parameters",
        "general-AQUAL-equivalence": "algebraic radial relation does not prove nonspherical field equivalence",
        "published-statistical-claims": "require complete data, calibrated likelihood and rerun artifacts",
    }.items():
        ledger.not_verified(label, reason)
    if args.plot:
        plt = _plotter()
        x = np.logspace(-8, 8, 600)
        fig, ax = plt.subplots()
        ax.loglog(x, c.mu_kk(x), label="KK mu(x)")
        ax.set(xlabel="x = g/a0", ylabel="mu(x)")
        ax.legend()
        # Historical filename retained; the axis is correctly x, not y.
        _save_figure(plt, fig, args.out/"figures"/"mu_vs_y.png")
    return ledger.report(), ledger.exit_code(require_all_claims=args.require_all_claims)


def solar_report(args):
    planets = [("Mercury", .387, .20563, .240846), ("Earth", 1., .0167, 1.),
               ("Mars", 1.524, .0934, 1.8808), ("Saturn", 9.537, .0565, 29.4571)]
    rows = []
    for name, a_au, e, period_yr in planets:
        a = a_au*c.AU
        g_n = c.GM_SUN/a**2
        per_orbit = c.perihelion_per_orbit(a, e, -c.A0/2)
        rows.append({"name": name, "g_n_m_s2": g_n,
                     "extra_inward_m_s2": c.delta_kk(g_n),
                     "fractional_force_change": c.delta_kk(g_n)/g_n,
                     "precession_arcsec_century_first_order":
                         per_orbit*(180/math.pi)*3600*100/period_yr})
    return {"status": "DIAGNOSTIC_ONLY", "planets": rows,
            "sign_convention": "extra attraction inward; precession retrograde",
            "solar_radial_differential_scale_m_s2": c.A0/2*3.844e8/c.AU,
            "differential_scale_scope": "geometric scale only, not an LLR residual prediction",
            "Cassini_LLR_binary_pulsar_status": "NOT_VERIFIED",
            "Pioneer_direction": "same inward sign; amplitude alone does not explain the anomaly",
            "screening_verdict": "not established by these force-law diagnostics"}, 0


def redshift_report(args):
    rows = []
    for z in (0., .5, .9, 1., 1.5, 2., 2.2, 3., 5.):
        dex, percent = c.velocity_shift(z)
        rows.append({"z": z, "E": c.E_z(z), "a0_m_s2": c.a0_z(z),
                     "delta_log10_velocity": dex, "velocity_percent_vs_z0": percent,
                     "N_expected_5sigma_idealized": forecast_sample_size(dex, sigma_independent=.11),
                     "N_90percent_power_onesided_5sigma_idealized":
                         forecast_sample_size(dex, sigma_independent=.11, power=.9)})
    return {"status": "CONDITIONAL_PREDICTIONS", "rows": rows,
            "BTFR_A_ratio": "E(z), not E(z)^2",
            "direction": "at fixed g_N, increasing a0 increases the acceleration boost",
            "assumptions": ["a0(z)=cH(z)/(2pi) is a model ansatz",
                            "BTFR asymptotic/deep-regime applicability",
                            "forecasts omit systematics unless explicitly supplied to the helper"],
            "observational_verdict": "NOT_VERIFIED; no remembered literature numbers substituted"}, 0


def dimensional_report(args):
    R = c.C/(2*math.pi*c.H0)
    L = math.pi*R
    g5 = c.G*L
    length, mass = c.planck5(g5)
    m4 = math.sqrt(c.HBAR*c.C/c.G)
    return {"status": "CONDITIONAL_1D_INTERVAL_DIAGNOSTIC",
            "geometry": "unwarped 1D interval of length pi R; NOT a Klein bottle",
            "coupling_convention": "unreduced Einstein-Hilbert action coupling; no Gauss-law redefinition",
            "R_m": R, "L_m": L, "G5_m4_kg-1_s-2": g5,
            "planck5_length_m": length, "planck5_mass_kg": mass,
            "M5_cubed_identity_ratio": mass**3 / ((c.HBAR/(c.C*L))*m4**2),
            "planck4_mass_kg": m4,
            "planck4_energy_GeV": m4*c.C**2/1.602176634e-10,
            "solid_angle_route_over_a0": 4.0,
            "mode_mass_route_over_a0": 2*math.pi,
            "thermal_conversion_status": "assumed normalization; not a unique derivation",
            "Klein_bottle_and_Hole_B_status": "NOT_VERIFIED"}, 0


def _sample(args):
    policy = Policy(args.objective, args.floor, args.ml_min, args.ml_max, args.bulge_ml_ratio)
    galaxies, metadata = load_sparc(args.data, min_points=args.min_points,
                                    allow_disk_only=args.disk_only)
    metadata["policy"] = asdict(policy)
    metadata["quality_distance_inclination_model"] = "not supplied; conditional fixed-catalogue diagnostics"
    return galaxies, metadata, policy


def _comparison(first, second, tie_threshold=.5):
    if len(first["fits"]) != len(second["fits"]):
        raise ValueError("samples differ")
    per_galaxy = []
    for a, b in zip(first["fits"], second["fits"], strict=True):
        d = float(np.sum(paired_delta(a, b)))
        per_galaxy.append({"name": a["name"], "delta_second_minus_first": d})
    differences = np.array([r["delta_second_minus_first"] for r in per_galaxy])
    return {"sign": "positive means first model has lower cost", "tie_threshold": tie_threshold,
            "first_wins": int(np.sum(differences > tie_threshold)),
            "second_wins": int(np.sum(differences < -tie_threshold)),
            "ties": int(np.sum(np.abs(differences) <= tie_threshold)), "per_galaxy": per_galaxy,
            "delta_cost": float(np.sum(differences))}


def sparc_report(args):
    galaxies, metadata, policy = _sample(args)
    kk = fit_sample(galaxies, policy=policy)
    mond = fit_sample(galaxies, policy=policy, model="mond")
    @lru_cache(maxsize=16384)
    def mond_profile(alpha):
        return fit_sample(galaxies, policy=policy, model="mond", a0=alpha*c.A0)["cost"]
    @lru_cache(maxsize=16384)
    def beta_profile(beta):
        return fit_sample(galaxies, policy=policy, model="generalized", beta=beta)["cost"]
    free_a = bounded_minimum(mond_profile, args.alpha_bounds)
    free_beta = bounded_minimum(beta_profile, args.beta_bounds)
    support = profile_intervals(beta_profile, free_beta, args.beta_bounds)
    if args.plot:
        plt = _plotter()
        fig, ax = plt.subplots()
        for galaxy in galaxies:
            ref_g = galaxy.gas_acceleration+.5*(galaxy.disk_acceleration
                                               +policy.bulge_ml_ratio*galaxy.bulge_acceleration)
            valid = ref_g > 0
            ax.scatter(ref_g[valid], galaxy.velocity_m_s[valid]**2/galaxy.radius_m[valid], s=3)
        g = np.logspace(-13, -8, 400)
        for model in ("kk", "mond"):
            ax.plot(g, c.acceleration(g, model=model), label=model)
        ax.set(xscale="log", yscale="log", xlabel="reference g_bar (M/L disk=0.5), m/s^2",
               ylabel="observed V^2/R, m/s^2")
        ax.legend()
        _save_figure(plt, fig, args.out/"figures"/"RAR_KK_vs_MOND.png")
    return {"status": "CONDITIONAL_FITS_NOT_CALIBRATED_INFERENCE", "input": metadata,
            "test_A": _comparison(kk, mond), "kk": kk, "mond": mond,
            "test_B": {"alpha_fit": asdict(free_a), "a0": free_a.x*c.A0,
                       "conditional_H0": free_a.x*c.H0_KM_S_MPC,
                       "nominal_dof": kk["nominal_dof"]-1,
                       "mapping_caveat": "H0 interpretation requires imposing the KK a0-H0 relation on MOND"},
            "test_C": {"beta_fit": asdict(free_beta), "support": support,
                       "nominal_dof": kk["nominal_dof"]-1,
                       "beta_half_means": "additive/gauge, not simple MOND"},
            "publication_gate": "complete mass model, selection, systematic uncertainties and simulation-based calibration required"}, 0


def ds_report(args):
    galaxies, metadata, policy = _sample(args)
    @lru_cache(maxsize=32768)
    def profile(beta, alpha):
        return fit_sample(galaxies, policy=policy, model="generalized", beta=beta,
                          a0=alpha*c.A0)["cost"]
    best = minimum_2d(profile, args.beta_bounds, args.alpha_bounds)
    fixed = profile(1.0, 1.0)
    beta_one = bounded_minimum(lambda a: profile(1.0, a), args.alpha_bounds)
    threshold = float(chi2_distribution.ppf(.95, 2))
    corrections = [{"mode": n, "radius_kpc": r,
                    "assumed_relative_propagator_correction":
                        -(4*math.pi**2*n*n-9/4)/6*(c.H0*r*c.KPC/c.C)**2}
                   for n in (1, 2, 3) for r in (10, 100, 1000)]
    return {"status": "PHENOMENOLOGICAL_SEARCH", "input": metadata,
            "searched_minimum": best, "fixed_KK_cost": fixed,
            "fixed_KK_delta": fixed-best["cost"],
            "fixed_KK_inside_nominal_joint_support": fixed-best["cost"] <= threshold,
            "beta1_with_free_alpha": asdict(beta_one),
            "nominal_joint_threshold": threshold,
            "best_nominal_dof": metadata["points"]-metadata["galaxies"]-2,
            "thermal_occupations": {str(n): c.occupation(n) for n in (1, 2, 3)},
            "propagator_diagnostics": corrections,
            "theory_status": "propagator ansatz, holonomy coupling, beta-to-dimension mapping NOT_VERIFIED",
            "support_caveat": "nominal objective contour only; rerun with denser scans and calibrated systematics"}, 0


def morphology_report(args):
    galaxies, metadata, policy = _sample(args)
    kk = fit_sample(galaxies, policy=policy)
    mond = fit_sample(galaxies, policy=policy, model="mond")
    brightness = np.array([np.median(g.sb_disk) for g in galaxies])
    cuts = np.quantile(brightness, [1/3, 2/3])
    classes = np.searchsorted(cuts, brightness, side="right")
    points, totals, counts = [], np.zeros(3), np.zeros(3, int)
    unbinned_delta = 0.0
    bins = np.array([0., .5, 3., np.inf])
    for galaxy, fkk, fmond, cls in zip(galaxies, kk["fits"], mond["fits"], classes, strict=True):
        delta = paired_delta(fkk, fmond)
        counts[cls] += 1
        totals[cls] += np.sum(delta)
        reference = galaxy.gas_acceleration+.5*(galaxy.disk_acceleration
                                                +policy.bulge_ml_ratio*galaxy.bulge_acceleration)
        for rid, d, ref in zip(galaxy.row_ids, delta, reference, strict=True):
            y = float(ref/c.A0)
            bin_number = int(np.searchsorted(bins, y, side="right")-1) if y > 0 else None
            if bin_number is None:
                unbinned_delta += float(d)
            points.append({"row_id": rid, "brightness_tertile": int(cls), "y_reference": y,
                           "bin": bin_number, "delta_cost": float(d)})
    groups = []
    rng = np.random.default_rng(20261006)
    for cls in range(3):
        indices = np.flatnonzero(classes == cls)
        wins = np.array([mond["fits"][i]["cost"]-kk["fits"][i]["cost"] > .5 for i in indices], float)
        if len(wins):
            rates = [float(np.mean(rng.choice(wins, len(wins), replace=True))) for _ in range(1000)]
            interval = np.quantile(rates, [.025, .975]).tolist()
        else:
            interval = None
        groups.append({"brightness_tertile": cls, "galaxies": int(counts[cls]),
                       "delta_cost": float(totals[cls]),
                       "KK_pairwise_win_fraction": float(np.mean(wins)) if len(wins) else None,
                       "galaxy_bootstrap_95percent_interval": interval})
    binned = [{"lower_y": float(bins[j]), "upper_y": float(bins[j+1]) if j < 2 else None,
               "delta_cost": math.fsum(p["delta_cost"] for p in points if p["bin"] == j),
               "points": sum(p["bin"] == j for p in points)} for j in range(3)]
    if not math.isclose(math.fsum(p["delta_cost"] for p in points), mond["cost"]-kk["cost"],
                        rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError("per-point delta does not sum to sample delta")
    if args.plot:
        plt = _plotter()
        fig, ax = plt.subplots()
        ax.bar(range(3), [g["KK_pairwise_win_fraction"] or 0 for g in groups])
        ax.set(xticks=range(3), xticklabels=["Low sampled SB", "Middle sampled SB", "High sampled SB"],
               ylim=(0, 1), ylabel="KK pairwise win fraction (delta cost > 0.5)")
        _save_figure(plt, fig, args.out/"figures"/"kk_rar_morphology.png")
    return {"status": "EXPLORATORY_STRATIFICATION", "input": metadata,
            "classification": "median sampled SBdisk tertiles; not central surface brightness or morphology",
            "brightness_cuts": cuts.tolist(), "groups": groups, "bins": binned,
            "binning_reference": "fixed disk M/L=0.5 and declared bulge ratio; not a KK-optimized coordinate",
            "unbinned_nonpositive_reference_delta": unbinned_delta,
            "points": points, "total_delta": mond["cost"]-kk["cost"],
            "bootstrap_scope": "galaxy sampling uncertainty only, fixed fitted outcomes; no systematic propagation"}, 0


def btfr_report(args):
    """Accept a curated, provenance-bearing observation product, never remembered anchors."""
    document = json.loads(args.btfr_data.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("BTFR product must have schema_version=1")
    rows = document.get("observations", [])
    if len(rows) < 3:
        raise ValueError("BTFR product needs >=3 observations")
    required = ("id", "z", "log10_A", "source", "sample_definition", "velocity_definition", "mass_definition")
    for row in rows:
        if any(key not in row for key in required):
            raise ValueError("BTFR observation lacks provenance/observable fields")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in required if k not in ("z", "log10_A")):
            raise ValueError("empty BTFR provenance field")
        if row["velocity_definition"] != "validated_asymptotic_Vflat":
            raise ValueError("Vmax/V2.2 proxies cannot silently be used as asymptotic Vflat")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate BTFR observation IDs")
    if not document.get("covariance_provenance"):
        raise ValueError("BTFR covariance requires a provenance description, including shared systematics")
    if document.get("z0_anchor") == "remembered_or_theory_scaled":
        raise ValueError("theory-scaled/remembered anchor is not an independent observation")
    z = np.array([r["z"] for r in rows], float)
    y = np.array([r["log10_A"] for r in rows], float)
    cov = np.asarray(document["covariance_dex2"], float)
    result = evolution_fit(z, y, cov)
    if args.plot:
        plt = _plotter()
        fig, ax = plt.subplots()
        ax.errorbar(z, y, yerr=np.sqrt(np.diag(cov)), fmt="o", label="curated observations")
        curve = np.linspace(0, max(3.0, float(z.max())), 300)
        ax.plot(curve, np.log10(c.a0_z(curve)), label="fixed KK ansatz")
        ax.plot(curve, np.full_like(curve, math.log10(1.2e-10)), label="fixed simple-MOND scale")
        ax.set(xlabel="redshift", ylabel="log10[Vflat^4/(G Mbar)] in SI")
        ax.legend()
        _save_figure(plt, fig, args.out/"figures"/"kk_z_btfr_data.png")
    return {"status": "CONDITIONAL_GLS_ANALYSIS", "input_sha256": sha256_file(args.btfr_data),
            "observations": rows, "covariance_provenance": document["covariance_provenance"],
            "result": result}, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "solar", "redshift", "dimensional", "sparc", "ds", "morphology", "btfr"])
    parser.add_argument("--out", type=Path, default=ROOT/"build"/"results")
    parser.add_argument("--data", type=Path, default=ROOT/"data"/"rotation_curves.tsv")
    parser.add_argument("--btfr-data", type=Path, default=ROOT/"data"/"btfr_observations.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--require-all-claims", action="store_true")
    parser.add_argument("--disk-only", action="store_true")
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--objective", choices=["velocity", "acceleration"], default="velocity")
    parser.add_argument("--floor", type=float, default=0.0)
    parser.add_argument("--ml-min", type=float, default=.1)
    parser.add_argument("--ml-max", type=float, default=10.)
    parser.add_argument("--bulge-ml-ratio", type=float, default=1.4)
    parser.add_argument("--alpha-bounds", nargs=2, type=float, default=[.1, 3.])
    parser.add_argument("--beta-bounds", nargs=2, type=float, default=[.3, 3.])
    args = parser.parse_args(argv)
    handlers = {"verify": verification_report, "solar": solar_report, "redshift": redshift_report,
                "dimensional": dimensional_report, "sparc": sparc_report, "ds": ds_report,
                "morphology": morphology_report, "btfr": btfr_report}
    try:
        result, code = handlers[args.command](args)
    except (OSError, ValueError, KeyError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result, code = {"status": "BLOCKED", "reason": str(exc)}, 2
    result["schema_version"] = 1
    result["provenance"] = provenance()
    result["command"] = args.command
    write_json(args.out/f"{args.command}.json", result)
    if args.command == "verify" and "counts" in result:
        print(json.dumps(result["counts"], sort_keys=True))
    else:
        print(f"{args.command}: {result.get('status', 'DONE')}")
    if code:
        print(result.get("reason", "numerical failure or unverified required claim"), file=sys.stderr)
    return code