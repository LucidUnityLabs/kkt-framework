"""Explicit fit policies, scaled searches, finite objectives and stable row IDs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2 as chi2_distribution
from .core import A0, acceleration
from .data import Galaxy


@dataclass(frozen=True)
class Policy:
    objective: str = "velocity"
    fractional_floor: float = 0.0
    ml_min: float = 0.1
    ml_max: float = 10.0
    bulge_ml_ratio: float = 1.4  # explicit modeling assumption, not a fitted measurement
    ml_grid_points: int = 17

    def __post_init__(self):
        if self.objective not in {"velocity", "acceleration"}:
            raise ValueError("objective must be velocity or acceleration")
        if not all(math.isfinite(v) for v in (self.fractional_floor, self.ml_min,
                                              self.ml_max, self.bulge_ml_ratio)):
            raise ValueError("policy contains non-finite values")
        if not (0 <= self.fractional_floor < 1 and 0 <= self.ml_min < self.ml_max
                and self.bulge_ml_ratio >= 0 and isinstance(self.ml_grid_points, int)
                and self.ml_grid_points >= 5):
            raise ValueError("invalid fit policy")


@dataclass(frozen=True)
class Minimum:
    x: float
    cost: float
    boundary: bool
    flat_on_grid: bool


def bounded_minimum(function, bounds, *, grid_points=33, xatol=1e-8):
    """Scan + local bounded refinements + both endpoints, in scaled coordinates.

    This is a searched minimum, not a mathematical global-optimality proof.
    Increase grid_points and compare results for difficult profiles.
    flat_on_grid flags small sampled cost variation in objective units; it does
    not suppress refinement of nonconstant sampled basins. Exactly equal sampled
    values cannot identify hidden between-sample features; additive offsets can
    erase differences in floating-point arithmetic.
    """
    lo, hi = map(float, bounds)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi
            and grid_points >= 5 and math.isfinite(xatol) and xatol > 0):
        raise ValueError("invalid search configuration")
    def evaluate(x):
        value = float(function(float(x)))
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite objective at x={x}")
        return value
    grid = np.linspace(lo, hi, grid_points)
    values = np.array([evaluate(x) for x in grid])
    candidates = list(zip(grid, values, strict=True))
    flat = bool(np.ptp(values) <= 1e-12 * max(1.0, abs(float(values.min()))))
    basins = []
    # Flatness is a units-dependent diagnostic, not a refinement veto.
    # An exactly constant sampled grid has no identifiable sampled basin.
    if np.ptp(values) > 0:
        # The lowest sampled endpoint can hide a minimum inside its edge cell.
        if values[0] < values[1]:
            basins.append((grid[0], grid[1]))
        if values[-1] < values[-2]:
            basins.append((grid[-2], grid[-1]))
        basins.extend((grid[i-1], grid[i+1]) for i in range(1, len(grid)-1)
                      if values[i] <= values[i-1] and values[i] <= values[i+1]
                      and (values[i] < values[i-1] or values[i] < values[i+1]))
    for basin in basins:
        result = minimize_scalar(evaluate, bounds=basin,
                                 method="bounded", options={"xatol": xatol, "maxiter": 1000})
        if not result.success:
            raise RuntimeError(f"optimization failed: {result.message}")
        candidates.append((float(result.x), evaluate(result.x)))
    x, cost = min(candidates, key=lambda pair: pair[1])
    edge = min(abs(x-lo), abs(x-hi)) <= max(10*xatol, 1e-7*(hi-lo))
    return Minimum(float(x), float(cost), bool(edge), flat)


def feasible_ml_bounds(galaxy: Galaxy, policy: Policy):
    star = galaxy.disk_acceleration + policy.bulge_ml_ratio*galaxy.bulge_acceleration
    gas = galaxy.gas_acceleration
    if np.any((star == 0) & (gas < 0)):
        raise ValueError(f"{galaxy.name}: no nonnegative baryonic acceleration is feasible")
    threshold = float(np.max(-gas[star > 0]/star[star > 0])) if np.any(star > 0) else -math.inf
    lower = max(policy.ml_min, threshold)
    # A nextafter step keeps cancellation at the physical boundary nonnegative.
    if threshold >= policy.ml_min:
        lower = float(np.nextafter(lower, math.inf))
    if lower >= policy.ml_max:
        raise ValueError(f"{galaxy.name}: ML domain empty after physical constraints")
    return lower, policy.ml_max


def residuals(galaxy: Galaxy, ml, *, model="kk", a0=A0, beta=1.0, policy=Policy()):
    g_n = galaxy.gas_acceleration + ml*(galaxy.disk_acceleration
                                       + policy.bulge_ml_ratio*galaxy.bulge_acceleration)
    if np.any(g_n < 0):
        raise ValueError("negative total baryonic acceleration: inadmissible parameter, not a row mask")
    predicted_g = acceleration(g_n, a0, model=model, beta=beta)
    if policy.objective == "velocity":
        observed = galaxy.velocity_m_s
        predicted = np.sqrt(predicted_g)*np.sqrt(galaxy.radius_m)
        sigma = np.maximum(galaxy.velocity_error_m_s, policy.fractional_floor*observed)
    else:
        observed = galaxy.velocity_m_s**2/galaxy.radius_m
        predicted = predicted_g
        sigma = np.maximum(2*galaxy.velocity_m_s*galaxy.velocity_error_m_s/galaxy.radius_m,
                           policy.fractional_floor*observed)
    r = (observed-predicted)/sigma
    if not np.all(np.isfinite(r)):
        raise FloatingPointError("non-finite residual")
    return r, g_n


def fit_galaxy(galaxy, *, model="kk", a0=A0, beta=1.0, policy=Policy()):
    bounds = feasible_ml_bounds(galaxy, policy)
    def objective(ml):
        r, _ = residuals(galaxy, ml, model=model, a0=a0, beta=beta, policy=policy)
        return float(r @ r)
    best = bounded_minimum(objective, bounds, grid_points=policy.ml_grid_points)
    r, g_n = residuals(galaxy, best.x, model=model, a0=a0, beta=beta, policy=policy)
    return {"name": galaxy.name, "row_ids": list(galaxy.row_ids), "ml": best.x,
            "cost": best.cost, "residuals": r.tolist(), "g_n": g_n.tolist(),
            "n": len(galaxy.row_ids), "nominal_dof": len(galaxy.row_ids)-1,
            "ml_bounds": list(bounds), "boundary": best.boundary,
            "flat_on_grid": best.flat_on_grid}


def fit_sample(galaxies, *, model="kk", a0=A0, beta=1.0, policy=Policy()):
    if not galaxies:
        raise ValueError("empty sample")
    fits = [fit_galaxy(g, model=model, a0=a0, beta=beta, policy=policy) for g in galaxies]
    return {"model": model, "a0": float(a0), "beta": float(beta), "policy": asdict(policy),
            "cost": math.fsum(f["cost"] for f in fits), "fits": fits,
            "n": sum(f["n"] for f in fits), "n_galaxies": len(fits),
            "nominal_dof": sum(f["nominal_dof"] for f in fits)}


def paired_delta(first, second):
    """Return Q_second - Q_first by ORIGINAL row ID, not filtered position."""
    if first["name"] != second["name"] or first["row_ids"] != second["row_ids"]:
        raise ValueError("cannot compare unaligned observations")
    a, b = np.asarray(first["residuals"]), np.asarray(second["residuals"])
    if a.shape != b.shape or a.shape != (len(first["row_ids"]),):
        raise ValueError("residual dimensions disagree")
    return b*b-a*a


def profile_intervals(function, best: Minimum, bounds, *, probability=.95, grid_points=161):
    """Nominal one-parameter profile-objective support intervals.

    Statistical coverage is NOT certified by this routine. It returns every
    detected connected interval and marks domain-truncated endpoints. A grid
    can miss narrow features: require scan-density stability in release work.
    """
    if not 0 < probability < 1 or not isinstance(grid_points, int) or grid_points < 5:
        raise ValueError("probability must be between zero and one and grid_points >= 5")
    threshold = float(chi2_distribution.ppf(probability, 1))
    lo, hi = bounds
    if not lo <= best.x <= hi:
        raise ValueError("minimum lies outside profile domain")
    def delta(x):
        q = float(function(float(x)))
        if not math.isfinite(q):
            raise FloatingPointError("non-finite profile")
        if q < best.cost-1e-6*max(1, abs(best.cost)):
            raise RuntimeError("profile scan found a better minimum; re-optimize")
        return q-best.cost-threshold
    grid = np.unique(np.r_[np.linspace(lo, hi, grid_points), best.x])
    values = np.array([delta(x) for x in grid])
    roots = []
    for left, right, vl, vr in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True):
        if vl == 0:
            roots.append(float(left))
        if vl*vr < 0:
            roots.append(float(brentq(delta, left, right, xtol=1e-10, rtol=1e-12)))
    if values[-1] == 0:
        roots.append(float(grid[-1]))
    edges = sorted(set([float(lo), *roots, float(hi)]))
    intervals = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if delta((left+right)/2) <= 0:
            intervals.append({"lower": left, "upper": right,
                "lower_truncated": left == lo and delta(lo) < 0,
                "upper_truncated": right == hi and delta(hi) < 0})
    return {"label": "nominal profile-objective support; coverage unvalidated",
            "delta_threshold": threshold, "intervals": intervals,
            "grid_points": grid_points, "root_xtol": 1e-10, "root_rtol": 1e-12}


def minimum_2d(function, beta_bounds, alpha_bounds, *, grid_points=11, starts=5):
    """Coarse multistart + bounded Powell search in dimensionless (beta, alpha)."""
    if (not isinstance(grid_points, int) or grid_points < 3
            or not isinstance(starts, int) or not 1 <= starts <= grid_points**2):
        raise ValueError("grid_points must be >= 3 and starts in [1, grid_points**2]")
    for lo, hi in (beta_bounds, alpha_bounds):
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
            raise ValueError("invalid positive 2D bounds")
    def objective(point):
        value = float(function(float(point[0]), float(point[1])))
        if not math.isfinite(value):
            raise FloatingPointError("non-finite 2D objective")
        return value
    candidates = [(objective((b, a)), (float(b), float(a)))
                  for b in np.linspace(*beta_bounds, grid_points)
                  for a in np.linspace(*alpha_bounds, grid_points)]
    for _, point in sorted(candidates)[:starts]:
        result = minimize(objective, point, method="Powell", bounds=(beta_bounds, alpha_bounds),
                          options={"xtol": 1e-7, "ftol": 1e-10, "maxiter": 1000})
        if not result.success:
            raise RuntimeError(f"2D search failed: {result.message}")
        candidates.append((objective(result.x), tuple(map(float, result.x))))
    cost, (beta, alpha) = min(candidates)
    boundary = any(min(abs(v-lo), abs(v-hi)) <= 1e-6*(hi-lo)
                   for v, (lo, hi) in zip((beta, alpha), (beta_bounds, alpha_bounds), strict=True))
    return {"beta": beta, "alpha": alpha, "cost": cost, "boundary": boundary,
            "method": "bounded multistart search; not certified global optimum",
            "grid_points_per_axis": grid_points, "refined_starts": starts,
            "xtol": 1e-7, "ftol": 1e-10}