"""BTFR utilities with explicit observables, units and uncertainty assumptions."""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution, norm
from .core import A0, C, E_z, G, M_SUN, MPC, finite_array


def log10_btfr_a(velocity_km_s, log10_mass_solar):
    v, m = np.broadcast_arrays(finite_array(velocity_km_s, "V", positive=True),
                                finite_array(log10_mass_solar, "log10(Mbar)"))
    result = 4*np.log10(v) + 12.0 - math.log10(G*M_SUN) - m
    return float(result) if result.ndim == 0 else result


def log10_a_variance(sigma_logv, sigma_logm, cov_logv_logm=0.0):
    sv, sm, cov = np.broadcast_arrays(
        finite_array(sigma_logv, "sigma_logV", nonnegative=True),
        finite_array(sigma_logm, "sigma_logM", nonnegative=True),
        finite_array(cov_logv_logm, "cov(logV,logM)"))
    if np.any(np.abs(cov) > sv*sm):
        raise ValueError("impossible velocity/mass covariance")
    variance = 16*sv*sv + sm*sm - 8*cov
    if np.any(variance <= 0):
        raise ValueError("log A needs positive variance")
    return variance


def dex_errors(center, sigma_dex):
    center = finite_array(center, "center", positive=True)
    sigma = finite_array(sigma_dex, "sigma_dex", nonnegative=True)
    return center*(-np.expm1(-math.log(10)*sigma)), center*np.expm1(math.log(10)*sigma)


def bootstrap_log_median(log_values, *, seed=20261006, repetitions=4000, probability=.682689492):
    """IID galaxy bootstrap of a log median, not a replacement for systematic errors.

    For repeated measurements of one galaxy, pass one independent galaxy-level
    value or implement a cluster bootstrap. Survey-wide errors do not average down.
    """
    y = finite_array(log_values, "log_values")
    if y.ndim != 1 or len(y) < 2 or repetitions < 100 or not 0 < probability < 1:
        raise ValueError("need >=2 values, >=100 resamples and 0<probability<1")
    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions)
    # Bounded memory: do not allocate repetitions x a potentially huge catalogue.
    for j in range(repetitions):
        medians[j] = np.median(y[rng.integers(0, len(y), size=len(y))])
    lo, hi = np.quantile(medians, [(1-probability)/2, (1+probability)/2])
    return {"median_log10_A": float(np.median(y)), "lower": float(lo), "upper": float(hi),
            "seed": seed, "repetitions": repetitions, "n": len(y),
            "scope": "sampling variability only; no measurement/systematic propagation"}


def gls(design, observations, covariance):
    """Generalized least squares with known, positive-definite covariance.

    Whitening + QR avoids explicitly inverting the covariance or normal matrix.
    The returned parameter covariance is not rescaled by reduced chi-square.
    """
    X = finite_array(design, "design")
    y = finite_array(observations, "observations")
    cov = finite_array(covariance, "covariance")
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y) or cov.shape != (len(y), len(y)):
        raise ValueError("GLS shape mismatch")
    if X.shape[1] < 1 or len(y) <= X.shape[1]:
        raise ValueError("GLS requires positive residual degrees of freedom")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=0):
        raise ValueError("covariance must be symmetric")
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    Xw = solve_triangular(L, X, lower=True)
    yw = solve_triangular(L, y, lower=True)
    singular = np.linalg.svd(Xw, compute_uv=False)
    if singular[-1] <= 1e-12*singular[0]:
        raise ValueError("rank-deficient or ill-conditioned design")
    Q, R = np.linalg.qr(Xw, mode="reduced")
    estimate = solve_triangular(R, Q.T @ yw)
    Rinv = solve_triangular(R, np.eye(R.shape[0]))
    parameter_covariance = Rinv @ Rinv.T
    residual = yw-Xw@estimate
    return {"parameters": estimate, "covariance": parameter_covariance,
            "chi2": float(residual@residual), "dof": len(y)-X.shape[1]}


def evolution_fit(z, log10_A, covariance_dex2):
    """Fit log10 A = b + gamma log10 E(z).

    The nested null is gamma=0 with b free. The fixed-shape gamma=1 model
    and the free-normalization constant model have the SAME parameter count;
    their cost difference alone is not a one-dof significance test.
    """
    z = finite_array(z, "z", nonnegative=True)
    y = finite_array(log10_A, "log10_A")
    if z.ndim != 1 or y.shape != z.shape or len(z) < 3:
        raise ValueError("evolution analysis needs >=3 aligned points")
    x = np.log10(E_z(z))
    one = np.ones((len(y), 1))
    free = gls(np.column_stack([np.ones(len(y)), x]), y, covariance_dex2)
    null = gls(one, y, covariance_dex2)
    kk_free_normalization = gls(one, y-x, covariance_dex2)
    improvement = null["chi2"]-free["chi2"]
    tolerance = 1e-9*max(1.0, null["chi2"], free["chi2"])
    if improvement < -tolerance:
        raise RuntimeError("nested model fitted worse; numerical or optimization failure")
    improvement = max(0.0, improvement)  # only roundoff in a genuinely nested test
    cov = np.asarray(covariance_dex2, float)
    L = np.linalg.cholesky(cov)
    def fixed_cost(prediction):
        r = solve_triangular(L, y-prediction, lower=True)
        return float(r @ r)
    gamma = float(free["parameters"][1])
    sigma_gamma = math.sqrt(float(free["covariance"][1, 1]))
    intercept = float(kk_free_normalization["parameters"][0])
    alpha = 10**intercept/A0
    return {"model": "log10(A) = b + gamma*log10(E(z))",
            "gamma": gamma, "sigma_gamma": sigma_gamma,
            "signed_wald_z": gamma/sigma_gamma,
            "nested_delta_chi2": improvement,
            "nominal_nested_p_value": float(chi2_distribution.sf(improvement, 1)),
            "free_intercept_and_gamma_chi2": free["chi2"], "free_dof": free["dof"],
            "constant_free_normalization_chi2": null["chi2"],
            "kk_free_normalization_chi2": kk_free_normalization["chi2"],
            "free_normalization_dof": len(y)-1,
            "fixed_kk_chi2": fixed_cost(math.log10(A0)+x),
            "fixed_mond_chi2": fixed_cost(np.full(len(y), math.log10(1.2e-10))),
            "fixed_dof": len(y), "kk_alpha": alpha,
            "conditional_H0_km_s_Mpc": (alpha*A0)*2*math.pi/C*MPC/1e3,
            "inference_scope": "nominal Gaussian GLS, known covariance and prespecified model; "
                               "not a certification of survey systematics, asymptotic velocities or theory"}


def forecast_sample_size(delta, *, sigma_independent, sigma_systematic=0.0,
                         sigma_anchor=0.0, significance=5.0, power=None):
    """Gaussian forecast, all effects/errors in the SAME log-velocity units.

    power=None returns N for the expected significance, not power. Otherwise
    use a one-sided test in a prespecified direction with the requested power.
    """
    vals = (delta, sigma_independent, sigma_systematic, sigma_anchor, significance)
    if not all(math.isfinite(v) for v in vals) or sigma_independent <= 0 \
            or min(sigma_systematic, sigma_anchor) < 0 or significance <= 0:
        raise ValueError("invalid forecast inputs")
    z = significance
    if power is not None:
        if not .5 < power < 1:
            raise ValueError("power must lie between 0.5 and 1")
        z += float(norm.ppf(power))
    budget = (abs(delta)/z)**2 - sigma_systematic**2 - sigma_anchor**2
    if budget <= 0:
        return None  # no finite sample reaches the target under these assumptions
    return max(1, math.ceil(sigma_independent**2/budget))