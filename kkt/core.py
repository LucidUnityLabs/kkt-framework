"""Shared definitions, SI units, and stable scalar/vector numerical functions.

H0 and density fractions below are declared fiducial study inputs, not exact
measurements. G and the solar-mass conversion also have physical uncertainty.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad

C = 299_792_458.0
H_PLANCK = 6.62607015e-34
HBAR = H_PLANCK / (2.0 * math.pi)
K_B = 1.380649e-23
G = 6.67430e-11
AU = 149_597_870_700.0
PARSEC = AU * 648_000.0 / math.pi
KPC = 1_000.0 * PARSEC
MPC = 1_000_000.0 * PARSEC
M_SUN = 1.989e30                 # explicit mass-conversion convention
GM_SUN = 1.32712440018e20        # use directly for solar orbits
H0_KM_S_MPC = 67.4              # fiducial, not inferred by these utilities
H0 = H0_KM_S_MPC * 1_000.0 / MPC
A0 = C * H0 / (2.0 * math.pi)


def finite_array(value, name: str, *, positive=False, nonnegative=False):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(a < 0):
        raise ValueError(f"{name} must be nonnegative")
    return a


def _finish(a):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise FloatingPointError("result is not representable as a finite float64")
    return float(a) if a.ndim == 0 else a


def nu_kk(y):
    y = finite_array(y, "y", positive=True)
    return _finish(np.hypot(1.0, 1.0 / np.sqrt(y)))


def nu_mond(y):
    y = finite_array(y, "y", positive=True)
    return _finish(0.5 + np.hypot(0.5, 1.0 / np.sqrt(y)))


def mu_kk(x):
    x = finite_array(x, "x", nonnegative=True)
    out = np.empty_like(x)
    small = x <= 1.0
    t = 2.0 * x[small]
    out[small] = t / (np.hypot(1.0, t) + 1.0)
    t = 0.5 / x[~small]
    out[~small] = 1.0 / (np.hypot(1.0, t) + t)
    return _finish(out)


def acceleration(g_n, a0=A0, *, model="kk", beta=1.0):
    """Positive inward magnitudes, not a nonspherical field-equation solver.

    The generalized family at beta=0.5 is the additive/gauge model,
    NOT simple MOND. At beta=1 it is KK quadrature.
    """
    g, a = np.broadcast_arrays(
        finite_array(g_n, "g_n", nonnegative=True),
        finite_array(a0, "a0", positive=True),
    )
    if model not in {"kk", "mond", "gauge", "generalized"}:
        raise ValueError(f"unknown model: {model}")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    out = np.zeros_like(g)
    mask = g > 0
    lg, la = np.log(g[mask]), np.log(a[mask])
    if model == "kk":
        log_g = lg + 0.5 * np.logaddexp(0.0, la - lg)
    elif model == "generalized":
        log_g = lg + np.logaddexp(0.0, beta * (la - lg)) / (2.0 * beta)
    elif model == "gauge":
        log_g = np.logaddexp(lg, 0.5 * (lg + la))
    else:
        log_g = np.logaddexp(lg - math.log(2.0),
                            0.5 * (lg + np.logaddexp(lg - math.log(4.0), la)))
    with np.errstate(over="raise", invalid="raise"):
        out[mask] = np.exp(log_g)
    return _finish(out)


def delta_kk(g_n, a0=A0):
    """Compute g_KK - g_N without subtracting nearly equal accelerations."""
    g, a = np.broadcast_arrays(
        finite_array(g_n, "g_n", nonnegative=True),
        finite_array(a0, "a0", positive=True),
    )
    out = np.zeros_like(g)
    mask = g > 0
    log_nu = 0.5 * np.logaddexp(0.0, np.log(a[mask]) - np.log(g[mask]))
    out[mask] = np.exp(np.log(a[mask]) - np.logaddexp(0.0, log_nu))
    return _finish(out)


def F_aqual(x: float) -> float:
    """Integral of sqrt(1+4t^2)-1 from 0 to x; documented float64 range.

    The series avoids subtractive cancellation near zero. The restricted
    range avoids pretending to represent a cubic integral below float range.
    """
    x = float(finite_array(x, "x", nonnegative=True))
    if x == 0.0:
        return 0.0
    if not 1e-100 <= x <= 1e100:
        raise ValueError("F_aqual supports 0 or 1e-100 <= x <= 1e100")
    if x <= 0.01:
        z = x*x
        return x**3 * (2/3 + z*(-2/5 + z*(4/7 + z*(-10/9 + z*(28/11)))))
    return 0.5*x*math.hypot(1.0, 2.0*x) + 0.25*math.asinh(2.0*x) - x


def F_aqual_integral(x: float) -> tuple[float, float]:
    """Independent scaled quadrature and its estimated absolute error.

    QUADPACK's error estimate is an estimate, not a rigorous enclosure.
    For this testing routine use 0 <= x <= 100.
    """
    x = float(finite_array(x, "x", nonnegative=True))
    if x > 100:
        raise ValueError("quadrature test range is 0 <= x <= 100")
    if x == 0:
        return 0.0, 0.0
    # Factor x^3 outside so the integrated quantity stays O(1) near zero.
    value, error = quad(lambda u: 4*u*u/(math.hypot(1.0, 2*x*u)+1.0),
                        0.0, 1.0, epsabs=2e-13, epsrel=2e-13, limit=200)
    return value*x**3, error*x**3


def momentum(x):
    x = finite_array(x, "x", nonnegative=True)
    return _finish(2.0 * x * mu_kk(x))


def x_from_momentum(p):
    p = finite_array(p, "p", nonnegative=True)
    return _finish(0.5 * np.sqrt(p) * np.sqrt(p+2.0))


def E_z(z, *, omega_m=0.315, omega_r=0.0):
    """Flat matter+radiation+Lambda background; E(0)=1 by construction."""
    z = finite_array(z, "z")
    if np.any(z <= -1) or np.any(z > 1e4):
        raise ValueError("supported redshift range is -1 < z <= 1e4")
    if not (math.isfinite(omega_m) and math.isfinite(omega_r)
            and 0 <= omega_m <= 1 and 0 <= omega_r <= 1
            and omega_m+omega_r <= 1):
        raise ValueError("invalid flat-universe density fractions")
    omega_l = 1.0-omega_m-omega_r
    return _finish(np.sqrt(omega_m*(1+z)**3 + omega_r*(1+z)**4 + omega_l))


def a0_z(z, **cosmology):
    return _finish(A0 * E_z(z, **cosmology))


def velocity_shift(z, **cosmology):
    ratio = E_z(z, **cosmology)
    dlogv = 0.25*np.log10(ratio)
    return _finish(dlogv), _finish(100*np.expm1(np.log(10.0)*dlogv))


def occupation(n=1):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("mode number must be a positive integer")
    q = (2*math.pi)**2 * n
    # exp(-q)/[1-exp(-q)] avoids overflow in exp(q).
    return math.exp(-q) / (-math.expm1(-q))


def perihelion_per_orbit(a_m: float, eccentricity: float,
                        radial_outward_acceleration: float, gm=GM_SUN):
    """First order in |A_r| a^2/GM. Positive A_r is outward.

    At e=0 this is the near-circular epicyclic limit; the periapsis of an
    exactly circular orbit is not itself defined.
    """
    if not all(math.isfinite(v) for v in (a_m, eccentricity,
                                          radial_outward_acceleration, gm)):
        raise ValueError("orbital inputs must be finite")
    if a_m <= 0 or gm <= 0 or not 0 <= eccentricity < 1:
        raise ValueError("invalid bound elliptic orbit")
    ratio = radial_outward_acceleration*a_m*a_m/gm
    if abs(ratio) > 1e-3:
        raise ValueError("perturbation too large for this first-order diagnostic")
    return 2*math.pi*ratio*math.sqrt(1-eccentricity**2)


def planck5(g5: float):
    g5 = float(finite_array(g5, "G5", positive=True))
    # G5 must be the coupling in the chosen unreduced action convention.
    length = math.exp((math.log(HBAR)+math.log(g5)-3*math.log(C))/3)
    mass = HBAR/(C*length)
    return length, mass