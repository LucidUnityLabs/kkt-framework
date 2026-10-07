# KKT framework: numerical diagnostics and reproducible analysis

This repository implements conditional force-law diagnostics and research fits.
Passing numerical checks does not validate a relativistic theory, cosmological
interpretation, or observational likelihood. The verifier reports its exact scope
and separately records claims that are not verified.

## Environment

The reference target is CPython 3.13.5 on Linux x86-64. `requirements.in` records
the audit-tested dependency versions. The committed `requirements.lock` and `requirements.manifest.json` were generated
from downloaded wheels on Linux x86-64 / CPython 3.13.5. The retained
`vendor/wheels` wheelhouse supports offline installation; archive it with releases.
`container-image.lock` and `Dockerfile.locked` pin the real Python base-image
digest. These platform-specific locks do not install on macOS ARM.

    python -m venv .venv
    .venv/bin/python -m pip install --no-index --find-links vendor/wheels --only-binary=:all: --require-hashes -r requirements.lock
    .venv/bin/python -m pip check
    .venv/bin/python -m pytest -q

Lock creation is a one-time acquisition step, not a normal build operation.
After installation, numerical builds need no network connection.

## Numerical checks

    python -m kkt verify
    python tools/build.py --suite numerics --out build/numerics

The implemented reference verifier has 73 numerical checks and eight separately
unverified claim groups. Any failed numerical check returns exit code 1. Requiring
all claims with `--require-all-claims` returns code 2 while unverified claims remain.
Do not describe a successful numerical-only invocation as full theory validation.

## Data and fitting

The default input is `data/rotation_curves_full.tsv`, acquired from the named
VizieR catalogue columns including `Vbulge` (175 galaxies, 3391 observations).
The original projection remains byte-for-byte intact at `data/rotation_curves.tsv`.
`data/inputs.full.lock.json` locks both products and the acquisition source record.
See `data/SOURCES.md` for provenance and remaining inference requirements.

    python scripts/sparc_tests_abc.py
    python scripts/kk_rar_morphology.py --plot
    python scripts/kk_dS_force_law.py

The default M/L ceiling 10 blocks on UGC01281 because its physical domain is
empty; no observations are silently deleted. An explicit conditional sensitivity
run uses `tools/build.py --suite diagnostic --ml-max 20 --out build/diagnostic`.
This bound is a declared diagnostic choice, not an inferred stellar-population
prior. A legacy projection run additionally needs
`--data data/rotation_curves.tsv --disk-only` on the individual commands.

The default objective is velocity-space least squares using the supplied random
velocity errors, with no additional floor. All models use the same observations.
Distance, inclination, quality selection and correlated systematic uncertainties
must be modeled before interpreting objective differences as calibrated evidence.
The shared bulge/disk M/L ratio (default 1.4) is an explicit, changeable assumption.

For a diagnostic closer to the original A/B/C configuration, use
`--objective acceleration --floor 0.10 --min-points 3
--data data/rotation_curves.tsv --disk-only`.
This does not reproduce the original bugs, and updated constants, physical
parameter bounds and refined searches can change the published statistics.

The generalized exponent beta=1 is KK quadrature. Beta=0.5 is the additive/gauge
law, not simple MOND. The actual simple-MOND function is fitted separately.
Profile support intervals are nominal objective contours, not calibrated coverage.
No new 101/175 win rate, global best-fit acceleration or exclusion is asserted
until the complete analysis has been rerun and its assumptions validated.

## High-redshift analysis

    python -m kkt btfr --btfr-data data/btfr_observations.json --plot

This requires a curated observation product with original sources, explicitly
validated asymptotic velocities, mass definitions and a full log-space covariance.
It will not substitute a remembered, theory-scaled local anchor or treat Vmax and
V2.2 as Vflat. Missing inputs return a nonzero BLOCKED result.

## Outputs and reproducibility

Commands write atomic JSON and optional figures beneath `build/results` or
`--out`. Inputs and code carry hashes; imports do not perform analyses. Figure
generators are replacements, not byte-for-byte reproductions of historical plots.
Use `tools/lock_inputs.py` to record/verify data, and `tools/compare_builds.py` to
compare same-environment runs. Hosted CI is a numerical check; archival binary
reproducibility additionally requires a retained, digest-pinned container and
wheels. The optional `tools/lock_container.py` generates real image-digest locks.

## License

Keep the repository's existing MIT code license. External catalogue provenance
and redistribution terms remain separate from the code license.

Search resolutions can be declared with `--ml-grid-points` (default 17),
`--scalar-grid-points` (33), `--profile-grid-points` (161),
`--joint-grid-points` (11 per axis), and `--joint-starts` (5).
Both `python -m kkt` and `tools/build.py` accept these controls; catalogue reports
record them. Scalar refinement uses xatol 1e-8; joint Powell uses xtol 1e-7 and
ftol 1e-10. Flat sampled cost variation is a diagnostic and does not prevent
refining identifiable nonconstant basins. Equal sampled values cannot rule out
hidden features between samples, and large additive offsets can erase differences
at floating-point precision. Denser selected-basin and profile comparisons are
required before asserting numerical convergence; calibrated scientific inference
also requires the declared nuisance, covariance and systematic-error model.
