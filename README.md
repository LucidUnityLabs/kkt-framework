# KKT framework: numerical diagnostics and reproducible analysis

This repository implements conditional force-law diagnostics and research fits.
Passing numerical checks does not validate a relativistic theory, cosmological
interpretation, or observational likelihood. The verifier reports its exact scope
and separately records claims that are not verified.

## Environment

The reference target is CPython 3.13.5 on Linux x86-64. `requirements.in` records
the audit-tested dependency versions. A maintainer must generate and review a
platform-specific `requirements.lock` and retain the wheelhouse; no dependency
hashes or container digests should be guessed.

    python tools/lock_environment.py
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

The original TSV is a VizieR projection without the bulge velocity column.
A complete mass-model run requires `Vbulge`; `--disk-only` is an explicit
legacy-data diagnostic, not permission to publish incomplete baryonic fits as
complete SPARC tests. See `data/SOURCES.md` and the audit for acquisition details.

    python scripts/sparc_tests_abc.py --disk-only
    python scripts/kk_rar_morphology.py --disk-only --plot
    python scripts/kk_dS_force_law.py --disk-only

The default objective is velocity-space least squares using the supplied random
velocity errors, with no additional floor. All models use the same observations.
Distance, inclination, quality selection and correlated systematic uncertainties
must be modeled before interpreting objective differences as calibrated evidence.
The shared bulge/disk M/L ratio (default 1.4) is an explicit, changeable assumption.

For a diagnostic closer to the original A/B/C configuration, use
`--objective acceleration --floor 0.10 --min-points 3 --disk-only`.
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