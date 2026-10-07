# Input provenance and outstanding acquisition work

## Original SPARC projection

The audited Git commit is `500e7acc1adfcdc9171f179d08df8aae1f7ae482`.
The original `rotation_curves.tsv` Git blob is
`8bd0fe8cf42fa0f5adf2f1a83d675abbf20ebb26` (209260 bytes).
A Git blob SHA-1 is not a SHA-256 content digest. The manifest tool authenticates
this object when explicitly requested and then calculates a real SHA-256.

Its header records retrieval on 2026-02-17T13:40:33 from VizieR catalogue
`J/AJ/152/157/table2`, Lelli, McGaugh & Schombert (2016), AJ 152, 157.
The catalogue DOI is `10.26093/cds/vizier.51520157`.
The selected columns are Name, Dist, Rad, Vobs, e_Vobs, Vgas, Vdisk and SBdisk.
SBdisk is surface brightness, not Vbulge. Keep signed gas contributions.

The complete catalogue mass-model schema additionally supplies Vbulge and
SBbulge. Restore the full selected velocity components before complete fits.
Use `tools/acquire_sparc.py --output data/rotation_curves_full.tsv` as an explicit
acquisition step, review the resulting source record and counts, and migrate the
analysis input deliberately. The helper has not been network-tested in the audit.
Preserve the original TSV under a legacy-data name before replacing its path.
Do not silently change the data file while retaining the old README statistics.

Official schema:

    https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/152/157?format=html&tex=true

The galaxy catalogue table1, not supplied in the original repository, is needed
for distance-method flags, quality, inclination and independently defined central
surface brightness. Record its actual selection and preprocessing separately.
Current reference fits condition on the supplied distances and inclinations;
they do not replace the missing nuisance-parameter analysis.

## High-redshift and local BTFR observation product

Neither the original high-redshift input files nor a reproducible local
Cepheid/TRGB anchor was present. No replacement measurements are fabricated.
Restore source tables and document every transformation from published quantities
to the curated observation product. A velocity at a finite radius or a maximum
velocity cannot silently replace an asymptotically flat rotation velocity.

The normalized `btfr_observations.json` contract is documented in the audit.
Record source identifiers, actual galaxy/sample membership, mass definitions,
velocity definitions, redshifts, log10 A, and a full covariance in dex squared.
The covariance must account for shared calibration, distance, inclination, gas
mass and overlapping-sample effects as applicable. A prose provenance field alone
does not validate those physical assumptions.

## Locking

After reviewing actual files, record their bytes; do not type placeholder hashes:

    python tools/lock_inputs.py record data/rotation_curves.tsv --original-sparc
    python tools/lock_inputs.py verify

For a new complete input set, deliberately create/review a new manifest rather
than passing `--original-sparc`. Include the curated BTFR product in the manifest
before requesting `tools/build.py --suite all`.

Retain VizieR/CDS attribution and original data terms. MIT licensing of the code
does not automatically relicense third-party catalogues.