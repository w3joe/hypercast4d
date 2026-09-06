# Third-party material

HyperCast4D is a clean-room implementation of the algebraic definitions and
experiment described in:

> Kycia, R. A. & Niemczynowicz, A. *4D hypercomplex-valued neural network in
> multivariate time series forecasting*. Scientific Reports (2025).
> DOI: 10.1038/s41598-025-08957-5.

The publisher's supplementary archive contains the paper's data and GPL-3.0
source code. Neither is copied into this repository. The download command
fetches the archive from Springer Nature into the ignored local `data/`
directory and records its SHA-256 checksum. Users remain responsible for the
publisher's and data providers' terms.

This repository's implementation was written from the equations and
multiplication tables in the paper. It does not import or vendor the archived
implementation.

## Time-Series-Library model cores

`src/hypercast4d/_vendor/tslib` includes selected MIT-licensed files from
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) at commit
`4e938a1767106324dd753b2a44832bf870a0252e`. These research extensions are
vendored code, unlike the clean-room hypercomplex implementation above.
The upstream MIT notice is retained in that directory's `LICENSE` and shipped
with the Python package. `PROVENANCE.json` lists the files and local fixes.
The surrounding adapter/padding/readout is a HyperCast4D integration, not part
of the published models. TSLib variants are not necessarily the original
authors' implementations; notably its TSMixer is a simplified variant.
