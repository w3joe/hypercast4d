# HyperCast4D

HyperCast4D is a standalone, clean-room evaluation of the paper
[*4D hypercomplex-valued neural network in multivariate time series
forecasting*](https://doi.org/10.1038/s41598-025-08957-5). It answers one
narrow question: do quaternion, coquaternion and `Cl(1,1)` front ends behave
distinctly and remain competitive under a chronological, leakage-safe
forecasting protocol?

This project has no connection to Numerion and does not reuse the paper's
archived implementation.

## What the paper proposes

The paper groups four related real time series into one four-component value
and replaces a model's first layer with a hypercomplex dense layer. It compares
three multiplication systems—quaternions, coquaternions and the Clifford
algebra `Cl(1,1)`—against Conv1D and LSTM alternatives on financial forecasting.
Its reported grid uses input windows of 10, 20, 40 and 60 observations and
forecast horizons of 1, 5, 10 and 20.

The attractive claim is parameter sharing: a hypercomplex map learns four real
weight matrices where an unconstrained real map of the same expanded width
would learn sixteen. That could encode relations between the four series more
efficiently. The paper reports broadly similar predictive performance across
the compared architectures, with benefits depending on setup rather than one
universally dominant algebra.

## Why this evaluation is needed

Inspection of the publisher's supplementary notebook exposes several issues
that make a direct numerical reproduction hard to interpret:

- The model factory accepts an algebra argument but constructs the hypercomplex
  layer with `quaternions` hard-coded. The archived algebra grid therefore does
  not demonstrate that the three algebras were actually compared.
- Min-max scaling is fit before splitting the data.
- Random `train_test_split` and ordinary shuffled cross-validation are applied
  to overlapping time windows, allowing temporal leakage.
- The repeatedly used validation subset is not an untouched final test set.
- The reported permutation study covers only two of the 24 possible mappings
  of four real series to four basis components.
- Failure to reject equal performance is not evidence of statistical
  equivalence; runtime and memory are also needed for efficiency claims.

HyperCast4D makes algebra selection executable and testable, fits normalization
only on training rows, divides samples chronologically by complete target
ranges, and reports an untouched test set. It includes persistence and linear
baselines in addition to CNN, LSTM and all three hypercomplex variants.

## Reproducible setup

Python 3.10 or later is required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
hypercast4d-download
pytest
hypercast4d-run --config configs/evaluation.yaml --quick
```

The quick run evaluates one window/horizon cell, one seed and at most three
epochs. The bounded default evaluates representative short, medium and long
horizons with five seeds:

```bash
hypercast4d-run --config configs/evaluation.yaml
```

Outputs are written to the ignored `results/evaluation/` directory:

- `runs.csv`: one row per model, cell and seed;
- `summary.csv`: means and standard deviations;
- `metadata.json`: environment, columns, row count and full configuration;
- `mae_by_cell.png` and `accuracy_vs_parameters.png`.

The downloader verifies the publisher archive against a pinned SHA-256, then
stores the archive, extracted workbooks and checksums under ignored `data/raw/`.
No third-party data is committed. If Springer Nature legitimately replaces the
archive, inspect it before using `--allow-updated-source`.

## Initial bounded result

The initial CPU run on the downloaded supplementary workbook used the committed
configuration (20 maximum epochs, early stopping, five seeds). Mean test MAE in
original Copper units was:

| Window / horizon | Persistence | Linear | Best hypercomplex variant |
|---|---:|---:|---:|
| 10 / 1 | **0.0534** | 0.1168 | quaternion, 0.1354 |
| 20 / 5 | **0.0868** | 0.1367 | quaternion, 0.2027 |
| 60 / 20 | **0.1503** | 0.2265 | coquaternion, 0.3304 |

Persistence wins all three cells, and the best algebra is not stable across
horizons. This is useful negative evidence: under this corrected, deliberately
bounded setup, model complexity is not yet justified. It is not a definitive
comparison—the architectures have not received equally extensive tuning—but it
sets the minimum bar that any follow-up model must clear.

## Protocol boundary

This repository intentionally does not claim bit-for-bit reproduction of the
published tables. The `chronological-v1` results answer a corrected question:
performance when future rows do not influence normalization, model selection
or training. Results should be interpreted as a falsification-oriented
evaluation, not as proof that one algebra is universally superior.

`process_peak_rss_mb` is the operating system's process-wide peak RSS sampled
after each fit. It is useful as a coarse ceiling and may be cumulative across
models; use isolated processes or a device-specific profiler for publication-
quality memory comparisons.

The most useful next research step after this check is a complete 24-permutation
study with paired multi-seed forecasts and an equivalence test (for example,
TOST with a predeclared practical error margin). That directly tests whether
component assignment or algebra choice matters, rather than rediscovering the
paper's stated low-frequency bias.
