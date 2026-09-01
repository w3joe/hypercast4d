# HyperCast4D

HyperCast4D is a runnable, clean-room evaluation of the paper
[*4D hypercomplex-valued neural network in multivariate time series
forecasting*](https://doi.org/10.1038/s41598-025-08957-5).

It implements quaternion, coquaternion, and Clifford `Cl(1,1)` dense layers in
PyTorch and compares them with persistence, linear, CNN, and LSTM forecasters.
The evaluation uses chronological splits and training-only normalization to
avoid future-data leakage.

This project is only about the 4D hypercomplex paper. It has no connection to
Numerion.

## Run it now on this machine

The environment and paper dataset are already prepared in this workspace. Open
a terminal and run:

```bash
cd "/Users/w3joe/Desktop/Quantum Works/hypercast4d"
source .venv/bin/activate
pytest
hypercast4d-run --quick
```

The quick evaluation runs one window/horizon combination, one random seed, and
at most three training epochs. It should finish in a few seconds. Its results
will appear in `results/evaluation_quick/`.

To run the complete bounded evaluation instead:

```bash
hypercast4d-run
```

The complete evaluation runs seven models on three forecasting configurations
with five random seeds, producing 105 result rows. On the development machine
it took roughly 30 seconds on CPU, although runtime will vary by computer.

When you are finished, leave the environment with:

```bash
deactivate
```

## Set up a fresh copy

Python 3.10 or newer is required; Python 3.12 is the tested version. Start in
the repository's root directory—the directory containing `pyproject.toml`.

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

If `python3.12` is unavailable but `python3 --version` reports a supported
version, use `python3` instead.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Download the paper data

After installation, download and extract the publisher's supplementary data:

```bash
hypercast4d-download
```

This command:

1. Downloads the supplementary ZIP from Springer Nature.
2. Checks it against a pinned SHA-256 checksum.
3. Extracts the experiment and summary workbooks into `data/raw/`.
4. Writes the observed checksums to `data/raw/checksums.json`.

The downloaded files are ignored by Git and are not included in this
repository. If the publisher changes its archive, the command stops instead of
silently using different inputs. Inspect any changed archive before using the
`--allow-updated-source` option.

Verify the installation with:

```bash
pytest
```

All 27 tests should pass.

## Run an evaluation

The default configuration is [configs/evaluation.yaml](configs/evaluation.yaml).

### Quick smoke run

```bash
hypercast4d-run --quick
```

Use this first to confirm that data loading, training, metrics, and plotting all
work on your machine. Quick outputs use `results/evaluation_quick/`, so they do
not overwrite the complete evaluation.

### Complete bounded run

```bash
hypercast4d-run
```

This is equivalent to:

```bash
hypercast4d-run --config configs/evaluation.yaml
```

## Live results dashboard

HyperCast4D includes a local browser dashboard with no extra web-framework
dependency. To view the existing complete results, run:

```bash
hypercast4d-dashboard
```

It opens `http://127.0.0.1:8765` and displays:

- experiment progress;
- the number of completed models and forecasting cells;
- a live grouped MAE graph;
- the latest completed runs and their timing;
- the best mean MAE observed so far.

To watch a new full evaluation live, use two terminals.

Terminal 1:

```bash
cd "/Users/w3joe/Desktop/Quantum Works/hypercast4d"
source .venv/bin/activate
hypercast4d-dashboard
```

Terminal 2:

```bash
cd "/Users/w3joe/Desktop/Quantum Works/hypercast4d"
source .venv/bin/activate
hypercast4d-run
```

The runner writes `runs.csv` and `status.json` atomically after every completed
model. The dashboard polls those files every 1.5 seconds, so bars and progress
appear while training continues.

To watch a quick evaluation instead:

```bash
hypercast4d-dashboard --results results/evaluation_quick
```

Then run `hypercast4d-run --quick` in the second terminal. Use `Ctrl-C` to stop
the dashboard server. Pass `--no-browser` if you do not want it to open a tab
automatically, or choose another port with `--port 9000`. The server binds to
`127.0.0.1` by default, and no experiment data is uploaded anywhere.

### Customize the experiment

Copy the configuration and edit the copy:

```bash
cp configs/evaluation.yaml configs/my_experiment.yaml
hypercast4d-run --config configs/my_experiment.yaml
```

The important settings are:

```yaml
experiment:
  output_dir: results/evaluation
  cells:
    - {window: 10, horizon: 1}
    - {window: 20, horizon: 5}
    - {window: 60, horizon: 20}
  seeds: [7, 19, 31, 43, 59]

training:
  batch_size: 64
  epochs: 20
  learning_rate: 0.001
  patience: 5
  device: cpu
```

- `window` is the number of historical observations supplied to a model.
- `horizon` is the number of future Copper values predicted at once.
- `seeds` controls repeated training runs.
- `epochs` is the maximum number of passes through the training data.
- `patience` enables early stopping when validation error stops improving.
- `device` can remain `cpu`; `auto` selects CUDA, Apple MPS, or CPU when
  available.

Give custom experiments a different `output_dir` so they do not overwrite an
earlier run.

## Models included

Every experiment evaluates:

- `persistence`: repeats the most recently observed Copper value;
- `linear`: an ordinary real-valued linear model;
- `cnn`: a real-valued one-dimensional convolutional model;
- `lstm`: a real-valued recurrent model;
- `hyper_quaternion`: a quaternion HyperDense front end;
- `hyper_coquaternion`: a coquaternion HyperDense front end;
- `hyper_cl11`: a Clifford `Cl(1,1)` HyperDense front end.

The three hypercomplex variants use the same surrounding architecture. Only
the multiplication table changes, making the algebra comparison explicit.

## Output files

Each run writes the following files under its configured output directory:

| File | Contents |
|---|---|
| `runs.csv` | One row per model, forecasting cell, and seed |
| `summary.csv` | Mean and standard deviation grouped by model and cell |
| `status.json` | Live runner state and completed/total model counts |
| `metadata.json` | Python, PyTorch, device, input columns, and full configuration |
| `mae_by_cell.png` | Test MAE comparison across forecasting cells |
| `accuracy_vs_parameters.png` | Accuracy versus trainable-parameter count |

The main metric is `mae`, measured in the original Copper units. Lower is
better. `mse` is also reported. Compare every trained model with persistence;
a more complicated model is not useful if it cannot beat that baseline.

`process_peak_rss_mb` is the operating system's process-wide peak resident
memory sampled after each fit. It is a coarse ceiling and can accumulate across
models. Use isolated processes or a device profiler for publication-quality
memory measurements.

## Initial result

The initial CPU run used the committed configuration with five seeds. Mean test
MAE was:

| Window / horizon | Persistence | Linear | Best hypercomplex variant |
|---|---:|---:|---:|
| 10 / 1 | **0.0534** | 0.1168 | quaternion, 0.1354 |
| 20 / 5 | **0.0868** | 0.1367 | quaternion, 0.2027 |
| 60 / 20 | **0.1503** | 0.2265 | coquaternion, 0.3304 |

Persistence won all three cells, and the best algebra was not stable across
horizons. This is useful negative evidence under a bounded evaluation, not a
claim that hypercomplex forecasting can never work. The architectures have not
received equally extensive hyperparameter tuning.

## What the paper proposes

The paper groups four related real time series into one four-component value
and replaces a model's first layer with a hypercomplex dense layer. Its dataset
contains Copper, FCX, the Chilean Peso exchange rate, and SCCO. It compares
quaternions, coquaternions, and `Cl(1,1)` with Conv1D and LSTM alternatives over
input windows of 10, 20, 40, and 60 and forecast horizons of 1, 5, 10, and 20.

A hypercomplex map learns four real weight matrices where an unconstrained real
map of matching expanded width would learn sixteen. The intended advantage is
therefore structured parameter sharing between the four series.

## Does the paper have a GitHub repository?

There is an author-owned, related repository:

- [rkycia/KHNN](https://github.com/rkycia/KHNN) — Keras-based Hypercomplex
  Neural Networks by Radosław Kycia and Agnieszka Niemczynowicz. It includes
  general algebra definitions, hyperdense layers, convolutional layers,
  examples, and experimental PyTorch files.

There is also an institutional fork:

- [ElsevierSoftwareX/SOFTX-D-25-00004](https://github.com/ElsevierSoftwareX/SOFTX-D-25-00004)
  — a fork of `rkycia/KHNN`, not an independent implementation of the
  forecasting paper.

As checked in September 2026, neither repository is a dedicated, turnkey
repository for this exact forecasting experiment. The paper-specific notebook,
dataset, saved results, model summaries, and plots are instead distributed in
the supplementary ZIP linked from the
[Scientific Reports article](https://www.nature.com/articles/s41598-025-08957-5).
The downloader in this repository retrieves that ZIP.

HyperCast4D does not copy KHNN or the supplementary source code. Its PyTorch
layer was independently implemented from the paper's multiplication tables and
checked against the archived block-matrix convention.

## Why this evaluation differs from the archived notebook

Inspection of the paper's supplementary notebook found several issues:

- Its model factory accepts an algebra argument but constructs the HyperDense
  layer with `quaternions` hard-coded.
- Min-max normalization is fitted before the dataset is split.
- Random train/test splitting and ordinary shuffled cross-validation are
  applied to overlapping time windows.
- The repeatedly used validation subset is not an untouched test set.
- Only two of the 24 possible assignments of four series to four algebra
  components are evaluated.
- Failure to reject equal performance is treated as evidence of equivalence,
  without a dedicated equivalence test.

HyperCast4D therefore uses a corrected `chronological-v1` protocol: scaling is
fitted only on training rows, target windows cannot cross split boundaries, and
the final test partition remains untouched during training and model selection.
It does not claim bit-for-bit reproduction of the published tables.

## Troubleshooting

### `hypercast4d-run: command not found`

Activate the environment and reinstall the project:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

### `Dataset not found`

Run:

```bash
hypercast4d-download
```

The expected file is `data/raw/paper_data.xlsx`.

### The complete run takes too long

Start with `hypercast4d-run --quick`. For a smaller custom run, reduce the
number of `seeds`, `cells`, or `epochs` in a copied YAML configuration.

### Results were overwritten

Change `experiment.output_dir` in each custom configuration. Results are
ignored by Git, so overwriting them cannot be recovered through Git history.

## Repository layout

```text
hypercast4d/
├── configs/evaluation.yaml       # Default experiment settings
├── src/hypercast4d/
│   ├── algebras.py               # 4D multiplication tables
│   ├── layers.py                 # HyperDense implementation
│   ├── data.py                   # Leakage-safe data preparation
│   ├── models.py                 # Seven forecasting models
│   ├── training.py               # Training and metrics
│   ├── experiment.py             # Evaluation CLI and result generation
│   ├── dashboard.py              # Auto-refreshing local results dashboard
│   └── download.py               # Supplement downloader and verification
├── tests/                        # Algebra, gradient, data, and model tests
├── data/                         # Ignored downloaded material
└── results/                      # Ignored generated results
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and
[LICENSE](LICENSE) for this implementation's license.
