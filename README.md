# HyperCast4D

The architecture playground includes a [paper-linked method collection](docs/method_collection.md)
with 52 entries and 15 runnable TSLib-core adaptations, plus three lightweight
mechanism-inspired templates. Full implementation of the collection is still
incomplete; see [method coverage](docs/method_collection.md).
Find it under **Builder → Method library**.

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
cd hypercast4d
source .venv/bin/activate
python -m pytest
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
with five random seeds, producing 105 result rows. Runtime varies by computer.

To run the separate archived-notebook reproduction smoke test:

```bash
hypercast4d-reproduce --quick
```

See [Exact released-notebook reproduction](#exact-released-notebook-reproduction)
before starting the complete grid search, which requires 143,360 cross-validation
fits.

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
python -m pytest
```

The full test suite should pass.

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

## Exact released-notebook reproduction

The repository has two intentionally separate experiment paths:

- `hypercast4d-run` is the practical, leakage-safe evaluation described above.
- `hypercast4d-reproduce` copies the statistical protocol and complete search
  space in the paper's released `AnalysisGoogle.ipynb`, including its known
  methodological issues.

All reproduction choices are declared in
[configs/paper_reproduction.yaml](configs/paper_reproduction.yaml): input order,
target, scaling scope, split behavior, random state, windows, horizons, folds,
training settings, architecture constants, all candidate values, algebra bug
compatibility, output paths, TensorBoard behavior, and smoke-test limits. The
runner does not contain a hidden best configuration or published score.

First run the bounded smoke test:

```bash
hypercast4d-reproduce --quick
```

It uses the `execution.quick` limits in the YAML and writes to
`results/paper_reproduction_quick/`. A successful smoke test checks the entire
pipeline, but its deliberately truncated results are not comparable with the
paper.

The complete command is:

```bash
hypercast4d-reproduce
```

For a representative subset calibrated to roughly 20 minutes on an Apple M4
Pro, run:

```bash
hypercast4d-reproduce --config configs/paper_reproduction_20min.yaml
```

This preset retains the paper's 50 epochs and 10-fold CV for w10/h1, but samples
68 configurations from the published search spaces for 680 total fits. Runtime
depends on hardware and competing workloads. Its results are written to
`results/paper_reproduction_20min/` and can be watched with:

```bash
hypercast4d-dashboard --results results/paper_reproduction_20min
```

The committed search contains 160 CNN, 160 LSTM, and 576 hypercomplex
candidates. Across 16 window/horizon cells and 10 folds, that is 143,360 model
fits of 50 epochs each. It can take many hours or days on one CPU. Progress is
persisted after every fold in:

| File | Contents |
|---|---|
| `folds.csv` | Every candidate's individual fold MAE and MSE |
| `candidates.csv` | Mean and standard deviation across folds |
| `best.csv` | Lowest-mean-MAE candidate for each model and cell |
| `status.json` | Completed and total fit counts |
| `resolved_config.yaml` | Exact settings captured before fitting starts |
| `metadata.json` | Software versions and the complete resolved configuration |

The notebook accepts an algebra candidate but always passes `quaternions` to
`HyperDense`. The default `hyper_algebra_behavior: force_quaternion` reproduces
that implementation exactly and records both the declared and effective
algebra. Set it to `declared` only when intentionally correcting that bug.

This path is a PyTorch numerical equivalent of the TensorFlow 2.12 notebook,
not a bit-for-bit TensorFlow rerun. Its layer topology, parameter counts,
preprocessing, splits, folds, optimizer settings, epoch count, and MAE selection
rule match the released code. Random initialization and framework kernels can
still produce different floating-point scores. The supplement does not include
the code that generated its separate reordered-input `HNNOrder` results, so the
configuration includes only the input order whose executable code was released.

## Architecture playground and live results

HyperCast4D includes a local visual workbench for composing neural networks,
running validation experiments, and comparing MAE/MSE against persistence. Run:

```bash
hypercast4d-playground
```

It opens `http://127.0.0.1:8765`. The existing `hypercast4d-dashboard`
command is retained as an alias. The workbench provides:

- a drag-and-drop sequential architecture builder with live tensor shapes;
- paper CNN, LSTM, Quaternion, Coquaternion, and `Cl(1,1)` presets;
- causal CNN, residual TCN, GRU, LSTM, HyperDense, pooling, normalization,
  activation, dropout, and dense blocks;
- direct, persistence-residual, and cumulative-residual forecast heads;
- a persistent one-at-a-time training queue with progress and cancellation;
- validation MAE/MSE, per-lead errors, persistence-relative scores, and
  parameter-efficiency comparisons.

The Quick preset is a pipeline smoke test. Standard evaluates the three
configured forecasting cells over five seeds. Robust evaluates all sixteen
paper window/horizon combinations over five seeds and three chronological
validation folds.

The playground ranks architecture candidates using validation data. Quick
runs cannot unlock the test set. A completed Standard or Robust candidate can
be sent to the held-out test set once through the explicit **Final test**
action. This is designed to prevent repeatedly tuning against test results.

The five locked **Paper** presets are executable equivalents of the canonical
models in `src/hypercast4d/models.py`, including Keras-style initialization,
causal padding, the full-sequence LSTM, dense/pooling/dropout scaffold, and
parameter counts. Playground runs use the same `chronological-v1` data path,
Adam defaults, MSE loss, batch size, epoch count, shuffle behavior, and disabled
early stopping declared in `configs/evaluation.yaml`. The validation-first and
test-once staging is an intentional guard around that main runner, not a change
to the neural network or its fit settings.

Architectures can be saved locally or exported as YAML. Playground state is
written beneath `results/playground/`:

```text
results/playground/
├── architectures/       # named architecture specifications
└── jobs/                 # request, status, logs, runs and summaries
```

The server only binds to `127.0.0.1` or `localhost`. Local runs keep training
and data on this machine. A Modal run sends the selected workbook and job
configuration to your Modal account and copies the standard result artifacts
back into the local job directory.

### Local or Modal GPU execution

Install the optional Modal integration and authenticate once:

```bash
uv pip install -e '.[modal]'
modal setup
```

Select **Run validation** in the builder, then choose either **Local** or
**Modal**. Modal jobs can request one of T4, L4, A10, L40S, A100 40 GB,
A100 80 GB, H100, or H200. L4 is the default. The current training loop uses a
single GPU; selecting a multi-GPU count would not accelerate it, so the UI does
not offer one.

Modal runs use the same normalized request, model compiler, data preparation,
training function, metrics, output CSVs, validation/test separation, and queue
as local runs. Their job cards include the requested GPU, actual GPU reported
by PyTorch, logs, cancellation, and a link to the Modal dashboard. GPU usage is
billed to the authenticated Modal account.

### Frontend development

The packaged Python command serves the committed production frontend. To work
on the React frontend with hot reload, use two terminals.

Terminal 1:

```bash
hypercast4d-playground --no-browser
```

Terminal 2:

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Run `npm test` for frontend tests and
`npm run build` to refresh the packaged assets.

To show the released-notebook reproduction results, including the quick run:

```bash
hypercast4d-dashboard --results results/paper_reproduction_quick
```

The same results can be shown in the playground's **Runs** view:

```bash
hypercast4d-playground --results results/paper_reproduction_20min
```

For reproduction outputs, the graph shows the lowest normalized
cross-validation MAE among the candidates completed so far. The table identifies
the candidate number and MAE standard deviation rather than an evaluation seed
and test MSE.

## TensorBoard training diagnostics

TensorBoard logging is enabled in the default configuration. Every invocation
gets a timestamped directory, and every model run is separated by forecasting
cell, model, and seed:

```text
runs/tensorboard/<experiment-id>/w10_h1/hyper_quaternion/seed_7/
```

Start TensorBoard in one terminal:

```bash
cd hypercast4d
source .venv/bin/activate
tensorboard --logdir runs/tensorboard --port 6006
```

Then open `http://127.0.0.1:6006` and run `hypercast4d-run` or
`hypercast4d-run --quick` in another terminal. Event files are flushed during
training, so the following data appears live:

- scaled training and validation loss for every epoch;
- final test MAE and MSE in original Copper units;
- trainable parameter count and training time;
- the model, window, horizon, seed, and effective model configuration;
- HParams entries for comparing completed runs.

TensorBoard is intended for training diagnostics. The built-in HyperCast4D
dashboard remains the simpler view for overall progress and final MAE bars.
Generated TensorBoard logs are ignored by Git. To disable logging or change its
location, edit the `tensorboard` section of your copied configuration.

### Customize the experiment

Copy the configuration and edit the copy:

```bash
cp configs/evaluation.yaml configs/my_experiment.yaml
hypercast4d-run --config configs/my_experiment.yaml
```

The important settings are:

```yaml
data:
  path: data/raw/paper_data.xlsx
  target_column: Copper
  train_fraction: 0.70
  validation_fraction: 0.15

experiment:
  output_dir: results/evaluation
  cells:
    - {window: 10, horizon: 1}
    - {window: 20, horizon: 5}
    - {window: 60, horizon: 20}
  seeds: [7, 19, 31, 43, 59]
  quick:
    cell_count: 1
    seed_count: 1
    epochs: 3

tensorboard:
  enabled: true
  log_dir: runs/tensorboard
  flush_seconds: 5

training:
  optimizer:
    name: adam
    learning_rate: 0.001
    beta1: 0.9
    beta2: 0.999
    epsilon: 1.0e-7
    amsgrad: false
  loss: mse
  batch_size: 32
  evaluation_batch_size: 256
  epochs: 50
  shuffle: true
  early_stopping_patience: null
  early_stopping_min_delta: 0.0
  restore_best_weights: false
  device: cpu

models:
  enabled:
    - persistence
    - linear
    - cnn
    - lstm
    - hyper_quaternion
    - hyper_coquaternion
    - hyper_cl11
  cnn:
    units: 16
    kernel_size: 3
    dense_before_pool: true
    dense_after_pool: true
    dense_units: 32
    activation: relu
    dropout: 0.50
    pool_size: 2
  lstm:
    units: 16
    dense_before_pool: true
    dense_after_pool: true
    dense_units: 32
    activation: relu
    dropout: 0.50
    pool_size: 2
  hyper:
    units: 8
    dense_before_pool: true
    dense_after_pool: true
    dense_units: 32
    activation: relu
    dropout: 0.50
    pool_size: 2
```

- `window` is the number of historical observations supplied to a model.
- `horizon` is the number of future Copper values predicted at once.
- `target_column` explicitly selects the series to forecast and moves it into
  the internal target component.
- `seeds` controls repeated training runs.
- `tensorboard.enabled` controls event logging; `log_dir` is the parent folder
  for timestamped experiments and `flush_seconds` controls live-update delay.
- `epochs` is the maximum number of passes through the training data.
- `optimizer`, `loss`, `batch_size`, `epochs`, and `shuffle` expose the paper's
  training choices. Adam's learning rate, betas, epsilon, and AMSGrad flag are
  explicit; the defaults match TensorFlow/Keras 2.12 rather than silently using
  PyTorch's slightly different Adam epsilon.
- Set `early_stopping_patience` to an integer and
  `restore_best_weights: true` to enable the corrected runner's optional
  early-stopping behavior. The paper-aligned defaults disable it.
- `device` can remain `cpu`; `auto` selects CUDA, Apple MPS, or CPU when
  available.
- `models.enabled` selects which model implementations are evaluated.
- Each trainable model has its own first-layer width, optional dense layers,
  dense width, activation, dropout, and pooling settings. The CNN kernel is
  independently configurable too.

Give custom experiments a different `output_dir` so they do not overwrite an
earlier run.

## Models included

The default configuration evaluates:

- `persistence`: repeats the most recently observed Copper value;
- `linear`: an ordinary real-valued linear model;
- `cnn`: a real-valued one-dimensional convolutional model;
- `lstm`: a real-valued recurrent model;
- `hyper_quaternion`: a quaternion HyperDense front end;
- `hyper_coquaternion`: a coquaternion HyperDense front end;
- `hyper_cl11`: a Clifford `Cl(1,1)` HyperDense front end.

The three hypercomplex variants use the same surrounding architecture. Only
the multiplication table changes, making the algebra comparison explicit and
avoiding the supplementary notebook's accidental quaternion override.

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

## Paper fidelity and scope

The implementation matches the paper and its supplementary notebook at the
model level:

- the input is an ordered 4-tuple of Copper, FCX, CLP, and SCCO, and the output
  is a sequence of future Copper values;
- quaternion, coquaternion, and `Cl(1,1)` products use the published tables and
  input-by-weight orientation;
- `HyperDense` uses four Glorot-normal component kernels and a zero bias;
- CNN uses a causal, stride-one Conv1D; LSTM returns its full sequence;
- the shared Figure 3 scaffold is optional dense layer, max pooling, flatten,
  optional dense layer, dropout, and a real dense forecast head;
- the default dropout is `0.5`, max-pool size is `2`, convolution kernel is
  `3`, and training uses Adam, MSE, 50 epochs, and batch size `32`.

The default `hypercast4d-run` command is a **bounded, corrected evaluation**,
not a bit-for-bit reproduction of the paper's grid search. Its committed YAML
chooses one explicit architecture from each published search space and evaluates
three of the paper's sixteen window/horizon cells over five seeds. Persistence
and linear models are additional baselines. The separate
`hypercast4d-reproduce` command implements the complete released-notebook
protocol. The Cartesian grids contain 160 CNN, 160 LSTM, and 576 hypercomplex
candidates; the paper text's 159/159/575 figures are each one below the actual
grid produced by its notebook.

The only values fixed in source are implementation invariants: four algebra
components, the three published multiplication tables, the supported model
registry, and the publisher download URL/checksum. Dataset paths, target,
splits, cells, seeds, training controls, first-layer widths, optional dense
layers, activations, dropout, pooling, and output locations are configuration
values.

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
This intentionally differs from the paper's evaluation protocol while keeping
the tested neural-network architecture aligned with Figure 3.

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
├── configs/evaluation.yaml          # Default experiment settings
├── configs/paper_reproduction.yaml # Released-notebook protocol and full grids
├── src/hypercast4d/
│   ├── algebras.py               # 4D multiplication tables
│   ├── layers.py                 # HyperDense implementation
│   ├── data.py                   # Leakage-safe data preparation
│   ├── models.py                 # Seven forecasting models
│   ├── architecture.py           # Composable model schema and compiler
│   ├── training.py               # Training and metrics
│   ├── experiment.py             # Evaluation CLI and result generation
│   ├── reproduction.py           # Complete released-notebook grid-search CLI
│   ├── playground.py             # Local API, queue and static app server
│   ├── playground_runner.py      # Isolated validation/test worker
│   ├── dashboard.py              # Legacy dashboard API compatibility
│   └── download.py               # Supplement downloader and verification
├── web/                          # React/TypeScript playground source
├── tests/                        # Algebra, gradient, data, and model tests
├── data/                         # Ignored downloaded material
└── results/                      # Ignored generated results
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and
[LICENSE](LICENSE) for this implementation's license.
