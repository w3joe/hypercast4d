# Method implementation audit — 2026-09-06

## Direct Dense / HyperDense replacement and Builder refinement

The inspector now switches layer type without rewiring, converts real output
width to four-component units, preserves bias, and checks every evaluation cell
before applying the change. Incompatible conversions leave the original graph
intact. Replacements use independent, newly initialized weights. Graph-only
HyperDense now supports vector and higher-rank feature tensors; v1 behavior is
unchanged. A searchable layer finder opens the containing group and focuses the
selected layer. Type/algebra controls, readiness status and a separate run panel
make the top-down Builder easier to navigate.

Verification: **466 Python tests, 29 frontend tests, production build passed**.
Tests include real internal-layer replacement with an optimizer update,
vector/sequence/higher-rank HyperDense gradients and checkpoints for all three
algebras, matched-width conversion, bias preservation, save/undo and rejection
when another evaluation cell changes shape. No browser is connected, so visual
review is still unverified.

## Latest: graph-native architecture canvas

The executable v2 graph replaces the earlier nested observer in the Builder.
All 15 cores support layer and topology edits, multi-model branches, explicit
sharing, independent duplication, saving/loading and all-cell validation.
Verification against the combined workspace: **456 Python tests and 18 frontend
tests passed**; the production build succeeded. All 15 pinned-source output and
gradient comparisons also passed. See [graph architecture](graph_architecture.md).
Live local API checks converted and validated every TSLib core. An edited
TSMixer graph (temporal hidden width 64) completed a one-epoch job successfully.
Browser visual review and GPU/Modal execution remain untested.

## Historical follow-up: expandable model preview

All 15 TSLib cores now support **Expand model** on their canvas cards. Graphs
record leaf-module calls and observed tensor dependencies from a deterministic
synthetic CPU eval pass, with explicit multi-source junctions. They do not infer
execution order from module registration order. Single-source tensor operations
are collapsed into edges; this is a sampled execution view, not an exhaustive
static graph. Nodes linked to editable dense targets open the Inspector. A
containing MLP can be selected as a group; its replacement nodes link back to
that same group. Graphs refresh after architecture edits.

Tests cover every core's graph, valid editable targets, acyclicity, output
reachability, unchanged specs and CPU RNG state, both TSMixer residual branches,
actual custom widths `64 → 16`, preceding-block width changes, and HTTP errors.
Frontend tests exercise expansion, selecting a canvas node, applying an internal
edit through the Inspector, collapsing/reopening, and selecting an MLP group.
The graph has no authority to change connections or non-dense operations.

Browser visual inspection could not be performed: there is no connected browser.
Verification: **319 Python tests and 11 frontend tests passed**; production build
succeeded.

## Follow-up: internal dense editing

Added shape-preserving replacements for actual Linear and pure dense Sequential
submodules. TSMixer's temporal/channel MLPs can now have their internal hidden
widths changed rather than just accepting layers around the model core.
The editor is deliberately not a general graph-rewiring interface.

Verification: **300 Python tests and 8 frontend tests passed**, production build
succeeded. Every exposed internal target was individually replaced and checked
for finite gradients through its new parameters and inputs. Tests also cover
optimizer updates, checkpoint serialization/reconstruction, unchanged-model
weights/output/RNG/hash parity, invalid and overlapping edits, API inspection,
saved-spec roundtrips and a real one-epoch edited-TSMixer playground job.
The pinned-source audit still passes for all 15 unmodified model cores.

Inactive calendar embeddings, TimeMixer's unused joint-mode `out_cross_layer`
and MSGNet's unused `predict_linear` are excluded rather than offering no-op
controls. SCINet exposes no dense targets because its projections are convolutions.
The UI can reset edits even when a changed core configuration invalidates a path.
Browser visual review remains unavailable (no connected browser).

The coverage verdict below is unchanged: internal editing does not implement
any of the remaining reference-only methods.

## Verdict: partial implementation, not completion of all requested methods

15 of the 50 unique methods on the requested pages now have runnable TSLib-core
adaptations (including the three methods previously available only as lightweight
inspired templates). The collection contains two additional methods from page 4
of No Champions, so its total is 52. **35 exact-page methods, or 37 collection
entries, remain reference-only.** Runnable does not mean a published experiment
has been reproduced. No accuracy or GPU-performance claim is made.

## Verified integration

- All 15 cores: initialization, forward output and parameter gradients compared
  with separately loaded model files from the clean pinned TSLib checkout.
  Reproduce with `python scripts/audit_tslib.py /path/to/Time-Series-Library`.
  Shared layer dependencies use the vendored namespace-relocated files.
- Adapter output and input-gradient parity against its underlying core.
- CPU float32: windows 2, 10 and 33; widths 1, 4 and 3; batches 1 and 2;
  random, zero and constant inputs; finite outputs and gradients.
- Dense → model core → GRU hybrid training, actual optimizer updates to core
  parameters, serialized checkpoint reload and exact evaluation parity.
- Upstream initialization survives playground compilation; sequence-only blocks
  reject placement after flattening.
- Each of the 15 presets completes a real one-epoch playground validation job.
- Exposed non-default core settings train with a seven-step horizon.
- Full Python suite: 255 passed. Frontend: 4 passed; production build succeeded.
- Python wheel built successfully and includes all 15 models, the MIT license
  and provenance manifest.

The original single-block research checkpoint paths and the three lightweight
editable templates remain covered by the regression tests. Adapter tests alone
cannot establish mathematical equivalence to every original paper.

## Bugs found and fixed

1. MICN dereferenced absent decoder inputs: supply a zero decoder tensor, never
   future targets.
2. SCINet returns a prefix as well as predictions: select its final forecast
   before applying the adapter's crop.
3. PatchTST, iTransformer, MSGNet, SCINet and TimeXer used in-place operations
   that broke gradients through preceding trainable blocks. Replace those with
   out-of-place expressions; preserve numerical outputs and parameter gradients.
4. FreTS expects the string `'0'`, not integer `0`, to enable channel-frequency
   learning. Correct the configuration and test that path's gradients.
5. The playground initializer overwrote upstream initialization. Exclude the
   vendored subtrees while retaining initialization of surrounding blocks.
6. FiLM selected a global CUDA device. Use registered buffers and input-device
   allocation instead; GPU execution itself has not been tested here.
7. TimesNet and MSGNet could select the DC bin on tied zero FFT amplitudes.
   Exclude it explicitly to prevent zero-frequency division.
8. Remove inspector controls ignored by the selected upstream forecasting path.

## Remaining work (not implemented)

AMD, ARMD, CARD, CrossGNN, CycleNet, D3U, D3VAE, DeepAR, DeformTime,
FourierGNN, KooNPro, LCESN, LDT, LEDDAM, LTBoost, ModernTCN, N-BEATS,
One Fits All, P-sLSTM, Pathformer, SAMformer, SimpleTM, SOFTS, T-PatchGNN,
TEMPO, Time-LLM, TimeCMA, TimeGrad, TimeKAN, TimeMachine, Timer-XL, TS-LIF,
TSDiff, TVNet and WITRAN; additionally S-Mamba and xLSTMTime from page 4.

Some need more than a forward module: diffusion needs denoising objectives and
sampling; LLM methods need their appropriate pretrained assets/tokenization;
SAMformer needs its optimizer protocol; CycleNet needs correct cycle-phase
indices; T-PatchGNN needs timestamps and masks; LTBoost is non-neural.
These are implementation requirements, not substitutes for completed work.
DeformTime's checked upstream repository did not contain a license, so its source
was not copied into this repository. A licensed source or independent implementation
is still needed. Several remaining ordinary forecasters also simply remain undone.

## Untested scope

### Graph-native refactor (2026-09-06)

All 15 runnable TSLib cores now lower to editable executable graphs on the same
architecture canvas. The v1 source remains a parity reference and initialization
recipe, not a whole-model execution fallback. The graph audit compares outputs,
gradients, optimizer updates and checkpoints, plus all-model one-epoch jobs.
Runtime period selection and Crossformer segment merging remain dynamic; MSGNet
in-place masking was functionalized into explicit graph dependencies. See
[graph architecture](graph_architecture.md) for the contract, workflow and tests.

### Boundaries

Published benchmark scores, GPU/MPS execution, live Modal deployment, broad
hyperparameter sweeps and browser visual review. Modal image dependencies were
updated, but remote execution was not performed. No browser connection was
available in this session. See `method_collection.md` for padding, readout and
TSLib-variant caveats.
