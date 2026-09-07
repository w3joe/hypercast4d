# Playground method collection

Open **Builder → Method collection** to search 52 deduplicated entries, filter by
architecture family or source, and follow links to the cited PDF page. The
collection includes 51 neural/linear-network methods and LTBoost, explicitly
marked as a non-neural baseline. Reference-only entries do not offer a Run button.

## Sources and coverage

* [Liao, Xuan and Ma, Frontiers of Computer Science (2026)](https://link.springer.com/content/pdf/10.1007/s11704-025-50947-3.pdf#page=13): all 44 rows of Figure 2, page 13. The figure was visually inspected because the model names are not present in its extracted text. Common spelling variants are normalized; source membership is preserved.
* [Brigato et al., TMLR (January 2026)](https://arxiv.org/pdf/2502.14045v2#page=3): page 3's models, including N-BEATS mentioned in the text, plus S-Mamba, xLSTMTime and ModernTCN from the evaluation table on page 4. The accepted title includes “Supervised”; the older position-paper version is not used.
* [Shu and Lampos, TMLR (April 2025)](https://arxiv.org/pdf/2406.07438v3#page=7): DeformTime and all nine neural/linear baselines named in Section 4.1 on page 7. Persistence is already part of the playground evaluation and is not a neural-network entry.

The collection is implemented in `src/hypercast4d/method_collection.py` and served
under `method_collection` in `/api/v1/catalog`. Each entry keeps all matching
sources, page numbers and an explicit implementation status.

## TSLib-core adaptations (15 methods)

The library now loads pinned TSLib cores for **DLinear, PatchTST, iTransformer,
TimesNet, Crossformer, FiLM, FreTS, LightTS, MICN, MSGNet, SCINet, SegRNN,
TimeMixer, TimeXer and TSMixer**. Load their presets from the library or use
**Add model to canvas**. Their core adapters accept and return
`[batch, time, channels]`; trainable blocks can come before and after them.

These use MIT-licensed code at TSLib revision
`4e938a1767106324dd753b2a44832bf870a0252e`, with provenance and the license in
`src/hypercast4d/_vendor/tslib`. **These are TSLib variants, not claims of
author-code or benchmark reproduction.** TSLib's TSMixer in particular is a
simplified variant rather than Google's original implementation.

Integration differences are explicit: the observed history is replication-padded
on the left to at least 32 steps and a multiple of 32; the core predicts that
many steps, and its final original-window-length outputs form the next block's
sequence. A separate playground readout maps that representation to the requested
target horizon. SCINet's extra prefix is discarded, as in TSLib's experiment
runner. MICN receives zeros as decoder values. No future targets or calendar
covariates are provided. This makes the cores composable, but adds padding and
a readout relative to each standalone forecaster.

Editable core settings include width, depth, heads and dropout where the selected
forecasting path uses them. Width must be a multiple of eight and divisible by
heads. Core-specific settings not exposed in the inspector retain defaults in
`upstream_models.py`. DLinear, FreTS, FiLM and SCINet currently have fixed core
settings; surrounding blocks, feature selection and heads remain editable.

## Current Builder: executable graph editing

The Builder now uses a single executable architecture canvas, including model
internals. Expand a model group, select a dense/conv layer, change its settings,
or add/delete/replace/reconnect operations and residual branches. Use
**Clone to edit** for a locked preset. Graphs are validated across every selected
evaluation cell before training. [Graph architecture](graph_architecture.md)
documents this workflow, save/import compatibility, sharing and verification.

## Historical v1 internal editor (superseded in the Builder)

The following describes the earlier v1 editor and its retained compatibility
APIs, not the current graph-native Builder UI.

Click **Expand model** directly on a TSLib model block in the architecture canvas.
The nested canvas shows an observed tensor-dependency graph, including branches
and residual joins. Click a green dense node to select it and open its settings
in the Inspector. Use **Edit a containing dense stack** below the graph to select
an entire MLP (for example `model.0.temporal` in TSMixer). Change hidden widths,
activation, dropout or biases, then click **Apply internal edit**. The graph is
regenerated from the edited model and shows its new internal dimensions. Use
**Collapse model** to return to the compact pipeline; expansion/collapse does
not change the architecture or its hash. Edited architectures receive
a `[custom]` name suffix and a custom-internals badge. The Inspector's searchable
module list remains available as an alternative, including reset controls when
a configuration is invalid.

The graph is generated on demand from **one deterministic synthetic CPU evaluation
forward pass**, using the selected window and the core's actual input width after
preceding blocks. It never reads the user's dataset. Leaf modules are atomic nodes;
single-source tensor operations (including reshapes) are collapsed into edges.
Observed multi-source tensor operations are explicit junctions. Shared modules can
appear at multiple call sites but still refer to the same editable path. This is
an execution preview, not a static graph of every possible input-dependent branch.
Non-dense nodes and all connections remain read-only. A failed trace is reported
explicitly rather than replaced with guessed edges. Large graphs scroll within
the expanded block; previews have a 1500-node limit. No preview metadata is saved
as part of the trainable architecture.

For example, select TSMixer's `model.0.temporal`, enter `64, 16`, and apply:
the *internal* temporal MLP becomes `32 → 64 → 16 → 32` for the padded quick
window. Its original residual connection remains in place. This does not append
a new layer after TSMixer. Individual Linear modules can likewise be replaced
with dense stacks; leave hidden widths empty for one linear projection.

This first pass edits **Linear and pure dense Sequential submodules**, not an
arbitrary computation graph. Input and output dimensions of the edited subtree
stay fixed; only its interior widths and operations change. Activations/dropout
follow hidden layers, not the final linear output. Existing non-dense operations,
attention wiring and connections remain unchanged. SCINet's projections are
convolutions and therefore do not appear as editable dense layers. Calendar
embeddings and other modules unused by the selected forecasting path are excluded.

Edits are stored in `layers[].internal_overrides`, included in architecture
hashes, JSON import/export, saved architectures, run requests and checkpoint
reconstruction. Applying edits creates freshly initialized replacement modules;
it does not transplant weights from earlier runs. **Restore original module**
removes an edit; with all edits removed the original structure, checkpoint layout
and initialization behavior are unchanged. A parent dense stack and one of its
children cannot both be overridden. Reset buttons remain available if changing
core depth makes a saved path invalid. Module paths describe containment, not
execution order. Other core configurations may change their boundary dimensions.

## Lightweight mechanism-inspired templates

Three lightweight presets use the existing local/Modal execution paths, chronological
splits, validation metrics and held-out-test controls. Their parameters are
editable in the Inspector. These are compact implementations of core mechanisms,
not reproductions of the papers' architectures, training setups or scores.

| Preset | Mechanism | Scope and differences |
| --- | --- | --- |
| DLinear-inspired | Decomposition → temporal projection → flatten → forecast head | Target-only default; trend and seasonal features remain separately editable; extra learned readout differs from DLinear |
| PatchTST-inspired | Patch embedding → temporal attention → flatten → forecast head | Target-only default; additional input features form joint patches; no window normalization, residual attention or pretraining |
| iTransformer-inspired | Cross-variable attention → flatten → forecast head | Variable tokens are projected back to time for composition; all features feed the readout; no window normalization or calendar covariates |

Implementation references: [DLinear](https://github.com/cure-lab/LTSF-Linear),
[PatchTST](https://github.com/yuqinie98/PatchTST), and
[iTransformer](https://github.com/thuml/iTransformer). These three lightweight
implementations are written locally; the separate TSLib-core presets above
vendor upstream code.

## Compose your own model

The **Research blocks** palette exposes five reusable blocks. Every block takes
and returns a sequence, so GRU, TCN, Dense, HyperDense and other processing blocks
can precede or follow them. Place a sequence reduction (Flatten, Mean pool or
Last state) before the forecast head. Dense layers can also follow a reduction.

| Block | Input → output | Editable settings |
| --- | --- | --- |
| Trend / seasonal split | `[B,T,C]` → `[B,T,2C]` | Odd moving-average kernel; output channels are seasonal followed by trend |
| Patch embedding | `[B,T,C]` → `[B,P,D]` | Patch size, stride, embedding width; learned positions and replication padding |
| Temporal attention | `[B,T,C]` → `[B,T,D]` | Embedding width, heads, depth, dropout; attends across time or patch tokens |
| Cross-variable attention | `[B,T,C]` → `[B,T,C]` | Embeds each variable's history, attends across variables, projects back to time |
| Temporal projection | `[B,T,C]` → `[B,U,C]` | Output steps; independently learned time-axis projection for each channel |

For example: **Patch embedding → Temporal attention → HyperDense → GRU → Last
state → Forecast head** combines patch attention with hypercomplex feature mixing
and recurrent processing. Keep the width entering HyperDense divisible by the
selected algebra dimension (2, 3, 4, or 8).

Feature subsets and order, levels/centered/difference inputs, and all forecast
heads remain editable. Validation rejects shape incompatibilities and invalid
parameters, not model families. Attention width must be divisible by head count;
patch stride cannot exceed patch size. Shape traces update for every block.

Existing saved single-block `dlinear`, `patchtst` and `itransformer` architectures
with their original direct, non-zero-initialized heads retain their prior model
and checkpoint layout. When used in multi-block pipelines, those legacy names
act as sequence encoders. New presets use the explicit blocks above. These are
different architectures with distinct hashes; older benchmark results should not
be treated as results for the new presets.

DeformTime and the other reference entries still require dedicated implementations.
In particular, a GRU alone is not DeformTime: its variable/temporal deformable
attention and neighbourhood-aware embedding are essential.

Validation covers optimizer updates, finite gradients, constant inputs,
short windows and horizons longer than the input, checkpoint reloads,
channel independence versus cross-variable dependence, hybrid training with GRU,
TCN, Dense and HyperDense, vector/sequence incompatibilities, legacy checkpoints,
and end-to-end one-epoch jobs through the playground runner. These checks establish
execution correctness, not predictive superiority or paper fidelity.
