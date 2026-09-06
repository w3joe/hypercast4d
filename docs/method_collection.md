# Playground method collection

Open **Builder → Method library** to search 52 deduplicated entries, filter by
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
TimeMixer, TimeXer and TSMixer**. Find them under **TSLib model blocks** in the
palette or load their editable presets from the library. All accept and return
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
and recurrent processing. Keep the width entering HyperDense divisible by four.

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
