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

## Runnable adaptations

Three new presets use the existing local/Modal execution paths, chronological
splits, validation metrics and held-out-test controls. Their parameters are
editable in the Inspector. These are compact implementations of core mechanisms,
not reproductions of the papers' architectures, training setups or scores.

| Preset | Mechanism | Scope and differences |
| --- | --- | --- |
| DLinear | Replicate-padded moving average; separate trend and seasonal projections | Target channel only; odd smoothing kernel; exogenous channels unused |
| PatchTST | Temporal patches, learned positions, attention, flattening projection | Target-only channel-independent adaptation; PyTorch LayerNorm encoder, non-affine window normalization, no residual attention or pretraining |
| iTransformer | Whole-window variable tokens and cross-variable attention | All selected variables inform the target; PyTorch encoder, no calendar covariates, target-only loss |

Implementation references: [DLinear](https://github.com/cure-lab/LTSF-Linear),
[PatchTST](https://github.com/yuqinie98/PatchTST), and
[iTransformer](https://github.com/thuml/iTransformer). The implementations here
are written locally without vendoring upstream code.

Each forecaster must be the only pipeline block, use levels input with target
feature 0 first, and use a direct head without zero initialization. Its own output
projection produces the forecast, so no extra head is applied. Invalid edits
produce validation errors. PatchTST pads short windows to accommodate its patch
size and rejects strides that would skip gaps between patches.

DeformTime and the other reference entries still require dedicated implementations.
In particular, a GRU alone is not DeformTime: its variable/temporal deformable
attention and neighbourhood-aware embedding are essential.

Validation covers optimizer updates, finite gradients, constant inputs,
short windows and horizons longer than the input, checkpoint reloads,
channel independence versus cross-variable dependence, invalid configurations,
and end-to-end one-epoch jobs through the playground runner. These checks establish
execution correctness, not predictive superiority or paper fidelity.
