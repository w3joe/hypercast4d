# Graph-native architecture playground

The Builder uses one React Flow canvas. A model is a presentation group of
executable operations, not a wrapper with a second internal viewer. Expand its
group to edit dense/conv layers, attention projections, normalization, branches,
and structural operations directly. All 15 existing TSLib presets convert;
the older paper and mechanism-inspired presets also remain loadable.

## Experiment workflow

The default workspace follows **Choose a model → Edit architecture → Run
experiment**. Graphs flow top-to-bottom, with input handles above layers and
output handles below. **Add layers** opens the palette. **Advanced tools** reveals
model insertion, import/export, weight sharing and grouping controls. **Arrange
top-down** reflows an older saved layout without changing the model or expanding
its groups. Layer and connection editing still use the same executable canvas.

### Swap Dense and HyperDense

Select a layer (or use **Find a layer**) and choose its **Layer type** in the
settings panel. Dense → HyperDense maps 32 output features to 8 hypercomplex
units, preserving the 32-wide output. Choose Quaternion, Coquaternion or Cl(1,1)
from **Algebra**. Switching back restores the equivalent real output width.

The switch checks every selected evaluation cell before committing, preserves
connections, group placement and bias, and rejects incompatible widths without
changing the graph. Input and output widths must both be divisible by four for
the HyperDense conversion. No padding/projection is silently inserted. Graph
HyperDense supports vector, sequence and higher-rank feature tensors; the v1
sequence-only contract is unchanged. Replacement weights are newly initialized;
a shared call becomes independent when switched. Undo restores the original node.

1. Load a preset, then **Clone to edit**. Expand a group or use **Expand all**.
2. Select a layer to change its settings. Compatible Linear/Conv input widths
   are inferred; incompatible joins produce errors rather than hidden adapters.
3. Drag operation handles to connect nodes. Select an edge to insert a unary
   operation, reconnect it, or delete it. Add/multiply require matching shapes;
   concatenate uses an explicit dimension. Replacements require matching input
   arity; otherwise add a new operation and reconnect it explicitly.
   Reshape, axis permutation and softmax are explicit palette operations too;
   reshape uses `0` to copy an input axis and `-1` to infer one dimension.
4. Use **Add model to canvas** for another independent subgraph. Connect its
   `Model input`, then join its output with another branch or choose a new
   forecast output. Disconnected branches do not execute.
5. Shift-select nodes to group, ungroup, duplicate, delete, or share weights.
   Duplicates have independent weights; calls sharing a reference must have
   compatible types and settings. Dense settings on shared calls update all
   calls. **Make weights independent** detaches an individual call.
6. Save a graph/layout, or export/import YAML/JSON. Undo/redo includes topology,
   settings and presentation. Delete/Backspace and Ctrl/Cmd-Z work when the
   canvas has focus. Fit/zoom/pan controls operate on the single viewport.
7. Training is enabled only after every selected window/horizon validates.
   The server repeats this check before queueing local or Modal execution.
   Drafts with incomplete connections can still be saved and repaired.

## Execution contract

Version 2 stores `nodes`, input-port `edges`, `output`, initialization `sources`,
and a pinned lowering `revision`. Groups and saved `view` positions/collapse
state affect presentation only. Node-array order is the deterministic tie-break
between independent branches, including dropout calls, and is part of the hash.

The compiler symbolically lowers trusted source recipes into individual
operators. Only the persisted graph's reachable nodes and edges execute;
the original whole-model forward is not an execution fallback. No arbitrary
Python expressions, imports, or user-supplied callable names are evaluated.
The recipes preserve initialization and window-dependent constants. Explicit
module/state references preserve shared weights, including raw FFT parameters.
Nonlearned tensor state is registered as buffers, not trainable parameters.

Period selection, fold/unfold and segment-layout operations run on actual inputs
at execution time. They are not frozen from a sample trace. Recurrent and
spectral kernels remain named operations; ordinary trainable projections and
convolutions around them remain exposed. Structural operators with no settings
can be rewired or replaced rather than edited through arbitrary Python code.
Expected ports are available even for an invalid draft via the describe API.

V1 loading, execution, hashing and checkpoint behavior are unchanged. Conversion
is in memory, preserves existing internal overrides, and the UI clears the old
record ID so saving creates a separate v2 record. V2 checkpoints are compiled
from the same graph before restoring their state dictionaries.

The TSLib adapter still pads to a multiple of 32, predicts a same-length latent
sequence, crops it, then uses the playground readout. This refactor does not turn
the adapters into reproductions of published benchmark protocols, and does not
implement the remaining reference-only methods in the collection.

## Verification

`tests/test_graph_architecture.py` covers all 15 cores at windows 2/10/33,
horizons 1/3/7 and feature subsets, comparing source versus graph outputs,
input/parameter gradients, an Adam update, seeded training dropout, and exact
checkpoint round trips. It also checks every legacy preset, nondefault core
settings, internal overrides, topology edits, shared/independent weights,
hybrid model branches, dynamic period changes, invalid edges and draft repair.

`tests/test_playground.py` runs one-epoch graph jobs for all 15 cores and tests
migration, saved layout and all-cell server validation. Frontend tests cover
same-canvas editing, persisted specifications, navigation/undo, invalid-cell
run gating, local/Modal controls, real ELK grouping/proxy projection, reconnects,
independent duplication and model insertion. Component tests mock the React Flow
surface; they are not a substitute for a real browser visual/interaction review.

GPU/MPS execution, a live Modal job and browser visual review remain untested.
The browser connector reported no connected browsers during this implementation.
