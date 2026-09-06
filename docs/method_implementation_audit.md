# Method implementation audit — 2026-09-06

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

Published benchmark scores, GPU/MPS execution, live Modal deployment, broad
hyperparameter sweeps and browser visual review. Modal image dependencies were
updated, but remote execution was not performed. No browser connection was
available in this session. See `method_collection.md` for padding, readout and
TSLib-variant caveats.
