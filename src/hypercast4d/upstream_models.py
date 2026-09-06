"""Pinned TSLib implementations exposed as composable sequence-to-sequence blocks.

These blocks preserve the upstream model internals. Their output is a learned
sequence representation for the surrounding playground architecture, not a claim
that an arbitrary hybrid reproduces the published end-to-end model.
"""

from importlib import import_module
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_MODELS = {
    name.lower(): name for name in (
        "DLinear", "PatchTST", "iTransformer", "TimesNet", "Crossformer", "FiLM",
        "FreTS", "LightTS", "MICN", "MSGNet", "SCINet", "SegRNN", "TimeMixer",
        "TimeXer", "TSMixer",
    )
}


def upstream_defaults(method: str) -> dict:
    """Only expose settings actually used by the selected forecasting path."""
    if method in {'dlinear', 'frets', 'film', 'scinet'}:
        return {}
    defaults = {'d_model': 32}
    if method != 'lightts':
        defaults['dropout'] = 0.1
    if method not in {'lightts', 'segrnn', 'micn'}:
        defaults['num_layers'] = 1
    if method in {'patchtst', 'itransformer', 'crossformer', 'msgnet', 'timexer'}:
        defaults['n_heads'] = 4
    return defaults


def upstream_config(steps: int, width: int, d_model: int, num_layers: int,
                    n_heads: int, dropout: float) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=steps, pred_len=steps,
        label_len=steps // 2, enc_in=width, dec_in=width, c_out=width,
        d_model=d_model, d_ff=d_model * 2, e_layers=num_layers, d_layers=1,
        n_heads=n_heads, dropout=dropout, activation="gelu", factor=3,
        embed="timeF", freq="h", moving_avg=25, individual=False,
        channel_independence=0, decomp_method="moving_avg", use_norm=1,
        down_sampling_layers=1, down_sampling_window=2, down_sampling_method="avg",
        top_k=3, num_kernels=3, seg_len=8, patch_len=8, features="M",
        conv_channel=16, skip_channel=16, gcn_depth=2, propalpha=0.05,
        node_dim=8, ratio=0.5,
    )


class UpstreamSequence(nn.Module):
    """Predict a same-length latent sequence; downstream blocks remain editable.

    Replication pads the observed history on the left to a multiple of 32 (at
    least 32), supporting patch/segment/multiscale divisibility. The upstream
    forecast is cropped back to the requested length. No future targets or
    calendar covariates are supplied. Shape adaptation is explicit and audited.
    """

    def __init__(self, method: str, steps: int, width: int, d_model: int = 32,
                 num_layers: int = 1, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.method, self.steps, self.width = method, steps, width
        self.padded_steps = max(32, ((steps + 31) // 32) * 32)
        self.config = upstream_config(self.padded_steps, width, d_model, num_layers, n_heads, dropout)
        if method == "frets":
            self.config.channel_independence = "0"
        module = import_module(f"hypercast4d._vendor.tslib.models.{UPSTREAM_MODELS[method]}")
        self.model = module.Model(self.config)

    def prepare_input(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.pad(inputs.transpose(1, 2), (self.padded_steps - self.steps, 0),
                     mode="replicate").transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        history = self.prepare_input(inputs)
        output = self.model(history, None, torch.zeros_like(history), None)
        if isinstance(output, tuple):
            output = output[0]
        # TSLib's SCINet returns a prefix as well as its forecast. Its own
        # experiment runner selects the final pred_len steps before loss.
        if self.method == "scinet":
            output = output[:, -self.padded_steps:, :]
        if output.ndim != 3 or output.shape[1:] != (self.padded_steps, self.width):
            raise RuntimeError(f"{self.method} returned unexpected shape {tuple(output.shape)}")
        return output[:, -self.steps:, :]
