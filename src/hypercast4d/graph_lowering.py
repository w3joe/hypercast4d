"""Source-symbolic lowering, never an observed/sample-value execution trace.

Shape/data-dependent routines that FX cannot express are explicit registered
kernels below. Trainable convolutions/projections remain separate graph calls.
"""
from contextlib import ExitStack
import math
import threading
from unittest.mock import patch

import einops
import torch
from torch import nn
from torch.fx import GraphModule, Proxy, Tracer

from .upstream_models import UpstreamSequence


LOWERING_LOCK = threading.RLock()


def period_select(x, k):
    amplitudes = torch.fft.rfft(x, dim=1).abs()
    frequency = amplitudes.mean(0).mean(-1)
    frequency[0] = -torch.inf
    indices = frequency.topk(k).indices
    return x.shape[1] // indices, amplitudes.mean(-1)[:, indices]


def period_fold(x, period):
    period = int(period)
    batch, length, width = x.shape
    padded = math.ceil(length / period) * period
    x = torch.nn.functional.pad(x, (0, 0, 0, padded - length))
    return x.reshape(batch, padded // period, period, width).permute(0, 3, 1, 2).contiguous()


def period_unfold(x, original):
    return x.permute(0, 2, 3, 1).reshape(original.shape[0], -1, original.shape[2])[:, :original.shape[1]]


def period_attention_fold(x, period):
    folded = period_fold(x, period)
    return folded.permute(0, 2, 3, 1).reshape(-1, int(period), x.shape[2])


def period_attention_unfold(x, original):
    return x.reshape(original.shape[0], -1, original.shape[2])[:, :original.shape[1]]


def period_combine(outputs, weights):
    stacked = torch.stack(outputs, dim=-1)
    weights = torch.softmax(weights, dim=1)[:, None, None, :]
    return (stacked * weights).sum(-1)


def even_pad(x):
    return torch.cat([x, x[:, -1:, :]], dim=1) if x.shape[1] % 2 else x


def segment_merge_layout(x, window):
    remainder = x.shape[2] % window
    if remainder:
        x = torch.cat((x, x[:, :, -(window - remainder):, :]), dim=-2)
    return torch.cat([x[:, :, i::window, :] for i in range(window)], dim=-1)


def even_trim(x, original):
    return x[:, :original.shape[1] // 2, :] if original.shape[1] % 2 else x


def interleave(even, odd):
    length = min(even.shape[1], odd.shape[1])
    result = torch.stack([even[:, :length], odd[:, :length]], dim=2).flatten(1, 2)
    return torch.cat([result, even[:, length:]], dim=1)


def multi_decompose(x, kernels):
    from ._vendor.tslib.layers.Autoformer_EncDec import series_decomp_multi
    return series_decomp_multi(kernels).to(x.device)(x)


def positional_encoding(x, inverse_timescales):
    scaled = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * inverse_timescales[None, :]
    signal = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
    return signal[None, :, :x.shape[2]]


def _has_proxy(value):
    if isinstance(value, Proxy):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_proxy(x) for x in value)
    if isinstance(value, dict):
        return any(_has_proxy(x) for x in value.values())
    return False


class SourceTracer(Tracer):
    def __init__(self):
        super().__init__(autowrap_functions=(einops.rearrange, einops.repeat))
        self.norm_stats = {}
        self.owners = []

    def create_proxy(self, *args, **kwargs):
        result = super().create_proxy(*args, **kwargs)
        if self.owners:
            result.node.meta['owner'] = self.owners[-1]
        return result

    def kernel(self, function, *args, **kwargs):
        return self.create_proxy('call_function', function, args, kwargs)

    def is_leaf_module(self, module, path):
        if type(module).__name__ in {'HyperDense', 'HiPPO_LegT', 'SpectralConv1d', 'PositionalEmbedding', 'FixedEmbedding'}:
            return True
        return super().is_leaf_module(module, path)

    def call_module(self, module, forward, args, kwargs):
        path = self.submodule_paths.get(module, self.owners[-1] if self.owners else '')
        self.owners.append(path)
        try:
            return self.lower_module(module, forward, args, kwargs)
        finally:
            self.owners.pop()

    def lower_module(self, module, forward, args, kwargs):
        kind = type(module).__name__
        package = type(module).__module__
        if isinstance(module, nn.AvgPool1d) and module not in self.submodule_paths:
            return self.kernel(torch.nn.functional.avg_pool1d, args[0], module.kernel_size,
                               module.stride, module.padding, module.ceil_mode, module.count_include_pad)
        if isinstance(module, UpstreamSequence):
            x = module.prepare_input(args[0])
            return module.model(x, None, torch.zeros_like(x), None)[:, -module.steps:, :]
        if kind == 'series_decomp_multi':
            return self.kernel(multi_decompose, args[0], module.kernel_size)
        if kind == 'SegMerging' and package.endswith('Crossformer_EncDec'):
            return module.linear_trans(module.norm(self.kernel(segment_merge_layout, args[0], module.win_size)))
        if kind == 'Normalize' and package.endswith('StandardNorm'):
            x, mode = args
            if module.non_norm:
                return x
            if mode == 'norm':
                mean = x[:, -1:, :] if module.subtract_last else x.mean(1, keepdim=True).detach()
                std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + module.eps).detach()
                self.norm_stats[id(module)] = mean, std
                x = (x - mean) / std
                return x * module.affine_weight + module.affine_bias if module.affine else x
            mean, std = self.norm_stats[id(module)]
            if module.affine:
                x = (x - module.affine_bias) / (module.affine_weight + module.eps * module.eps)
            return x * std + mean
        if kind == 'TimesBlock':
            x = args[0]
            periods, weights = self.kernel(period_select, x, module.k)
            outputs = [self.kernel(period_unfold, module.conv(self.kernel(period_fold, x, periods[i])), x)
                       for i in range(module.k)]
            return self.kernel(period_combine, outputs, weights) + x
        if kind == 'ScaleGraphBlock':
            x = args[0]
            periods, weights = self.kernel(period_select, x, module.k)
            outputs = []
            for i in range(module.k):
                x = module.gconv[i](x)
                out = self.kernel(period_attention_fold, x, periods[i])
                out = module.gelu(module.norm(module.att0(out)))
                outputs.append(self.kernel(period_attention_unfold, out, x))
            return self.kernel(period_combine, outputs, weights) + x
        if kind == 'SCINet' and package.endswith('.SCINet'):
            x = args[0]
            even, odd = module.working_block(self.kernel(even_pad, x))
            odd = self.kernel(even_trim, odd, x)
            if module.current_level:
                even, odd = module.SCINet_Tree_even(even), module.SCINet_Tree_odd(odd)
            return self.kernel(interleave, even, odd)
        if kind == 'Model' and package.endswith('.SCINet'):
            x = args[0]
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - mean) / std
            x = x + self.kernel(positional_encoding, x, module.inv_timescales)
            out = module.projection_1(module.sci_net_1(x) + x)
            if module.num_stacks != 1:
                combined = torch.cat((x, out), dim=1)
                out = module.projection_2(module.sci_net_2(combined) + combined)
            return torch.cat([torch.zeros_like(args[0]), out * std + mean], dim=1)
        return super().call_module(module, forward, args, kwargs)


def lower_source(model):
    """Lower registered source to a symbolic program; data-dependent bools fail."""
    with LOWERING_LOCK, ExitStack() as stack:
        tracer = SourceTracer()
        for name in ('zeros', 'ones', 'arange', 'empty', 'full', 'tensor', 'eye', 'reshape'):
            original = getattr(torch, name)
            def factory(*args, _original=original, **kwargs):
                if _has_proxy((args, kwargs)):
                    return tracer.kernel(_original, *args, **kwargs)
                return _original(*args, **kwargs)
            stack.enter_context(patch.object(torch, name, factory))
        graph = tracer.trace(model)
        # Make mutation dependencies explicit: attention reads the result of
        # masked_fill_, even when the source ignores that method's return value.
        sequence = list(graph.nodes)
        for index, node in enumerate(sequence):
            if node.op == 'call_method' and node.target in {'masked_fill_', 'add_', 'sub_', 'mul_', 'div_'}:
                base = node.args[0]
                node.target = str(node.target)[:-1]
                for later in sequence[index + 1:]:
                    later.replace_input_with(base, node)
        # Functional dropout must follow GraphModule.train()/eval(), not capture
        # the source module's mode at conversion time.
        for node in list(graph.nodes):
            if node.op == 'call_function' and node.target in {
                torch.nn.functional.dropout, torch.nn.functional.dropout1d,
                torch.nn.functional.dropout2d, torch.nn.functional.dropout3d,
            }:
                with graph.inserting_before(node):
                    training = graph.get_attr('training')
                if len(node.args) > 2:
                    node.args = (*node.args[:2], training, *node.args[3:])
                else:
                    node.kwargs = {**node.kwargs, 'training': training}
    return GraphModule(model, graph)
