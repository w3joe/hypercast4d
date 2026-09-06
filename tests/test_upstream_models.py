"""Adapter contract audit; this does not certify published benchmark scores."""
import copy
import io

import pytest
import torch

from hypercast4d.architecture import ArchitectureError, build_architecture, presets
from hypercast4d.upstream_models import UPSTREAM_MODELS, UpstreamSequence
from hypercast4d.upstream_models import upstream_defaults


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def spec(method):
    return next(p for p in presets() if p['preset_id'] == f'tslib-{method}')


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
@pytest.mark.parametrize('window,width,batch', [(2, 1, 1), (10, 4, 2), (33, 3, 1)])
def test_shape_gradients_constants_and_no_input_mutation(method, window, width, batch):
    torch.manual_seed(31)
    core = UpstreamSequence(method, window, width, dropout=0).eval()
    x = torch.randn(batch, window, width, requires_grad=True)
    before = x.detach().clone()
    output = core(x)
    assert output.shape == (batch, window, width)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.equal(x.detach(), before)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = [p.grad for p in core.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    with torch.no_grad():
        for value in (0., 1.):
            assert torch.isfinite(core(torch.full_like(x, value))).all()


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_adapter_matches_core_and_input_gradient(method):
    torch.manual_seed(23)
    adapter = UpstreamSequence(method, 10, 4, dropout=0).eval()
    reference = copy.deepcopy(adapter.model).eval()
    x = torch.randn(2, 10, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    history = adapter.prepare_input(x_ref)
    expected = reference(history, None, torch.zeros_like(history), None)[:, -10:, :]
    actual = adapter(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_train_checkpoint_and_composition(method):
    architecture = spec(method)
    architecture['layers'].insert(0, {'id': 'mix', 'type': 'dense', 'params': {'units': 4}})
    architecture['layers'].insert(-1, {'id': 'memory', 'type': 'gru', 'params': {'hidden_size': 8}})
    model = build_architecture(architecture, 10, 3)
    x, target = torch.randn(2, 10, 4), torch.randn(2, 3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = (model(x) - target).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if '.model.' in n)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = build_architecture(architecture, 10, 3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_initialization_preserved_and_sequence_required(method):
    torch.manual_seed(43)
    reference = UpstreamSequence(method, 10, 4)
    torch.manual_seed(43)
    compiled = build_architecture(spec(method), 10, 3)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, compiled.layers[0].state_dict()[key], rtol=0, atol=0)
    architecture = spec(method)
    architecture['layers'].reverse()
    with pytest.raises(ArchitectureError, match='requires a sequence'):
        build_architecture(architecture, 10, 3)


@pytest.mark.parametrize('module_name', ['TimesNet', 'MSGNet'])
def test_fft_periods_never_select_dc_on_zero_input(module_name):
    from importlib import import_module
    import numpy as np
    module = import_module(f'hypercast4d._vendor.tslib.models.{module_name}')
    periods, weights = module.FFT_for_Period(torch.zeros(2, 32, 4), 3)
    assert np.isfinite(periods).all() and (periods > 0).all()
    assert torch.isfinite(weights).all()


def test_frets_uses_channel_frequency_learning():
    core = UpstreamSequence('frets', 10, 4)
    assert core.model.channel_independence == '0'
    core(torch.randn(2, 10, 4)).square().mean().backward()
    assert core.model.r1.grad is not None


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_exposed_nondefault_settings_train(method):
    architecture = spec(method)
    values = {'d_model': 24, 'n_heads': 8, 'num_layers': 2, 'dropout': 0.2}
    architecture['layers'][0]['params'] = {key: values[key] for key in upstream_defaults(method)}
    model = build_architecture(architecture, 11, 7)
    result = model(torch.randn(2, 11, 4))
    assert result.shape == (2, 7)
    result.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
