import copy
import io
import json

import pytest
import torch

from hypercast4d.architecture import architecture_hash, build_architecture, normalize_architecture_spec, presets, validate_architecture
from hypercast4d.internal_editing import normalize_internal_overrides
from hypercast4d.upstream_models import UPSTREAM_MODELS, UpstreamSequence


EDIT = {'hidden_units': [12, 8], 'activation': 'gelu', 'dropout': 0.1, 'bias': True}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def spec(method):
    return next(p for p in presets() if p['preset_id'] == f'tslib-{method}')


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_every_advertised_internal_target_can_train(method):
    original = UpstreamSequence(method, 10, 4)
    # SCINet has convolutional projections, not nn.Linear modules.
    if not original.internal_targets:
        assert method == 'scinet'
    for target in original.internal_targets:
        path = target['path']
        model = UpstreamSequence(method, 10, 4, internal_overrides={path: EDIT})
        inputs = torch.randn(2, 10, 4, requires_grad=True)
        result = model(inputs)
        assert result.shape == (2, 10, 4), (method, path)
        result.square().mean().backward()
        parameters = list(model.model.get_submodule(path).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), (method, path)
        assert torch.isfinite(inputs.grad).all(), (method, path)


def test_existing_dense_stack_width_changes_inside_tsmixer_and_reloads():
    architecture = spec('tsmixer')
    original = copy.deepcopy(architecture)
    architecture['layers'][0]['internal_overrides'] = {'model.0.temporal': EDIT}
    architecture['layers'].insert(-1, {'id': 'gru', 'type': 'gru', 'params': {'hidden_size': 8}})
    model = build_architecture(architecture, 10, 3)
    dense = model.layers[0].model.model[0].temporal
    assert dense[0].out_features == 12 and dense[3].out_features == 8
    x = torch.randn(2, 10, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = dense[0].weight.detach().clone()
    model(x).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, dense[0].weight)
    serialized = json.loads(json.dumps(architecture))
    restored = build_architecture(serialized, 10, 3)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model.eval()(x), restored.eval()(x), rtol=0, atol=0)
    assert architecture_hash(original) != architecture_hash(architecture)
    metadata = validate_architecture(architecture, 10, 3)['internals']['core']
    edited = next(t for t in metadata if t['path'] == 'model.0.temporal')
    assert edited['override'] == EDIT
    child = next(t for t in metadata if t['path'] == 'model.0.temporal.0')
    assert child['blocked_by'] == 'model.0.temporal'


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_empty_edits_preserve_hash_weights_outputs_and_random_state(method):
    architecture = spec(method)
    torch.manual_seed(77)
    baseline = build_architecture(architecture, 10, 3).eval()
    state = torch.get_rng_state().clone()
    changed = copy.deepcopy(architecture)
    changed['layers'][0]['internal_overrides'] = {}
    assert architecture_hash(changed) == architecture_hash(architecture)
    torch.manual_seed(77)
    restored = build_architecture(changed, 10, 3).eval()
    assert torch.equal(state, torch.get_rng_state())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    x = torch.randn(2, 10, 4)
    assert torch.equal(baseline(x), restored(x))


@pytest.mark.parametrize('raw', [
    [], {'bad.path': {'hidden_units': [0]}}, {'x': {'hidden_units': [3.5]}},
    {'x': {'hidden_units': [True]}}, {'x': {'hidden_units': [2]*5}},
    {'x': {'activation': 'exec'}}, {'x': {'dropout': float('nan')}},
    {'x': {'bias': 'false'}}, {'x': {'arbitrary': 2}},
    {'x': EDIT, 'x.0': EDIT}, {'a..b': EDIT},
])
def test_reject_invalid_internal_specs(raw):
    with pytest.raises(ValueError):
        normalize_internal_overrides(raw)


def test_reject_nonexistent_paths_non_dense_targets_and_wrong_block_type():
    for path in ('missing', 'model.0', '__class__'):
        with pytest.raises(ValueError, match='not an editable dense'):
            UpstreamSequence('tsmixer', 10, 4, internal_overrides={path: EDIT})
    architecture = spec('tsmixer')
    architecture['layers'][-1]['internal_overrides'] = {'x': EDIT}
    with pytest.raises(ValueError, match='only on TSLib'):
        normalize_architecture_spec(architecture)


def test_api_inspection_save_reload_and_invalid_edit(tmp_path):
    from fastapi.testclient import TestClient
    from hypercast4d.playground import create_app
    architecture = spec('tsmixer')
    with TestClient(create_app(tmp_path / 'playground', tmp_path)) as client:
        response = client.post('/api/v1/architectures/validate', json={'architecture': architecture})
        assert response.status_code == 200
        assert any(t['path'] == 'model.0.temporal' for t in response.json()['internals']['core'])
        architecture['layers'][0]['internal_overrides'] = {'model.0.temporal': EDIT}
        saved = client.post('/api/v1/architectures', json=architecture)
        assert saved.status_code == 200
        records = client.get('/api/v1/architectures').json()
        assert records[0]['spec']['layers'][0]['internal_overrides'] == {'model.0.temporal': EDIT}
        architecture['layers'][0]['internal_overrides'] = {'no_such_module': EDIT}
        bad = client.post('/api/v1/architectures/validate', json={'architecture': architecture})
        assert bad.status_code == 422
        assert 'not an editable dense' in bad.json()['detail']
