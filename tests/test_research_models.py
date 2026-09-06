import copy

import pytest
import torch

from hypercast4d.architecture import ArchitectureError, build_architecture, presets, validate_architecture
from hypercast4d.method_collection import method_collection, SURVEY_FAMILIES
from hypercast4d.research_models import DLinear


def spec(kind):
    return next(item for item in presets() if item['preset_id'] == f'research-{kind}')


@pytest.mark.parametrize('kind', ['dlinear', 'patchtst', 'itransformer'])
@pytest.mark.parametrize('window,horizon', [(2, 1), (10, 5), (21, 30)])
def test_research_models_train_and_reload(kind, window, horizon):
    torch.manual_seed(7)
    model = build_architecture(spec(kind), window, horizon)
    inputs = torch.randn(3, window, 4)
    target = torch.randn(3, horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    restored = build_architecture(spec(kind), window, horizon)
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    assert torch.equal(model(inputs), restored(inputs))
    assert model(inputs).shape == (3, horizon)
    assert torch.isfinite(model(torch.ones_like(inputs))).all()
    assert validate_architecture(spec(kind), window, horizon)['receptive_field'] == window


@pytest.mark.parametrize('kind', ['dlinear', 'patchtst', 'itransformer'])
def test_exogenous_dependency_matches_method(kind):
    torch.manual_seed(9)
    model = build_architecture(spec(kind), 10, 3).eval()
    inputs = torch.randn(2, 10, 4)
    altered = inputs.clone()
    altered[:, :, 1:] = torch.randn_like(altered[:, :, 1:]) * 3
    equal = torch.equal(model(inputs), model(altered))
    assert equal == (kind != 'itransformer')


def test_dlinear_decomposition_reconstructs_last_value():
    model = DLinear(5, 2, 25)
    with torch.no_grad():
        for projection in (model.seasonal, model.trend):
            projection.weight.zero_()
            projection.weight[:, -1] = 1
            projection.bias.zero_()
    inputs = torch.randn(4, 5, 4)
    assert torch.allclose(model(inputs), inputs[:, -1, 0:1].expand(-1, 2), atol=1e-6)


@pytest.mark.parametrize('change,match', [
    ({'kernel_size': 4}, 'odd'),
])
def test_invalid_dlinear_parameters(change, match):
    architecture = spec('dlinear')
    architecture['layers'][0]['params'] = change
    with pytest.raises(ArchitectureError, match=match):
        build_architecture(architecture, 10, 1)


def test_attention_rejects_invalid_heads():
    architecture = spec('itransformer')
    architecture['layers'][0]['params']['n_heads'] = 3
    with pytest.raises(ArchitectureError, match='divisible'):
        build_architecture(architecture, 10, 1)


@pytest.mark.parametrize('kind', ['dlinear', 'patchtst', 'itransformer'])
@pytest.mark.parametrize('extra,params', [
    ('gru', {'hidden_size': 8}), ('tcn', {'channels': 8}),
    ('dense', {'units': 8}), ('hyper_dense', {'units': 2}),
])
def test_research_presets_accept_hybrid_blocks(kind, extra, params):
    architecture = spec(kind)
    # Supply four features for HyperDense's genuine divisibility requirement.
    architecture['input'] = {'representation': 'differences', 'feature_order': [2, 0, 3, 1]}
    architecture['layers'].insert(-1, {'id': 'hybrid', 'type': extra, 'params': params})
    architecture['head'] = {'type': 'cumulative_residual', 'zero_initialize': False}
    model = build_architecture(architecture, 11, 5)
    output = model(torch.randn(3, 11, 4))
    assert output.shape == (3, 5)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize('block', ['decomposition', 'patch_embedding', 'temporal_attention', 'variable_attention', 'temporal_projection'])
def test_sequence_blocks_reject_vector_inputs(block):
    architecture = spec('dlinear')
    architecture['layers'] = [
        {'id': 'reduce', 'type': 'flatten', 'params': {}},
        {'id': 'invalid', 'type': block, 'params': {}},
    ]
    with pytest.raises(ArchitectureError, match='requires a sequence'):
        build_architecture(architecture, 10, 1)


def test_decomposition_preserves_signal_and_temporal_projection_shape():
    from hypercast4d.research_models import Decomposition, TemporalProjection
    inputs = torch.randn(2, 7, 3)
    split = Decomposition(25)(inputs)
    assert split.shape == (2, 7, 6)
    assert torch.allclose(split[..., :3] + split[..., 3:], inputs, atol=1e-6)
    assert TemporalProjection(7, 6, 11)(split).shape == (2, 11, 6)


@pytest.mark.parametrize('kind', ['dlinear', 'patchtst', 'itransformer'])
def test_legacy_model_names_can_feed_hybrid_layers(kind):
    architecture = spec(kind)
    architecture['input']['feature_order'] = [0, 1, 2, 3]
    architecture['layers'] = [
        {'id': 'old', 'type': kind, 'params': {}},
        {'id': 'gru', 'type': 'gru', 'params': {'hidden_size': 8}},
        {'id': 'last', 'type': 'last_state', 'params': {}},
    ]
    assert build_architecture(architecture, 10, 3)(torch.randn(2, 10, 4)).shape == (2, 3)


@pytest.mark.parametrize('kind', ['dlinear', 'patchtst', 'itransformer'])
def test_legacy_single_model_checkpoint_layout_is_preserved(kind):
    from hypercast4d.research_models import AttentionForecast
    architecture = spec(kind)
    architecture['input']['feature_order'] = [0, 1, 2, 3]
    architecture['layers'] = [{'id': 'forecaster', 'type': kind, 'params': {}}]
    model = build_architecture(architecture, 10, 3).eval()
    old = DLinear(10, 3, 25) if kind == 'dlinear' else AttentionForecast(
        kind, 10, 3, d_model=32, n_heads=4, num_layers=2, dropout=0.1)
    old.load_state_dict(model.layers[0].state_dict())
    old.eval()
    inputs = torch.randn(2, 10, 4)
    assert torch.equal(model(inputs), old(inputs))


def test_collection_has_complete_sources_and_resolvable_presets():
    collection = method_collection()
    assert sum(map(len, SURVEY_FAMILIES.values())) == 44
    methods = collection['methods']
    assert len({item['id'] for item in methods}) == len(methods) == 52
    assert len([item for item in methods if item['status'] == 'adaptation']) == 15
    preset_ids = {item['preset_id'] for item in presets()}
    for method in methods:
        assert method['sources']
        assert method['preset_id'] is None or method['preset_id'] in preset_ids
        for source in method['sources']:
            assert source['source_id'] in collection['sources']
    deformtime = next(item for item in methods if item['name'] == 'DeformTime')
    assert deformtime['status'] == 'reference'
    assert deformtime['preset_id'] is None
