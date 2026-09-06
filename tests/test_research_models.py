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


@pytest.mark.parametrize('mutation,match', [
    ('extra', 'only block'), ('target', 'target feature'),
    ('centered', 'levels'), ('head', 'direct'), ('heads', 'divisible'),
])
def test_research_contract_rejects_invalid_edits(mutation, match):
    architecture = copy.deepcopy(spec('itransformer'))
    if mutation == 'extra':
        architecture['layers'].append({'id': 'flatten', 'type': 'flatten', 'params': {}})
    elif mutation == 'target':
        architecture['input']['feature_order'] = [1, 0, 2, 3]
    elif mutation == 'centered':
        architecture['input']['representation'] = 'centered'
    elif mutation == 'head':
        architecture['head']['zero_initialize'] = True
    else:
        architecture['layers'][0]['params']['n_heads'] = 3
    with pytest.raises(ArchitectureError, match=match):
        build_architecture(architecture, 10, 1)


def test_collection_has_complete_sources_and_resolvable_presets():
    collection = method_collection()
    assert sum(map(len, SURVEY_FAMILIES.values())) == 44
    methods = collection['methods']
    assert len({item['id'] for item in methods}) == len(methods) == 52
    assert len([item for item in methods if item['status'] == 'adaptation']) == 3
    preset_ids = {item['preset_id'] for item in presets()}
    for method in methods:
        assert method['sources']
        assert method['preset_id'] is None or method['preset_id'] in preset_ids
        for source in method['sources']:
            assert source['source_id'] in collection['sources']
    deformtime = next(item for item in methods if item['name'] == 'DeformTime')
    assert deformtime['status'] == 'reference'
    assert deformtime['preset_id'] is None
