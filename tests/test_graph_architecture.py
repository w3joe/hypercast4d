import copy
import io
import json

import pytest
import torch

from hypercast4d.architecture import architecture_hash, build_architecture, presets, validate_architecture
from hypercast4d.graph_architecture import convert_to_graph
from hypercast4d.upstream_models import UPSTREAM_MODELS


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def spec(method):
    return next(p for p in presets() if p['preset_id'] == f'tslib-{method}')


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
@pytest.mark.parametrize('window,horizon,width', [(2, 1, 1), (10, 3, 4), (33, 7, 3)])
def test_native_graph_matches_source_outputs_gradients_and_optimizer(method, window, horizon, width):
    v1 = spec(method)
    v1['input']['feature_order'] = list(range(width))
    v2 = json.loads(json.dumps(convert_to_graph(v1, 10, 1)))
    torch.manual_seed(7)
    source = build_architecture(v1, window, horizon).eval()
    torch.manual_seed(7)
    graph = build_architecture(v2, window, horizon).eval()
    x = torch.randn(2, window, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = source(x), graph(y)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-6)
    for name, p in source.named_parameters():
        if p.grad is not None:
            q = graph._param_origins['s0', name]
            torch.testing.assert_close(q.grad, p.grad, rtol=1e-5, atol=1e-6)
    torch.optim.Adam(source.parameters(), lr=.001).step()
    torch.optim.Adam(graph.parameters(), lr=.001).step()
    torch.testing.assert_close(graph(y), source(x), rtol=1e-4, atol=1e-5)
    for value in (0., 1.):
        assert torch.isfinite(graph(torch.full_like(y, value))).all()


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_train_eval_dropout_and_checkpoint_roundtrip(method):
    v1 = spec(method)
    v2 = convert_to_graph(v1)
    torch.manual_seed(11)
    source = build_architecture(v1, 10, 3)
    torch.manual_seed(11)
    graph = build_architecture(v2, 10, 3)
    x = torch.randn(2, 10, 4)
    torch.manual_seed(15)
    expected = source(x)
    torch.manual_seed(15)
    actual = graph(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    checkpoint = io.BytesIO()
    torch.save(graph.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = build_architecture(v2, 10, 3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(graph.eval()(x), restored.eval()(x), rtol=0, atol=0)


def test_dense_resize_insert_activation_remove_residual_and_train():
    graph = convert_to_graph(spec('tsmixer'))
    info = validate_architecture(graph, 10, 3)['graph_nodes']
    dense = next(n for n in graph['nodes'] if info.get(n['id'], {}).get('source_path') == 'layers.0.model.model.0.temporal.0')
    dense['params'] = {'out_features': 64}
    model = build_architecture(graph, 10, 3)
    assert model.blocks[model.bindings[dense['id']]].out_features == 64
    # The downstream Linear infers the new 64-wide input without a hidden adapter.
    following = next(n for n in graph['nodes'] if info.get(n['id'], {}).get('source_path') == 'layers.0.model.model.0.temporal.2')
    assert model.blocks[model.bindings[following['id']]].in_features == 64
    edge = next(e for e in graph['edges'] if e['source'] == dense['id'])
    previous = edge['source']
    edge['source'] = 'custom_gelu'
    graph['nodes'].append({'id': 'custom_gelu', 'kind': 'activation', 'params': {'kind': 'gelu'}})
    graph['edges'].append({'source': previous, 'target': 'custom_gelu', 'port': 'x'})
    compiled = build_architecture(graph, 10, 3)
    joins = [n for n in graph['nodes'] if info.get(n['id'], {}).get('label') == 'add']
    join = joins[0]['id']
    replacement = next(e['source'] for e in graph['edges'] if e['target'] == join and e['port'] == 'args/1')
    graph['edges'] = [{**e, 'source': replacement if e['source'] == join else e['source']} for e in graph['edges'] if e['target'] != join]
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] != join]
    edited = build_architecture(graph, 10, 3)
    x = torch.randn(2, 10, 4)
    assert not torch.equal(compiled.eval()(x), edited.eval()(x))
    edited(x).square().mean().backward()


def test_hash_excludes_presentation_and_keeps_v1_unchanged():
    v1 = spec('tsmixer')
    original = copy.deepcopy(v1)
    before = architecture_hash(v1)
    graph = convert_to_graph(v1)
    assert v1 == original and architecture_hash(v1) == before
    fingerprint = architecture_hash(graph)
    graph['name'] = 'Renamed'
    graph['groups'] = []
    for node in graph['nodes']:
        node['group'] = 'display-only'
        node['label'] = 'New title'
    assert architecture_hash(graph) == fingerprint
    assert fingerprint != before


def test_invalid_edges_cycles_and_untrusted_bindings_rejected():
    graph = convert_to_graph(spec('tsmixer'))
    broken = copy.deepcopy(graph)
    broken['edges'].append(copy.deepcopy(broken['edges'][0]))
    with pytest.raises(ValueError, match='multiple connections'):
        build_architecture(broken, 10, 1)
    broken = copy.deepcopy(graph)
    broken['edges'][0]['source'] = broken['edges'][0]['target']
    with pytest.raises(ValueError, match='cycle'):
        build_architecture(broken, 10, 1)
    broken = copy.deepcopy(graph)
    broken['nodes'][-1]['source_ref']['node'] = '__import__'
    with pytest.raises(ValueError, match='unavailable'):
        build_architecture(broken, 10, 1)


@pytest.mark.parametrize('preset', presets(), ids=lambda p: p['preset_id'])
def test_every_legacy_preset_converts_without_changing_initial_output(preset):
    converted = convert_to_graph(preset)
    torch.manual_seed(19)
    source = build_architecture(preset, 10, 3).eval()
    torch.manual_seed(19)
    graph = build_architecture(converted, 10, 3).eval()
    x = torch.randn(3, 10, 4)
    torch.testing.assert_close(graph(x), source(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_nondefault_core_settings_lower_without_sample_control_flow(method):
    original = spec(method)
    params = original['layers'][0]['params']
    for key, value in {'d_model': 64, 'num_layers': 2, 'n_heads': 8, 'dropout': .2}.items():
        if key in params:
            params[key] = value
    graph = convert_to_graph(original)
    torch.manual_seed(29)
    reference = build_architecture(original, 33, 3).eval()
    torch.manual_seed(29)
    compiled = build_architecture(graph, 33, 3).eval()
    x = torch.randn(2, 33, 4)
    torch.testing.assert_close(compiled(x), reference(x), rtol=1e-5, atol=1e-6)


def test_legacy_internal_override_is_preserved_in_conversion():
    original = spec('tsmixer')
    original['layers'][0]['internal_overrides'] = {'model.0.temporal': {
        'hidden_units': [16, 24], 'activation': 'gelu', 'dropout': .2, 'bias': True}}
    converted = convert_to_graph(original)
    torch.manual_seed(31)
    reference = build_architecture(original, 10, 3).eval()
    torch.manual_seed(31)
    compiled = build_architecture(converted, 10, 3).eval()
    x = torch.randn(2, 10, 4)
    torch.testing.assert_close(compiled(x), reference(x))
    assert any(v['settings'].get('out_features') == 24 for v in compiled.metadata.values())


def test_custom_branches_explicit_sharing_and_independent_duplication():
    graph = convert_to_graph(spec('tsmixer'))
    previous = graph['output']
    graph['nodes'] += [
        {'id': 'left', 'kind': 'dense', 'params': {'units': 1}, 'module_ref': 'shared'},
        {'id': 'right', 'kind': 'dense', 'params': {'units': 1}, 'module_ref': 'shared'},
        {'id': 'join', 'kind': 'add', 'params': {}},
    ]
    graph['edges'] += [
        {'source': previous, 'target': 'left', 'port': 'x'},
        {'source': previous, 'target': 'right', 'port': 'x'},
        {'source': 'left', 'target': 'join', 'port': 'a'},
        {'source': 'right', 'target': 'join', 'port': 'b'},
    ]
    graph['output'] = 'join'
    model = build_architecture(graph, 10, 1)
    assert model.bindings['left'] == model.bindings['right']
    model(torch.randn(3, 10, 4)).square().mean().backward()
    assert model.blocks[model.bindings['left']].weight.grad is not None
    graph['nodes'][-2]['module_ref'] = 'independent'
    model = build_architecture(graph, 10, 1)
    assert model.bindings['left'] != model.bindings['right']
    graph['nodes'][-2]['module_ref'] = 'shared'
    graph['nodes'][-2]['params']['units'] = 2
    with pytest.raises(ValueError, match='identical'):
        build_architecture(graph, 10, 1)


def test_runtime_period_selection_changes_with_input_frequency():
    from hypercast4d.graph_lowering import period_select, period_fold, period_unfold
    time = torch.arange(64).float()
    low = torch.sin(2 * torch.pi * time / 32)[None, :, None].repeat(2, 1, 4)
    high = torch.sin(2 * torch.pi * time / 8)[None, :, None].repeat(2, 1, 4)
    low_period = period_select(low, 1)[0][0]
    high_period = period_select(high, 1)[0][0]
    assert low_period == 32 and high_period == 8
    torch.testing.assert_close(period_unfold(period_fold(high, high_period), high), high)


def test_raw_frequency_weights_duplicate_independently_and_receive_gradients():
    graph = convert_to_graph(spec('frets'))
    from hypercast4d.graph_architecture import describe_graph
    contracts = describe_graph(graph)['graph_nodes']
    weight = next(n for n in graph['nodes'] if contracts[n['id']].get('source_path') == 'layers.0.model.r1')
    duplicate = {**copy.deepcopy(weight), 'id': 'independent-frequency-weight', 'module_ref': 'independent-weight'}
    graph['nodes'] += [duplicate, {'id': 'weight-sum', 'kind': 'add', 'params': {}}]
    graph['edges'] = [{**e, 'source': 'weight-sum' if e['source'] == weight['id'] else e['source']} for e in graph['edges']]
    graph['edges'] += [{'source': weight['id'], 'target': 'weight-sum', 'port': 'a'}, {'source': duplicate['id'], 'target': 'weight-sum', 'port': 'b'}]
    model = build_architecture(graph, 10, 1)
    first = model.constants[model.bindings['state:' + weight['module_ref']]]
    second = model.constants[model.bindings['state:independent-weight']]
    assert first is not second and first.data_ptr() != second.data_ptr()
    model(torch.randn(2, 10, 4)).square().mean().backward()
    assert first.grad is not None and second.grad is not None


def test_draft_port_contracts_survive_missing_connections():
    from hypercast4d.graph_architecture import describe_graph
    graph = convert_to_graph(spec('tsmixer'))
    expected = describe_graph(graph)
    graph['edges'] = []
    assert describe_graph(graph) == expected
    with pytest.raises(ValueError, match='ports'):
        build_architecture(graph, 10, 1)


def test_two_model_graph_executes_both_branches_and_backpropagates():
    graph = convert_to_graph(spec('tsmixer'))
    second = convert_to_graph(spec('dlinear'))
    graph['sources']['s1'] = second['sources']['s0']
    for node in second['nodes']:
        node['id'] = 'second_' + node['id']
        node['source_ref']['source'] = 's1'
        if 'module_ref' in node:
            node['module_ref'] = 'second_' + node['module_ref']
    graph['nodes'] += second['nodes'] + [{'id': 'hybrid', 'kind': 'add', 'params': {}}]
    graph['edges'] += [{**edge, 'source': 'second_' + edge['source'], 'target': 'second_' + edge['target']} for edge in second['edges']]
    graph['edges'] += [{'source': graph['output'], 'target': 'hybrid', 'port': 'a'}, {'source': 'second_' + second['output'], 'target': 'hybrid', 'port': 'b'}]
    graph['output'] = 'hybrid'
    model = build_architecture(graph, 10, 3)
    model(torch.randn(3, 10, 4)).square().mean().backward()
    for source in ('s0', 's1'):
        assert any(parameter.grad is not None for (key, _), parameter in model._param_origins.items() if key == source)


@pytest.mark.parametrize('kind,params', [('reshape', {'shape': [0, -1]}), ('permute', {'dims': [0, 1]}), ('softmax', {'dim': -1})])
def test_explicit_shape_and_attention_operations(kind, params):
    graph = convert_to_graph(spec('tsmixer'))
    graph['nodes'].append({'id': 'operation', 'kind': kind, 'params': params})
    graph['edges'].append({'source': graph['output'], 'target': 'operation', 'port': 'x'})
    graph['output'] = 'operation'
    model = build_architecture(graph, 10, 3)
    x = torch.randn(3, 10, 4, requires_grad=True)
    result = model(x)
    assert result.shape == (3, 3)
    result.square().mean().backward()
    assert x.grad is not None


@pytest.mark.parametrize('algebra', ['quaternion', 'coquaternion', 'cl11'])
@pytest.mark.parametrize('rank', [2, 3, 4])
def test_hyperdense_replacement_supports_feature_tensors_and_checkpoint(algebra, rank):
    graph = convert_to_graph(spec('tsmixer'))
    previous = graph['output']
    graph['nodes'] += [
        {'id': 'shape-in', 'kind': 'reshape', 'params': {'shape': [0] + [1] * (rank - 2) + [4]}},
        {'id': 'hyper', 'kind': 'hyper_dense', 'params': {'units': 1, 'algebra': algebra, 'bias': False}},
        {'id': 'shape-out', 'kind': 'reshape', 'params': {'shape': [0, 4]}},
    ]
    graph['edges'] += [{'source': previous, 'target': 'shape-in', 'port': 'x'}, {'source': 'shape-in', 'target': 'hyper', 'port': 'x'}, {'source': 'hyper', 'target': 'shape-out', 'port': 'x'}]
    graph['output'] = 'shape-out'
    model = build_architecture(graph, 10, 4)
    layer = model.blocks[model.bindings['hyper']]
    assert layer.bias is None
    x = torch.randn(3, 10, 4, requires_grad=True)
    model(x).square().mean().backward()
    assert layer.weight.grad is not None and x.grad is not None
    restored = build_architecture(graph, 10, 4)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.eval()(x), model.eval()(x))


def test_replace_internal_tsmixer_dense_with_hyperdense_preserves_wiring_and_trains():
    graph = convert_to_graph(spec('tsmixer'))
    metadata = validate_architecture(graph, 10, 3)['graph_nodes']
    node = next(n for n in graph['nodes'] if metadata.get(n['id'], {}).get('source_path') == 'layers.0.model.model.0.temporal.0')
    width = metadata[node['id']]['settings']['out_features']
    key = node['id']
    node.update(kind='hyper_dense', params={'units': width // 4, 'algebra': 'quaternion', 'bias': False})
    node.pop('source_ref'); node.pop('module_ref')
    for edge in graph['edges']:
        if edge['target'] == key:
            edge['port'] = 'x'
    model = build_architecture(graph, 10, 3)
    assert model.metadata[key]['shape'] == metadata[key]['shape']
    layer = model.blocks[model.bindings[key]]
    assert type(layer).__name__ == 'HyperDense'
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = layer.weight.detach().clone()
    model(torch.randn(3, 10, 4)).square().mean().backward(); optimizer.step()
    assert not torch.equal(before, layer.weight)
