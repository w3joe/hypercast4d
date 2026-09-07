import copy

import pytest
import torch

from hypercast4d.architecture import presets
from hypercast4d.internal_graph import architecture_internal_graph
from hypercast4d.upstream_models import UPSTREAM_MODELS


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def spec(method):
    return next(p for p in presets() if p['preset_id'] == f'tslib-{method}')


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
def test_each_model_has_observed_acyclic_graph_and_valid_editor_links(method):
    architecture = spec(method)
    before = copy.deepcopy(architecture)
    rng = torch.get_rng_state().clone()
    graph = architecture_internal_graph(architecture, 'core')
    assert torch.equal(rng, torch.get_rng_state())
    assert architecture == before
    assert graph['mode'] == 'observed'
    nodes = {n['id']: n for n in graph['nodes']}
    assert any(n['kind'] == 'input' for n in nodes.values())
    assert graph['nodes'][-1]['kind'] == 'output'
    assert graph['nodes'][-1]['shapes'] == [[1, 10, 4]]
    order = {n['id']: i for i, n in enumerate(graph['nodes'])}
    assert all(order[e['source']] < order[e['target']] for e in graph['edges'])
    targets = {t['path'] for t in graph['targets'] if not t['blocked_by']}
    assert all(n['target'] in targets for n in nodes.values() if n['target'])
    # Every shown node contributes to the output, not merely registered in the module tree.
    downstream = {graph['nodes'][-1]['id']}
    for edge in reversed(graph['edges']):
        if edge['target'] in downstream:
            downstream.add(edge['source'])
    assert downstream == set(nodes)


def test_tsmixer_graph_preserves_both_residual_branches():
    graph = architecture_internal_graph(spec('tsmixer'), 'core')
    nodes = {n['id']: n for n in graph['nodes']}
    joins = [n for n in graph['nodes'] if n['label'] == 'Add']
    assert len(joins) == 2
    incoming = lambda node: [nodes[e['source']] for e in graph['edges'] if e['target'] == node['id']]
    assert {n['path'] for n in incoming(joins[0])} == {None, 'model.0.temporal.3'}
    assert {n['path'] for n in incoming(joins[1])} == {None, 'model.0.channel.3'}
    assert joins[0]['id'] in [n['id'] for n in incoming(joins[1])]


def test_edited_stack_graph_reports_actual_hidden_shapes_and_parent_editor():
    architecture = spec('tsmixer')
    architecture['layers'][0]['internal_overrides'] = {'model.0.temporal': {
        'hidden_units': [64, 16], 'activation': 'gelu', 'dropout': .1, 'bias': True}}
    graph = architecture_internal_graph(architecture, 'core')
    nodes = [n for n in graph['nodes'] if n['target'] == 'model.0.temporal' and n['label'] == 'Linear']
    assert [n['shapes'][0][-1] for n in nodes] == [64, 16, 32]
    assert len([n for n in graph['nodes'] if n['label'] == 'Add']) == 2


def test_graph_uses_actual_preceding_block_shape():
    architecture = spec('tsmixer')
    architecture['layers'].insert(0, {'id': 'project', 'type': 'dense', 'params': {'units': 8}})
    graph = architecture_internal_graph(architecture, 'core', 33, 5)
    assert graph['nodes'][0]['shapes'] == [[1, 33, 8]]
    assert graph['nodes'][-1]['shapes'] == [[1, 33, 8]]


def test_graph_api_and_invalid_target(tmp_path):
    from fastapi.testclient import TestClient
    from hypercast4d.playground import create_app
    with TestClient(create_app(tmp_path / 'playground', tmp_path)) as client:
        response = client.post('/api/v1/architectures/internal-graph', json={
            'architecture': spec('tsmixer'), 'layer_id': 'core'})
        assert response.status_code == 200 and response.json()['nodes']
        for layer_id in ('missing', 'flatten'):
            response = client.post('/api/v1/architectures/internal-graph', json={
                'architecture': spec('tsmixer'), 'layer_id': layer_id})
            assert response.status_code == 422
