import { describe, expect, it } from 'vitest'
import { appendGraph, connectGraph, duplicateGraph, edgeId, emptyView, parseHandle, projectGraph } from './graphCanvas'
import type { GraphSpec } from './types'
const graph: GraphSpec = { schema_version: 2, revision: 'test', name: 'test', sources: {},
  nodes: [{ id: 'input', kind: 'source', params: {} }, { id: 'dense', kind: 'dense', params: { units: 4 }, group: 'core', module_ref: 'shared' }, { id: 'out', kind: 'activation', params: { kind: 'gelu' } }],
  edges: [{ source: 'input', target: 'dense', port: 'x' }, { source: 'dense', target: 'out', port: 'x' }], output: 'out', groups: [{ id: 'core', label: 'Model' }] }
describe('canonical canvas topology', () => {
  it('lays out the pipeline from top to bottom', async () => {
    const projected = await projectGraph(graph, {}, { positions: {}, collapsed: ['core'] }, () => {})
    const input = projected.nodes.find(n => n.id === 'input')!
    const model = projected.nodes.find(n => n.id === 'group:core')!
    const output = projected.nodes.find(n => n.id === 'out')!
    expect(input.position.y).toBeLessThan(model.position.y)
    expect(model.position.y).toBeLessThan(output.position.y)
    const expanded = await projectGraph({ ...graph, nodes: graph.nodes.map(n => ({ ...n, group: 'core' })) }, {}, emptyView(), () => {})
    expect(expanded.nodes.find(n => n.id === 'input')!.position.y).toBeLessThan(expanded.nodes.find(n => n.id === 'dense')!.position.y)
    expect(expanded.nodes.find(n => n.id === 'dense')!.position.y).toBeLessThan(expanded.nodes.find(n => n.id === 'out')!.position.y)
  })
  it('maps collapsed proxy handles back to real nodes without changing graph edges', async () => {
    const before = JSON.stringify(graph)
    const collapsed = await projectGraph(graph, {}, { positions: {}, collapsed: ['core'] }, () => {})
    expect(collapsed.nodes.map(n => n.id)).not.toContain('dense')
    expect(parseHandle(collapsed.edges[0].targetHandle)).toEqual(['dense', 'x'])
    const expanded = await projectGraph(graph, {}, emptyView(), () => {})
    expect(expanded.nodes.find(n => n.id === 'dense')?.parentId).toBe('group:core')
    expect(expanded.edges[0].target).toBe('dense')
    expect(JSON.stringify(graph)).toBe(before)
  })
  it('rejects cycles and duplicate drivers and supports reconnecting an edge', () => {
    expect(() => connectGraph(graph, { source: 'out', target: 'input', port: 'x' })).toThrow(/cycle/)
    expect(() => connectGraph(graph, { source: 'input', target: 'out', port: 'x' })).toThrow(/already/)
    const next = connectGraph(graph, { source: 'input', target: 'out', port: 'x' }, edgeId(graph.edges[1]))
    expect(next.edges[1].source).toBe('input')
    expect(graph.edges[1].source).toBe('dense')
  })
  it('duplicates independent weights and remaps internal connections', () => {
    const next = duplicateGraph(graph, ['dense', 'out'])
    expect(next.nodes).toHaveLength(5)
    const copy = next.nodes[3]
    expect(copy.module_ref).not.toBe('shared')
    expect(next.edges).toContainEqual({ source: copy.id, target: next.nodes[4].id, port: 'x' })
  })
  it('inserts another model as an independently wired subgraph', () => {
    const next = appendGraph(graph, graph, { input: { label: 'Input', category: 'placeholder', ports: [], settings: {}, shape: null } })
    expect(next.nodes).toHaveLength(6)
    expect(next.nodes[3].kind).toBe('activation')
    expect(next.nodes[3].params).toEqual({ kind: 'linear' })
    expect(next.nodes[4].module_ref).not.toBe(graph.nodes[1].module_ref)
    expect(next.output).toBe(graph.output)
    expect(next.groups).toHaveLength(2)
  })
})
