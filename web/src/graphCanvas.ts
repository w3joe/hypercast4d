import ELK from 'elkjs/lib/elk.bundled.js'
import { MarkerType } from '@xyflow/react'
import type { Edge, Node } from '@xyflow/react'
import type { GraphSpec, GraphNodeSpec, GraphNodeInfo, GraphViewState, GraphEdgeSpec } from './types'

export const uid = (prefix = 'node') => `${prefix}-${crypto.randomUUID().slice(0, 8)}`
export const emptyView = (): GraphViewState => ({ positions: {}, collapsed: [] })
export function safeView(raw: unknown, graph: GraphSpec): GraphViewState {
  const value = raw as Partial<GraphViewState> | undefined
  const known = new Set([...graph.nodes.map(n => n.id), ...graph.groups.map(g => groupId(g.id))])
  return { positions: Object.fromEntries(Object.entries(value?.positions ?? {}).filter(([id, p]) => known.has(id) && p && Number.isFinite(p.x) && Number.isFinite(p.y))),
    collapsed: Array.isArray(value?.collapsed) ? value.collapsed.filter(id => graph.groups.some(g => g.id === id)) : [] }
}
export const edgeId = (e: GraphEdgeSpec) => JSON.stringify([e.source, e.target, e.port])
export const groupId = (id: string) => `group:${id}`
export const portHandle = (node: string, port: string) => JSON.stringify([node, port])
export function parseHandle(handle: string | null | undefined): [string, string] {
  if (!handle) throw new Error('Choose a connection handle.')
  const parsed = JSON.parse(handle)
  if (!Array.isArray(parsed) || parsed.length !== 2) throw new Error('Invalid connection handle.')
  return parsed as [string, string]
}
export function portsFor(node: GraphNodeSpec, graph: GraphSpec, info: Record<string, GraphNodeInfo>): string[] {
  return info[node.id]?.ports ?? (node.kind === 'source'
    ? [...new Set(graph.edges.filter(e => e.target === node.id).map(e => e.port))]
    : ['add', 'multiply', 'concat'].includes(node.kind) ? ['a', 'b'] : ['x'])
}
export function connectGraph(graph: GraphSpec, edge: GraphEdgeSpec, replacing?: string): GraphSpec {
  const edges = graph.edges.filter(e => edgeId(e) !== replacing)
  if (edges.some(e => e.target === edge.target && e.port === edge.port)) throw new Error('This input already has a connection. Reconnect or delete that edge first.')
  if (!graph.nodes.some(n => n.id === edge.source) || !graph.nodes.some(n => n.id === edge.target)) throw new Error('Connection endpoint is missing.')
  const visited = new Set<string>()
  const pending = [edge.target]
  while (pending.length) {
    const id = pending.pop()!
    if (id === edge.source) throw new Error('This connection would create a cycle. Use a recurrent operation for recurrence.')
    if (visited.has(id)) continue
    visited.add(id)
    pending.push(...edges.filter(e => e.source === id).map(e => e.target))
  }
  return { ...graph, edges: [...edges, edge] }
}
export function duplicateGraph(graph: GraphSpec, ids: string[]): GraphSpec {
  const mapping = new Map(ids.map(id => [id, uid()]))
  const refs = new Map<string, string>()
  const nodes = graph.nodes.filter(n => mapping.has(n.id)).map(n => {
    if (n.module_ref && !refs.has(n.module_ref)) refs.set(n.module_ref, uid('weights'))
    return { ...structuredClone(n), id: mapping.get(n.id)!, module_ref: n.module_ref ? refs.get(n.module_ref) : undefined, label: `${n.label ?? n.kind} copy` }
  })
  return { ...graph, nodes: [...graph.nodes, ...nodes], edges: [...graph.edges,
    ...graph.edges.filter(e => mapping.has(e.target)).map(e => ({ ...e, source: mapping.get(e.source) ?? e.source, target: mapping.get(e.target)! }))] }
}
export function appendGraph(graph: GraphSpec, template: GraphSpec, info: Record<string, GraphNodeInfo>): GraphSpec {
  const prefix = uid('model'), group = `${prefix}-group`
  const mapping = Object.fromEntries(template.nodes.map(n => [n.id, `${prefix}:${n.id}`]))
  const sourceMapping = Object.fromEntries(Object.keys(template.sources).map(k => [k, uid('source')]))
  const refs = new Map<string, string>()
  const nodes = template.nodes.map(n => {
    if (info[n.id]?.category === 'placeholder') return { id: mapping[n.id], kind: 'activation', params: { kind: 'linear' }, group, label: 'Model input · connect history here' }
    if (n.module_ref && !refs.has(n.module_ref)) refs.set(n.module_ref, uid('weights'))
    return { ...structuredClone(n), id: mapping[n.id], group,
      source_ref: n.source_ref ? { source: sourceMapping[n.source_ref.source], node: n.source_ref.node } : undefined,
      module_ref: n.module_ref ? refs.get(n.module_ref) : undefined }
  })
  return { ...graph, sources: { ...graph.sources, ...Object.fromEntries(Object.entries(template.sources).map(([k, v]) => [sourceMapping[k], v])) },
    nodes: [...graph.nodes, ...nodes], edges: [...graph.edges, ...template.edges.map(e => ({ ...e, source: mapping[e.source], target: mapping[e.target] }))],
    groups: [...graph.groups, { id: group, label: template.name }] }
}
export type CanvasData = {
  label: string; ports: { id: string; label: string }[]; outputs: { id: string; label: string }[]
  shape?: unknown; category?: string; output?: boolean; group?: string; collapsed?: boolean
  onToggle?: () => void
  [key: string]: unknown
}
const elk = new ELK()
async function arrange(ids: string[], edges: { source: string; target: string }[], sizes: Record<string, { width: number; height: number }>) {
  if (!ids.length) return { positions: {}, width: 300, height: 160 }
  const result = await elk.layout({ id: 'root', width: 0, height: 0, layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': '40', 'elk.layered.spacing.nodeNodeBetweenLayers': '65' },
    children: ids.map(id => ({ id, ...sizes[id] })),
    edges: edges.filter(e => ids.includes(e.source) && ids.includes(e.target) && e.source !== e.target).map((e, i) => ({ id: `e${i}`, sources: [e.source], targets: [e.target] })) })
  return { positions: Object.fromEntries((result.children ?? []).map(n => [n.id, { x: n.x ?? 0, y: n.y ?? 0 }])), width: result.width ?? 300, height: result.height ?? 160 }
}
export async function projectGraph(graph: GraphSpec, info: Record<string, GraphNodeInfo>, view: GraphViewState, toggle: (id: string) => void): Promise<{ nodes: Node<CanvasData>[]; edges: Edge[] }> {
  const groups = graph.groups.filter(g => graph.nodes.some(n => n.group === g.id))
  const groupSet = new Set(groups.map(g => g.id))
  const byId = Object.fromEntries(graph.nodes.map(n => [n.id, n]))
  const collapsed = new Set(view.collapsed)
  const visibleId = (id: string) => byId[id]?.group && collapsed.has(byId[id].group!) ? groupId(byId[id].group!) : id
  const edges: Edge[] = graph.edges.flatMap(e => {
    const source = visibleId(e.source), target = visibleId(e.target)
    if (source === target && source.startsWith('group:')) return []
    return [{ id: edgeId(e), source, target, sourceHandle: portHandle(e.source, 'out'), targetHandle: portHandle(e.target, e.port), type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' } }]
  })
  const sizes: Record<string, { width: number; height: number }> = {}
  for (const n of graph.nodes) sizes[n.id] = { width: Math.max(240, portsFor(n, graph, info).length * 22 + 40), height: 90 }
  const layouts: Record<string, Awaited<ReturnType<typeof arrange>>> = {}
  for (const g of groups) {
    const id = groupId(g.id)
    if (collapsed.has(g.id)) {
      const inputs = new Set(edges.filter(e => e.target === id).map(e => e.targetHandle))
      const outputs = new Set(edges.filter(e => e.source === id).map(e => e.sourceHandle))
      sizes[id] = { width: Math.max(300, Math.max(inputs.size, outputs.size) * 22 + 40), height: 110 }
    } else {
      layouts[g.id] = await arrange(graph.nodes.filter(n => n.group === g.id).map(n => n.id), graph.edges, sizes)
      sizes[id] = { width: layouts[g.id].width + 60, height: layouts[g.id].height + 90 }
      // Keep a manually moved child inside its parent frame.
      for (const n of graph.nodes.filter(n => n.group === g.id)) {
        const p = view.positions[n.id]
        if (p) { sizes[id].width = Math.max(sizes[id].width, p.x + sizes[n.id].width + 30); sizes[id].height = Math.max(sizes[id].height, p.y + sizes[n.id].height + 30) }
      }
    }
  }
  const topId = (id: string) => byId[id]?.group && groupSet.has(byId[id].group!) ? groupId(byId[id].group!) : id
  const topIds = [...groups.map(g => groupId(g.id)), ...graph.nodes.filter(n => !n.group || !groupSet.has(n.group)).map(n => n.id)]
  const top = await arrange(topIds, graph.edges.map(e => ({ source: topId(e.source), target: topId(e.target) })), sizes)
  const nodes: Node<CanvasData>[] = groups.map(g => {
    const id = groupId(g.id), isCollapsed = collapsed.has(g.id)
    const inputs = [...new Set(edges.filter(e => e.target === id).map(e => e.targetHandle!))]
    const outputs = [...new Set(edges.filter(e => e.source === id).map(e => e.sourceHandle!))]
    return { id, type: 'graphGroup', position: view.positions[id] ?? top.positions[id] ?? { x: 0, y: 0 }, style: sizes[id],
      data: { label: g.label, group: g.id, collapsed: isCollapsed, onToggle: () => toggle(g.id), ports: inputs.map(h => ({ id: h, label: parseHandle(h)[1] })), outputs: outputs.map(h => ({ id: h, label: byId[parseHandle(h)[0]]?.label ?? 'out' })) } }
  })
  for (const n of graph.nodes) {
    if (n.group && collapsed.has(n.group)) continue
    const grouped = n.group && groupSet.has(n.group)
    const pos = grouped ? layouts[n.group!]?.positions[n.id] : top.positions[n.id]
    nodes.push({ id: n.id, type: 'graphOp', parentId: grouped ? groupId(n.group!) : undefined,
      position: view.positions[n.id] ?? { x: (pos?.x ?? 0) + (grouped ? 30 : 0), y: (pos?.y ?? 0) + (grouped ? 60 : 0) }, style: sizes[n.id],
      data: { label: n.label ?? info[n.id]?.label ?? n.source_ref?.node ?? n.kind, category: info[n.id]?.category ?? n.kind, shape: info[n.id]?.shape, output: graph.output === n.id,
        ports: portsFor(n, graph, info).map(p => ({ id: portHandle(n.id, p), label: p })), outputs: [{ id: portHandle(n.id, 'out'), label: 'out' }] } })
  }
  return { nodes, edges }
}
