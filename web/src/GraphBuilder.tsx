import { useCallback, useEffect, useRef, useState } from 'react'
import { ReactFlow, ReactFlowProvider, Handle, Position, Background, Controls, MiniMap, applyNodeChanges, useReactFlow } from '@xyflow/react'
import type { Node, Edge, NodeProps, Connection } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import YAML from 'yaml'
import { api } from './api'
import { EvaluationPanel, RunControls, evaluationFromPreset } from './BuilderControls'
import MethodCollection from './MethodCollection'
import type { ArchitectureSpec, Catalog, GraphSpec, GraphNodeSpec, GraphNodeInfo, GraphRecord, GraphViewState } from './types'
import { appendGraph, connectGraph, duplicateGraph, edgeId, emptyView, parseHandle, projectGraph, safeView, uid } from './graphCanvas'
import type { CanvasData } from './graphCanvas'
import { denseKind, prepareDenseSwap } from './layerSwap'

function Handles({ data }: { data: CanvasData }) {
  return <>{data.ports.map((p, i) => <Handle key={p.id} id={p.id} type="target" position={Position.Top} style={{ left: `${100 * (i + 1) / (data.ports.length + 1)}%` }} title={`Input · ${p.label}`} />)}
    {data.outputs.map((p, i) => <Handle key={p.id} id={p.id} type="source" position={Position.Bottom} style={{ left: `${100 * (i + 1) / (data.outputs.length + 1)}%` }} title={`Output · ${p.label}`} />)}</>
}
function OpNode({ data, selected }: NodeProps<Node<CanvasData>>) {
  return <div className={`graph-operation ${selected ? 'selected' : ''} ${data.output ? 'forecast-output' : ''}`}>
    <small>{data.output ? 'FORECAST' : data.category === 'placeholder' ? 'INPUT DATA' : ['call_module', 'custom'].includes(data.category ?? '') ? 'LAYER' : 'OPERATION'}</small><strong>{data.label}</strong>
    <code>{data.shape ? JSON.stringify(data.shape) : 'Connect to validate'}</code><Handles data={data} />
  </div>
}
function GroupNode({ data, selected }: NodeProps<Node<CanvasData>>) {
  return <div className={`graph-group ${data.collapsed ? 'collapsed' : ''} ${selected ? 'selected' : ''}`}>
    <div className="graph-group-title"><strong>{data.label}</strong><button className="nodrag" onClick={data.onToggle}>{data.collapsed ? 'Expand' : 'Collapse'}</button></div>
    {data.collapsed && <><small>Expand to see and edit the layers inside</small><Handles data={data} /></>}
  </div>
}
const nodeTypes = { graphOp: OpNode, graphGroup: GroupNode }
type Snapshot = { graph: GraphSpec; view: GraphViewState }
const extras = [
  { type: 'add', label: 'Residual / add', defaults: {} }, { type: 'multiply', label: 'Multiply', defaults: {} },
  { type: 'concat', label: 'Concatenate', defaults: { dim: -1 } },
  { type: 'reshape', label: 'Reshape (0 copies axis)', defaults: { shape: [0, -1] } },
  { type: 'permute', label: 'Permute axes', defaults: { dims: [0, 2, 1] } },
  { type: 'softmax', label: 'Softmax', defaults: { dim: -1 } },
]
const allowed = new Set(['dense', 'activation', 'dropout', 'flatten', 'mean_pool', 'last_state', 'layer_norm', 'causal_conv', 'tcn', 'gru', 'lstm', 'hyper_dense'])

function Builder({ catalog }: { catalog: Catalog }) {
  const flow = useReactFlow<Node<CanvasData>>()
  const queryClient = useQueryClient()
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const current = useRef(snapshot); current.current = snapshot
  const past = useRef<Snapshot[]>([]), future = useRef<Snapshot[]>([])
  const [nodes, setNodes] = useState<Node<CanvasData>[]>([]), [edges, setEdges] = useState<Edge[]>([])
  const [selected, setSelected] = useState<string[]>([]), [selectedEdge, setSelectedEdge] = useState<string | null>(null)
  const selection = useRef({ nodes: selected, edge: selectedEdge }); selection.current = { nodes: selected, edge: selectedEdge }
  const [metadata, setMetadata] = useState<Record<string, GraphNodeInfo>>({})
  const [error, setError] = useState(''), [validationError, setValidationError] = useState('')
  const [validatedGraph, setValidatedGraph] = useState<GraphSpec | null>(null)
  const [validating, setValidating] = useState(false), [loading, setLoading] = useState(false)
  const [warnings, setWarnings] = useState<string[]>([]), [message, setMessage] = useState('')
  const [savedId, setSavedId] = useState<string>(), [library, setLibrary] = useState(false)
  const [advanced, setAdvanced] = useState(false), [showLayers, setShowLayers] = useState(false)
  const [swapping, setSwapping] = useState(false), [swapError, setSwapError] = useState('')
  const [layerSearch, setLayerSearch] = useState('')
  const pendingFocus = useRef<string | null>(null)
  useEffect(() => { setSwapError('') }, [selected[0]])
  const [evaluation, setEvaluation] = useState(() => evaluationFromPreset(catalog, 'quick'))
  const evaluationRef = useRef(evaluation); evaluationRef.current = evaluation
  const records = useQuery({ queryKey: ['architectures'], queryFn: api.graphRecords })
  const graph = snapshot?.graph
  const presetGroups = [
    { label: 'Research models', items: catalog.presets.filter(p => p.preset_id?.startsWith('tslib-')) },
    { label: 'Paper baselines', items: catalog.presets.filter(p => p.preset_id?.startsWith('paper-')) },
    { label: 'Other starting points', items: catalog.presets.filter(p => !p.preset_id?.startsWith('tslib-') && !p.preset_id?.startsWith('paper-')) },
  ]
  const presetOptions = presetGroups.filter(g => g.items.length).map(g => <optgroup label={g.label} key={g.label}>{g.items.map(p => <option key={p.preset_id} value={p.preset_id}>{p.name.replace(/ \(TSLib core, editable\)$/, '')}</option>)}</optgroup>)
  const palette = [...catalog.categories.flatMap(c => c.layers).filter(l => allowed.has(l.type)), ...extras]
  const [operation, setOperation] = useState('dense')
  const loadSequence = useRef(0)
  const load = async (spec: ArchitectureSpec | GraphSpec, record?: GraphRecord) => {
    const sequence = ++loadSequence.current
    setLoading(true); setError('')
    try {
      const cell = evaluation.cells[0] ?? { window: 10, horizon: 1 }
      const converted = await api.convert(spec, cell.window, cell.horizon)
      const description = await api.describeGraph(converted, cell.window, cell.horizon)
      if (sequence !== loadSequence.current) return
      const next = { ...converted, locked: Boolean(spec.locked), preset_id: spec.preset_id }
      setSnapshot({ graph: next, view: record?.view ? safeView(record.view, converted) : { positions: {}, collapsed: converted.groups.map(g => g.id) } })
      setSavedId(spec.schema_version === 2 ? record?.id : undefined)
      setMetadata(description.graph_nodes ?? {}); setValidatedGraph(null); setSelected([]); setSelectedEdge(null)
      past.current = []; future.current = []; setLibrary(false)
      setMessage('')
    } catch (e) { setError(String(e)) }
    finally { if (sequence === loadSequence.current) setLoading(false) }
  }
  useEffect(() => { void load(catalog.presets.find(p => p.preset_id === 'tslib-tsmixer') ?? catalog.presets[0]) }, []) // Initial preset only.
  const commit = useCallback((next: Snapshot) => {
    if (current.current) past.current = [...past.current.slice(-49), current.current]
    future.current = []; current.current = next; setSnapshot(next); setError('')
  }, [])
  const edit = (next: GraphSpec) => { if (snapshot && !graph?.locked) commit({ ...snapshot, graph: next }) }
  const toggle = useCallback((id: string) => {
    const s = current.current
    if (s) commit({ ...s, view: { ...s.view, collapsed: s.view.collapsed.includes(id) ? s.view.collapsed.filter(g => g !== id) : [...s.view.collapsed, id] } })
  }, [commit])
  useEffect(() => {
    if (!snapshot) return
    let cancelled = false
    void projectGraph(snapshot.graph, metadata, snapshot.view, toggle).then(result => {
      if (!cancelled) {
        setNodes(result.nodes.map(n => ({ ...n, selected: selection.current.nodes.includes(n.id) }))); setEdges(result.edges.map(e => ({ ...e, selected: selection.current.edge === e.id })))
        if (pendingFocus.current) {
          const id = pendingFocus.current; pendingFocus.current = null
          requestAnimationFrame(() => requestAnimationFrame(() => { if (!cancelled) void flow.fitView({ nodes: [{ id }], padding: .6, maxZoom: 1, duration: 250 }) }))
        }
      }
    }).catch(e => { if (!cancelled) setError(`Layout: ${e}`) })
    return () => { cancelled = true }
  }, [snapshot, metadata, toggle])
  useEffect(() => {
    if (!graph) return
    let cancelled = false
    setValidating(true); setValidatedGraph(null)
    const timer = setTimeout(async () => {
      try {
        if (!evaluation.cells.length) throw new Error('Select at least one evaluation cell.')
        let first: Record<string, GraphNodeInfo> | undefined
        const notices: string[] = []
        for (const cell of evaluation.cells) {
          try {
            const result = await api.validateGraph(graph, cell.window, cell.horizon)
            first ??= result.graph_nodes; notices.push(...result.warnings)
          } catch (e) { throw new Error(`${cell.window}/${cell.horizon}: ${e instanceof Error ? e.message : e}`) }
          if (cancelled) return
        }
        if (!cancelled) { setMetadata(previous => ({ ...previous, ...first })); setValidatedGraph(graph); setValidationError(''); setWarnings([...new Set(notices)]) }
      } catch (e) { if (!cancelled) setValidationError(String(e)) }
      finally { if (!cancelled) setValidating(false) }
    }, 300)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [graph, evaluation.cells])
  const connect = (c: Connection, replacing?: string) => {
    if (!graph || graph.locked) return
    try {
      const [source] = parseHandle(c.sourceHandle), [target, port] = parseHandle(c.targetHandle)
      edit(connectGraph(graph, { source, target, port }, replacing))
    } catch (e) { setError(String(e)) }
  }
  const addModel = async (id: string) => {
    const preset = catalog.presets.find(p => p.preset_id === id)
    if (!preset || !graph || graph.locked) return
    setLoading(true)
    try {
      const cell = evaluation.cells[0] ?? { window: 10, horizon: 1 }
      const template = await api.convert(preset, cell.window, cell.horizon)
      const description = await api.describeGraph(template, cell.window, cell.horizon)
      const s = current.current
      if (!s || s.graph !== graph) throw new Error('The graph changed while loading. Please add the model again.')
      const combined = appendGraph(graph, template, description.graph_nodes)
      const contracts = await api.describeGraph(combined, cell.window, cell.horizon)
      if (current.current?.graph !== graph) throw new Error('The graph changed while loading. Please add the model again.')
      setMetadata(previous => ({ ...previous, ...contracts.graph_nodes }))
      commit({ ...s, graph: combined })
      setMessage('Model added as a disconnected subgraph. Connect its Model input, then join its output or select it as the forecast output.')
    } catch (e) { setError(String(e)) }
    finally { setLoading(false) }
  }
  const remove = () => {
    if (!graph || graph.locked) return
    const ids = new Set(selected.flatMap(id => id.startsWith('group:') ? graph.nodes.filter(n => n.group === id.slice(6)).map(n => n.id) : [id]))
    if (ids.has(graph.output)) { setError('Choose another forecast output before deleting this node.'); return }
    edit({ ...graph, nodes: graph.nodes.filter(n => !ids.has(n.id)), edges: graph.edges.filter(e => !ids.has(e.source) && !ids.has(e.target) && edgeId(e) !== selectedEdge) })
    setSelected([]); setSelectedEdge(null)
  }
  const add = (mode: 'add' | 'insert' | 'replace', position?: { x: number; y: number }, kind = operation) => {
    if (!graph || graph.locked || !snapshot) return
    const choice = palette.find(p => p.type === kind)
    if (!choice) return
    const node: GraphNodeSpec = { id: uid(kind), kind, params: { ...choice.defaults }, label: choice.label }
    let next = { ...graph, nodes: [...graph.nodes, node], edges: [...graph.edges] }
    if (mode === 'insert') {
      const edge = graph.edges.find(e => edgeId(e) === selectedEdge)
      if (!edge || ['add', 'multiply', 'concat'].includes(kind)) { setError('Select an edge and a single-input operation to insert.'); return }
      node.group = graph.nodes.find(n => n.id === edge.target)?.group
      next.edges = [...graph.edges.filter(e => edgeId(e) !== selectedEdge), { ...edge, source: node.id }, { source: edge.source, target: node.id, port: 'x' }]
    } else if (mode === 'replace') {
      const old = graph.nodes.find(n => n.id === selected[0])
      if (!old) return
      const inputs = graph.edges.filter(e => e.target === old.id)
      const ports = ['add', 'multiply', 'concat'].includes(kind) ? ['a', 'b'] : ['x']
      if (inputs.length !== ports.length) { setError(`Replacement requires ${ports.length} inputs. Add and reconnect it explicitly instead.`); return }
      node.group = old.group
      next = { ...next, output: graph.output === old.id ? node.id : graph.output, nodes: next.nodes.filter(n => n.id !== old.id), edges: graph.edges.filter(e => e.target !== old.id).map(e => ({ ...e, source: e.source === old.id ? node.id : e.source })) }
      next.edges.push(...inputs.map((e, i) => ({ ...e, target: node.id, port: ports[i] })))
    }
    commit({ graph: next, view: position ? { ...snapshot.view, positions: { ...snapshot.view.positions, [node.id]: position } } : snapshot.view })
    setSelected([node.id]); setSelectedEdge(null)
  }
  const inspect = graph?.nodes.find(n => n.id === selected[0])
  const info = inspect ? metadata[inspect.id] : undefined
  const settings = inspect ? { ...info?.settings, ...inspect.params } : {}
  const layerKind = inspect ? denseKind(inspect, info) : null
  const switchLayer = async (target: 'dense' | 'hyper_dense') => {
    if (!graph || !inspect || graph.locked || validatedGraph !== graph || swapping) return
    setSwapError(''); setSwapping(true)
    const selectedEvaluation = evaluation
    try {
      const next = prepareDenseSwap(graph, inspect.id, target, metadata)
      let preview: Record<string, GraphNodeInfo> | undefined
      for (const cell of selectedEvaluation.cells) {
        const before = await api.validateGraph(graph, cell.window, cell.horizon)
        const after = await api.validateGraph(next, cell.window, cell.horizon)
        if (JSON.stringify(before.graph_nodes[inspect.id]?.shape) !== JSON.stringify(after.graph_nodes[inspect.id]?.shape)) throw new Error(`The replacement changes this layer’s shape for window ${cell.window} / horizon ${cell.horizon}. Original layer kept.`)
        preview ??= after.graph_nodes
      }
      if (current.current?.graph !== graph || evaluationRef.current !== selectedEvaluation) throw new Error('The experiment changed while checking. Please try the switch again.')
      commit({ ...current.current, graph: next })
      setMetadata(previous => ({ ...previous, ...preview }))
      setMessage(`${target === 'hyper_dense' ? 'HyperDense' : 'Dense'} applied. Connections and output shape preserved; replacement weights are freshly initialized.`)
    } catch (e) { setSwapError(e instanceof Error ? e.message : String(e)) }
    finally { setSwapping(false) }
  }
  const editableLayers = graph?.nodes.filter(n => metadata[n.id]?.category === 'call_module' || n.kind !== 'source') ?? []
  const updateParams = (key: string, value: unknown) => {
    if (!graph || !inspect) return
    edit({ ...graph, nodes: graph.nodes.map(n => n.id === inspect.id || (inspect.module_ref && n.module_ref === inspect.module_ref) ? { ...n, params: { ...n.params, [key]: value } } : n) })
  }
  const selectedNodeIds = graph?.nodes.filter(n => selected.includes(n.id) || (n.group && selected.includes(`group:${n.group}`))).map(n => n.id) ?? []
  const save = async () => {
    if (!graph || !snapshot) return
    try { const record = await api.saveGraph(graph, snapshot.view, savedId); setSavedId(record.id); setMessage('Graph and canvas layout saved.'); void queryClient.invalidateQueries({ queryKey: ['architectures'] }) }
    catch (e) { setError(String(e)) }
  }
  const exportGraph = () => {
    if (!snapshot) return
    const url = URL.createObjectURL(new Blob([YAML.stringify({ ...snapshot.graph, view: snapshot.view })], { type: 'text/yaml' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'architecture-graph.yaml'; anchor.click(); URL.revokeObjectURL(url)
  }
  return <section className="graph-workspace" aria-label="Architecture workspace">
    <div className="graph-topbar"><div><h2>Build your experiment</h2><p>Choose a model → edit its layers → run and compare.</p></div>
      <div className="builder-header-actions"><span className={`builder-status ${validatedGraph === graph && graph ? 'ready' : ''}`}>{loading || validating ? 'Checking…' : validatedGraph === graph && graph ? 'Ready to experiment' : 'Draft'}</span><button onClick={() => setLibrary(!library)}>{library ? 'Architecture canvas' : 'Method collection'}</button></div></div>
    {library ? <MethodCollection catalog={catalog} onLoad={id => { const p = catalog.presets.find(p => p.preset_id === id); if (p) void load(p) }} /> : <>
      <div className="graph-toolbar">
        <label className="builder-model-picker">1. Choose a model<select aria-label="Load graph preset" title={graph?.name} disabled={loading} value={graph?.preset_id ?? ''} onChange={e => { const p = catalog.presets.find(p => p.preset_id === e.target.value); if (p) void load(p) }}><option value="" disabled>{graph?.name ?? 'Choose a starting point…'}</option>{presetOptions}</select></label>
        <label className="builder-saved-picker">Saved experiments<select aria-label="Load saved graph" disabled={loading || !records.data?.length} value="" onChange={e => { const r = records.data?.find(r => r.id === e.target.value); if (r) void load(r.spec, r) }}><option value="" disabled>{records.isLoading ? 'Loading experiments…' : records.data?.length ? 'Open a saved experiment…' : 'No saved experiments yet'}</option>{records.data?.map(r => <option key={r.id} value={r.id}>{r.spec.name} · v{r.spec.schema_version}</option>)}</select></label>
        <button onClick={save} disabled={!graph || graph.locked || loading}>Save graph</button>
        <button aria-pressed={advanced} onClick={() => setAdvanced(!advanced)}>Advanced tools</button>
      </div>
      {advanced && <div className="graph-toolbar graph-advanced-tools">
        <label className="builder-model-picker">Add another model<select value="" disabled={!graph || graph.locked || loading} onChange={e => void addModel(e.target.value)}><option value="" disabled>Choose a model to add…</option>{presetOptions}</select></label>
        <button onClick={exportGraph} disabled={!graph}>Export YAML</button>
        <label className="graph-import">Import<input type="file" accept=".json,.yaml,.yml" onChange={async e => { const file = e.target.files?.[0]; if (!file) return; try { const raw = YAML.parse(await file.text()); await load(raw, raw.schema_version === 2 ? { id: '', spec: raw, view: raw.view } : undefined) } catch (err) { setError(String(err)) } e.target.value = '' }} /></label>
      </div>}
      {graph && snapshot && <>
        <div className="graph-toolbar graph-canvas-toolbar"><strong>2. Edit architecture</strong><input aria-label="Graph name" value={graph.name} disabled={graph.locked} onChange={e => edit({ ...graph, name: e.target.value })} />
          {graph.locked ? <button onClick={() => { commit({ ...snapshot, graph: { ...graph, name: `${graph.name} experiment`.slice(0, 80), locked: false, preset_id: undefined } }); setSavedId(undefined) }}>Clone to edit</button> : <span>Editable experiment</span>}
          <button disabled={!past.current.length} onClick={() => { const previous = past.current.pop(); if (previous) { future.current.push(snapshot); setSnapshot(previous) } }}>Undo</button>
          <button disabled={!future.current.length} onClick={() => { const next = future.current.pop(); if (next) { past.current.push(snapshot); setSnapshot(next) } }}>Redo</button>
          <button aria-pressed={showLayers} onClick={() => setShowLayers(!showLayers)}>Add layers</button>
          <button onClick={() => commit({ ...snapshot, view: { ...snapshot.view, positions: {} } })}>Arrange top-down</button>
          {advanced && <button onClick={() => commit({ ...snapshot, view: emptyView() })}>Expand all</button>}
          {advanced && <button onClick={() => commit({ ...snapshot, view: { positions: {}, collapsed: graph.groups.map(g => g.id) } })}>Collapse groups</button>}
          <button onClick={() => void flow.fitView({ duration: 200, maxZoom: 1 })}>Fit graph</button>
        </div>
        <p className="graph-reading-guide">Input at the top, forecast at the bottom. Expand a model to see its layers; click any layer to change its settings.</p>
        <div className={`graph-editor-grid ${showLayers ? 'with-palette' : ''}`}>
          {showLayers && <aside className="graph-palette"><h3>Add a layer</h3><p>Pick a layer below. Select a connection to insert it between two layers.</p>
            <select aria-label="Operation" value={operation} onChange={e => setOperation(e.target.value)}>{palette.map(p => <option key={p.type} value={p.type}>{p.label}</option>)}</select>
            <button disabled={graph.locked} onClick={() => add('add')}>Add to canvas</button>
            <button disabled={graph.locked || !selectedEdge} onClick={() => add('insert')}>Insert on edge</button>
            <button disabled={graph.locked || !inspect} onClick={() => add('replace')}>Replace selected</button>
            <div className="graph-palette-list">{palette.map(p => <div key={p.type} draggable={!graph.locked} onDragStart={e => e.dataTransfer.setData('application/hypercast-op', p.type)}>{p.label}</div>)}</div>
          </aside>}
          <div className="graph-canvas" aria-label="Architecture canvas" tabIndex={0} onKeyDown={e => {
            if ((e.target as HTMLElement).closest('input, select, textarea, button')) return
            if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); remove() }
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
              e.preventDefault()
              const history = e.shiftKey ? future : past, destination = e.shiftKey ? past : future
              const next = history.current.pop(); if (next) { destination.current.push(snapshot); setSnapshot(next) }
            }
          }} onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'move' }} onDrop={e => { e.preventDefault(); const kind = e.dataTransfer.getData('application/hypercast-op'); if (kind) add('add', flow.screenToFlowPosition({ x: e.clientX, y: e.clientY }), kind) }}>
            <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.02} maxZoom={2} nodesDraggable={!graph.locked} nodesConnectable={!graph.locked} edgesReconnectable={!graph.locked} deleteKeyCode={null}
              onNodesChange={changes => setNodes(previous => applyNodeChanges(changes, previous))}
              onNodeDragStop={(_, node, dragged) => { const positions = { ...snapshot.view.positions }; for (const n of dragged.length ? dragged : [node]) positions[n.id] = n.position; commit({ ...snapshot, view: { ...snapshot.view, positions } }) }}
              onSelectionChange={s => { setSelected(s.nodes.map(n => n.id)); setSelectedEdge(s.edges[0]?.id ?? null) }}
              onConnect={c => connect(c)} onReconnect={(edge, c) => connect(c, edge.id)}>
              <Background gap={24} /><Controls />{advanced && <MiniMap pannable zoomable />}</ReactFlow>
          </div>
          <aside className="graph-inspector"><div className="inspector-heading"><small>EDIT YOUR MODEL</small><h3>Layer settings</h3></div>
            <details className="layer-finder" open={!inspect}><summary>Find a layer <span>{editableLayers.length}</span></summary>
              <input aria-label="Search layers" placeholder="Search dense, projection, conv…" value={layerSearch} onChange={e => setLayerSearch(e.target.value)} />
              <div className="layer-finder-results">{editableLayers.filter(n => `${n.label ?? ''} ${metadata[n.id]?.label === 'Linear' ? 'Dense Linear' : metadata[n.id]?.label ?? n.kind} ${metadata[n.id]?.source_path ?? ''}`.toLowerCase().includes(layerSearch.toLowerCase())).map(n => <button key={n.id} className={inspect?.id === n.id ? 'active' : ''} onClick={() => {
                setSelected([n.id]); setSelectedEdge(null); setSwapError('')
                setNodes(previous => previous.map(item => ({ ...item, selected: item.id === n.id })))
                setEdges(previous => previous.map(item => ({ ...item, selected: false })))
                if (n.group && snapshot.view.collapsed.includes(n.group)) { pendingFocus.current = n.id; commit({ ...snapshot, view: { ...snapshot.view, collapsed: snapshot.view.collapsed.filter(g => g !== n.group) } }) }
                else void flow.fitView({ nodes: [{ id: n.id }], padding: .6, maxZoom: 1, duration: 250 })
              }}><strong>{n.label ?? metadata[n.id]?.label ?? n.kind}</strong><small>{metadata[n.id]?.source_path ?? n.id}</small></button>)}</div>
            </details>
            {graph.groups.filter(g => selected.includes(`group:${g.id}`)).map(g => <label key={g.id}>Group name<input value={g.label} disabled={graph.locked} onChange={e => edit({ ...graph, groups: graph.groups.map(item => item.id === g.id ? { ...item, label: e.target.value } : item) })} /></label>)}
            {inspect ? <>
              {layerKind && <div className="layer-type-card">
                <label>Layer type<select aria-label="Layer type" value={layerKind} disabled={graph.locked || swapping || validating || validatedGraph !== graph} onChange={e => void switchLayer(e.target.value as 'dense' | 'hyper_dense')}><option value="dense">Dense · real-valued</option><option value="hyper_dense">HyperDense · hypercomplex</option></select></label>
                <p>{layerKind === 'hyper_dense' ? `${Number(settings.units ?? settings.out_features) * 4} output features = ${settings.units ?? settings.out_features} hypercomplex units × 4 components.` : 'Switch to HyperDense without reconnecting this layer.'}</p>
                <small>Checks every evaluation size. New weights; no hidden padding.</small>
                {inspect.module_ref && graph.nodes.filter(n => n.module_ref === inspect.module_ref).length > 1 && <p>Switching affects only this call and gives it independent weights.</p>}
                {swapping && <p role="status">Checking replacement compatibility…</p>}
                {swapError && <p className="swap-error" role="alert">{swapError}</p>}
              </div>}
              <label>Label<input value={inspect.label ?? info?.label ?? inspect.kind} disabled={graph.locked} onChange={e => edit({ ...graph, nodes: graph.nodes.map(n => n.id === inspect.id ? { ...n, label: e.target.value } : n) })} /></label>
              {advanced && <code>{info?.source_path ?? inspect.kind}</code>}<p>Output shape: {JSON.stringify(info?.shape ?? 'Not validated')}</p>
              {validatedGraph !== graph && <small>Shape information is from the last valid graph.</small>}
              {Object.entries(settings).map(([key, value]) => <label key={`${inspect.id}:${key}`}>{layerKind === 'hyper_dense' && ['units', 'out_features'].includes(key) ? 'Hypercomplex units (4 features each)' : ({ out_features: 'Neurons (output width)', units: 'Neurons', p: 'Dropout rate', bias: 'Use bias', out_channels: 'Output channels', kind: 'Activation', num_layers: 'Number of layers' } as Record<string, string>)[key] ?? key.replaceAll('_', ' ')}
                {key === 'algebra' ? <select aria-label="Algebra" disabled={graph.locked || swapping} value={String(value)} onChange={e => updateParams(key, e.target.value)}><option value="quaternion">Quaternion</option><option value="coquaternion">Coquaternion</option><option value="cl11">Cl(1,1)</option></select> : typeof value === 'boolean' ? <input type="checkbox" disabled={graph.locked || swapping} checked={value} onChange={e => updateParams(key, e.target.checked)} /> : <input aria-label={key} key={`${inspect.id}:${key}:${JSON.stringify(value)}`} disabled={graph.locked || swapping} defaultValue={typeof value === 'object' ? JSON.stringify(value) : String(value)} onBlur={e => { try { const raw = e.target.value; const parsed = typeof value === 'number' ? Number(raw) : typeof value === 'object' ? JSON.parse(raw) : raw; if (JSON.stringify(parsed) !== JSON.stringify(value)) updateParams(key, parsed) } catch (err) { setError(String(err)) } }} />}
              </label>)}
              {!Object.keys(settings).length && <p>This is a structural operation. Reconnect its inputs or replace it with a palette operation.</p>}
              {advanced && info?.ports.map(port => <small key={port} className="graph-port-row">{port} ← {graph.edges.find(e => e.target === inspect.id && e.port === port)?.source ?? 'unconnected'}</small>)}
              {advanced && inspect.module_ref && <p>Shared weights: <code>{inspect.module_ref}</code> · {graph.nodes.filter(n => n.module_ref === inspect.module_ref).length} calls. Settings apply to every shared call.</p>}
              {advanced && <>
              <button disabled={graph.locked} onClick={() => edit({ ...graph, output: inspect.id })}>Use as forecast output</button>
              {inspect.module_ref && <button disabled={graph.locked} onClick={() => edit({ ...graph, nodes: graph.nodes.map(n => n.id === inspect.id ? { ...n, module_ref: uid('weights') } : n) })}>Make weights independent</button>}
              </>}
            </> : <p>Click a layer in the canvas to see its settings here. Start by expanding a model.</p>}
            {(selected.length > 0 || selectedEdge) && <>
            <button disabled={graph.locked || !selectedNodeIds.length} onClick={() => edit(duplicateGraph(graph, selectedNodeIds))}>Duplicate independently</button>
            <button disabled={graph.locked || (!selected.length && !selectedEdge)} onClick={remove}>Delete selected</button>
            </>}
            {advanced && <>
            <button disabled={graph.locked || selectedNodeIds.length < 2} onClick={() => {
              const calls = graph.nodes.filter(n => selectedNodeIds.includes(n.id))
              if (calls.some(n => n.kind === 'source' && metadata[n.id]?.category !== 'call_module') || calls.some(n => n.kind !== calls[0].kind)) { setError('Select matching layer operations to share weights.'); return }
              const ref = uid('shared')
              edit({ ...graph, nodes: graph.nodes.map(n => selectedNodeIds.includes(n.id) ? { ...n, module_ref: ref, params: { ...calls[0].params } } : n) })
            }}>Share selected weights</button>
            <button disabled={graph.locked || !selectedNodeIds.length} onClick={() => { const id = uid('group'); edit({ ...graph, groups: [...graph.groups, { id, label: 'Custom group' }], nodes: graph.nodes.map(n => selectedNodeIds.includes(n.id) ? { ...n, group: id } : n) }) }}>Group selected</button>
            <button disabled={graph.locked || !selectedNodeIds.length} onClick={() => edit({ ...graph, nodes: graph.nodes.map(n => selectedNodeIds.includes(n.id) ? { ...n, group: undefined } : n) })}>Ungroup selected</button>
            </>}
          </aside>
        </div>
        <div className="builder-run-card"><h3 className="graph-run-heading">3. Run experiment</h3>
        <div className="graph-validation" role="status">{validating ? 'Checking your architecture…' : validatedGraph === graph ? 'Ready to run — architecture checks passed.' : 'Not ready yet — fix the connection or shape issue below.'}</div>
        {validationError && <div className="error-notice" role="alert">{validationError}</div>}
        {!!warnings.length && <details><summary>{warnings.length} disconnected operations (not executed)</summary>{warnings.map(w => <p key={w}>{w}</p>)}</details>}
        <EvaluationPanel catalog={catalog} evaluation={evaluation} onChange={setEvaluation} />
        <RunControls architecture={graph} evaluation={evaluation} disabled={loading || swapping || validating || validatedGraph !== graph} onQueued={job => setMessage(`Queued run ${job.id}`)} />
        </div>
      </>}
      {loading && <p role="status">Preparing executable graph…</p>}
    </>}
    {error && <div role="alert" className="error-notice">{error}</div>}{message && <p role="status">{message}</p>}
  </section>
}

export default function GraphBuilder(props: { catalog: Catalog }) {
  return <ReactFlowProvider><Builder {...props} /></ReactFlowProvider>
}
