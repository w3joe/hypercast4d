import { useQuery } from '@tanstack/react-query'
import { useId, useMemo } from 'react'
import { api } from './api'
import type { ArchitectureSpec, InternalGraph, LayerSpec } from './types'

export function GraphView({ graph, selectedPath, onSelect }: {
  graph: InternalGraph
  selectedPath: string | null
  onSelect: (path: string) => void
}) {
  const marker = useId().replaceAll(':', '')
  const layout = useMemo(() => {
    const ranks = new Map<string, number>()
    const rows = new Map<number, string[]>()
    for (const node of graph.nodes) {
      const parents = graph.edges.filter(edge => edge.target === node.id).map(edge => ranks.get(edge.source) ?? 0)
      const rank = parents.length ? Math.max(...parents) + 1 : 0
      ranks.set(node.id, rank)
      rows.set(rank, [...(rows.get(rank) ?? []), node.id])
    }
    const width = Math.max(480, ...[...rows.values()].map(row => row.length * 230 + 30))
    const positions = new Map<string, { x: number; y: number }>()
    for (const [rank, row] of rows) row.forEach((id, index) => positions.set(id, {
      x: (width - row.length * 230) / 2 + index * 230 + 10, y: rank * 140 + 25,
    }))
    return { width, height: rows.size * 140 + 25, positions }
  }, [graph])
  return <>
    <div className="internal-graph-viewport" role="region" aria-label="Internal model graph" tabIndex={0}>
      <div className="internal-graph-surface" style={{ width: layout.width, height: layout.height }}>
        <svg className="internal-graph-edges" width={layout.width} height={layout.height} aria-hidden="true">
          <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#8c9e91" /></marker></defs>
          {graph.edges.map(edge => {
            const from = layout.positions.get(edge.source)!, to = layout.positions.get(edge.target)!
            const x1 = from.x + 100, y1 = from.y + 94, x2 = to.x + 100, y2 = to.y
            // Offset long residual/skip edges so they do not run through intermediate nodes.
            const skip = y2 - y1 > 100
            const side = Math.max(from.x, to.x) + 215
            return <path key={`${edge.source}-${edge.target}`} markerEnd={`url(#${marker})`} d={skip
              ? `M ${x1} ${y1} L ${side} ${y1 + 12} L ${side} ${y2 - 14} L ${x2} ${y2}`
              : `M ${x1} ${y1} C ${x1} ${y1 + 24}, ${x2} ${y2 - 24}, ${x2} ${y2}`} />
          })}
        </svg>
        {graph.nodes.map(node => {
          const position = layout.positions.get(node.id)!
          const content = <><strong>{node.label}</strong><code title={node.path ?? undefined}>{node.path ?? node.kind}</code><span>{node.shapes.map(shape => `[${shape.join(', ')}]`).join(' · ')}</span>{node.target && <small>Edit in Inspector ↗</small>}</>
          const isSelected = Boolean(selectedPath && (selectedPath === node.target || node.path?.startsWith(selectedPath + '.')))
          const className = `internal-graph-node ${node.kind} ${node.target ? 'editable' : ''} ${isSelected ? 'selected' : ''}`
          return node.target ? <button key={node.id} className={className} style={{ left: position.x, top: position.y }}
            aria-label={`Edit ${node.path ?? node.label}`} aria-pressed={isSelected} onClick={() => onSelect(node.target!)}>{content}</button>
            : <div key={node.id} className={className} style={{ left: position.x, top: position.y }}>{content}</div>
        })}
      </div>
    </div>
    <div className="internal-stack-select"><label>Edit a containing dense stack<select value={graph.targets.some(t => t.kind === 'dense_stack' && t.path === selectedPath) ? selectedPath! : ''}
      onChange={event => event.target.value && onSelect(event.target.value)}><option value="">Select an MLP group…</option>
      {graph.targets.filter(t => t.kind === 'dense_stack' && !t.blocked_by).map(t => <option value={t.path} key={t.path}>{t.path}{t.override ? ' · custom' : ''}</option>)}
    </select></label></div>
  </>
}

export default function InternalCanvas({ architecture, layer, window, horizon, selectedPath, onSelect }: {
  architecture: ArchitectureSpec
  layer: LayerSpec
  window: number
  horizon: number
  selectedPath: string | null
  onSelect: (path: string) => void
}) {
  const query = useQuery({
    queryKey: ['internal-graph', architecture, layer.id, window, horizon],
    queryFn: () => api.internalGraph(architecture, layer.id, window, horizon),
    staleTime: 60000, gcTime: 60000, retry: false,
  })
  return <section className="internal-canvas" aria-label={`${layer.type} expanded internals`}>
    <header><strong>Inside {layer.type.replace('tslib_', '')}</strong><span>Observed execution · w{window} / h{horizon}</span></header>
    {query.isPending && <p role="status">Tracing the model’s internal connections…</p>}
    {query.error && <div role="alert"><p>Unable to draw this configuration: {query.error.message}</p><button onClick={() => query.refetch()}>Retry graph</button><p>The model block and its reset controls remain available.</p></div>}
    {query.data && <><p className="internal-graph-notice">{query.data.notice}</p>
      <p className="internal-graph-legend">Green nodes: editable dense modules · Other nodes: read-only operations · Arrows: observed tensor dependencies. Scroll to explore.</p>
      <GraphView graph={query.data} selectedPath={selectedPath} onSelect={onSelect} />
    </>}
  </section>
}
