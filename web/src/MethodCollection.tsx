import { useState } from 'react'
import type { Catalog } from './types'

export default function MethodCollection({ catalog, onLoad }: {
  catalog: Catalog
  onLoad: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState('all')
  const [source, setSource] = useState('all')
  const [available, setAvailable] = useState(false)
  const collection = catalog.method_collection
  if (!collection) return null
  const methods = collection.methods.filter((method) =>
    `${method.name} ${method.family}`.toLowerCase().includes(query.toLowerCase()) &&
    (family === 'all' || method.family === family) &&
    (source === 'all' || method.sources.some((item) => item.source_id === source)) &&
    (!available || method.preset_id !== null))
  return <details className="method-collection panel-surface">
    <summary>Method collection <span>{collection.methods.length} methods from 3 papers</span></summary>
    <p>Explore the cited methods. Runnable adaptations use this playground’s target and evaluation protocol; they are not published benchmark reproductions.</p>
    <div className="method-filters">
      <label>Search methods<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name or architecture family" /></label>
      <label>Architecture family<select value={family} onChange={(event) => setFamily(event.target.value)}><option value="all">All families</option>{[...new Set(collection.methods.map((method) => method.family))].sort().map((item) => <option key={item}>{item}</option>)}</select></label>
      <label>Source paper<select value={source} onChange={(event) => setSource(event.target.value)}><option value="all">All papers</option>{Object.entries(collection.sources).map(([id, item]) => <option value={id} key={id}>{item.title}</option>)}</select></label>
      <label className="checkbox-row"><input type="checkbox" checked={available} onChange={(event) => setAvailable(event.target.checked)} />Runnable only</label>
    </div>
    <p>{methods.length} matching methods</p>
    <div className="method-grid">{methods.map((method) => <article className="method-card" key={method.id}>
      <div><h3>{method.name}</h3><span>{method.family} · {method.kind === 'non-neural baseline' ? method.kind : method.status === 'adaptation' ? 'Runnable adaptation' : 'Reference only'}</span></div>
      <p>{method.notes}</p>
      <div className="method-sources">{method.sources.map((reference) => {
        const paper = collection.sources[reference.source_id]
        return <a key={reference.source_id} href={`${paper.url}#page=${reference.page}`} target="_blank" rel="noreferrer" title={paper.title}>{paper.venue} · p. {reference.page}</a>
      })}</div>
      {method.preset_id && <button className="secondary-button" onClick={() => onLoad(method.preset_id!)}>Load {method.name}</button>}
    </article>)}</div>
    {!methods.length && <p>No methods match these filters.</p>}
  </details>
}
