import { useState } from 'react'
import { ArrowUpRight, Layers3, Search, BookOpen } from 'lucide-react'
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
  const orderedMethods = [...methods].sort((a, b) => Number(Boolean(b.preset_id)) - Number(Boolean(a.preset_id)))
  return <section className="method-collection" aria-label="Method collection">
    <div className="library-intro"><BookOpen size={20} /><div><h2>Method collection</h2><p>{collection.methods.length} methods · {Object.keys(collection.sources).length} research papers · {collection.methods.filter((method) => method.preset_id).length} ready to try</p></div></div>
    <div className="method-filters">
      <label className="method-search">Search methods<div><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a model or architecture…" /></div></label>
      <label>Architecture family<select value={family} onChange={(event) => setFamily(event.target.value)}><option value="all">All families</option>{[...new Set(collection.methods.map((method) => method.family))].sort().map((item) => <option key={item}>{item}</option>)}</select></label>
      <label>Source paper<select value={source} onChange={(event) => setSource(event.target.value)}><option value="all">All papers</option>{Object.entries(collection.sources).map(([id, item]) => <option value={id} key={id}>{item.title}</option>)}</select></label>
      <label className="checkbox-row"><input type="checkbox" checked={available} onChange={(event) => setAvailable(event.target.checked)} />Runnable only</label>
    </div>
    <div className="library-results"><p>{methods.length} matching methods</p><span>Runnable methods first</span></div>
    <div className="method-grid">{orderedMethods.map((method) => <article className={`method-card ${method.preset_id ? 'method-ready' : ''}`} key={method.id}>
      <div className="method-card-top"><span className="method-icon"><Layers3 size={20} /></span><span className={`method-status ${method.preset_id ? 'ready' : ''}`}>{method.kind === 'non-neural baseline' ? 'Non-neural baseline' : method.preset_id ? 'Ready to try' : 'Paper reference'}</span></div>
      <div><h3>{method.name}</h3><span className="method-family">{method.family}</span></div>
      <p>{method.status === 'reference' && method.notes.startsWith('Listed for research') ? 'Explore the original paper for the architecture and evaluation. Training integration is not available yet.' : method.notes}</p>
      <div className="method-sources">{method.sources.map((reference) => {
        const paper = collection.sources[reference.source_id]
        return <a key={reference.source_id} href={`${paper.url}#page=${reference.page}`} target="_blank" rel="noreferrer" title={paper.title}>{paper.venue} · p. {reference.page}<ArrowUpRight size={13} /></a>
      })}</div>
      {method.preset_id && <button className="secondary-button" onClick={() => onLoad(method.preset_id!)}>Load {method.name}<ArrowUpRight size={15} /></button>}
    </article>)}</div>
    {!methods.length && <p>No methods match these filters.</p>}
    <p className="library-footnote">Runnable adaptations use the playground’s evaluation protocol. They are not reproductions of published benchmarks.</p>
  </section>
}
