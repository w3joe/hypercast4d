import { useState } from 'react'
import type { InternalOverride, InternalTarget, LayerSpec } from './types'

function TargetEditor({ target, locked, onApply, onReset, selected = false }: {
  target: InternalTarget
  locked: boolean
  onApply: (edit: InternalOverride) => void
  onReset: () => void
  selected?: boolean
}) {
  const initial = target.override
  const [widths, setWidths] = useState((initial?.hidden_units ?? target.hidden_units).join(', '))
  const [activation, setActivation] = useState<InternalOverride['activation']>(initial?.activation ?? 'relu')
  const [dropout, setDropout] = useState(initial?.dropout ?? 0)
  const [bias, setBias] = useState(initial?.bias ?? true)
  const [error, setError] = useState('')
  const apply = () => {
    const units = widths.trim() ? widths.split(',').map(value => Number(value.trim())) : []
    if (units.length > 4 || units.some(n => !Number.isInteger(n) || n < 1 || n > 512)) {
      setError('Enter up to four comma-separated widths, each from 1 to 512.'); return
    }
    if (!Number.isFinite(dropout) || dropout < 0 || dropout > .95) {
      setError('Dropout must be between 0 and 0.95.'); return
    }
    setError('')
    onApply({ hidden_units: units, activation, dropout, bias })
  }
  return <details className="internal-target" open={selected || undefined}>
    <summary><code>{target.path}</code><span>{target.kind === 'dense_stack' ? 'Dense stack' : 'Dense'} · {target.in_features} → {target.out_features}{initial ? ' · custom' : ''}</span></summary>
    <p className="internal-structure">Original: {target.structure.join(' → ')}</p>
    {target.blocked_by ? <p>Reset the overlapping edit at <code>{target.blocked_by}</code> to edit this module.</p> : <>
      <p>Fixed boundary: {target.in_features} inputs → editable hidden layers → {target.out_features} outputs.</p>
      <label>Hidden widths<input aria-label={`Hidden widths for ${target.path}`} value={widths} disabled={locked} placeholder="e.g. 64, 32; empty = one linear" onChange={event => setWidths(event.target.value)} /></label>
      <label>Hidden activation<select value={activation} disabled={locked} onChange={event => setActivation(event.target.value as InternalOverride['activation'])}>{['relu', 'gelu', 'silu', 'tanh', 'linear'].map(value => <option key={value}>{value}</option>)}</select></label>
      <label>Hidden dropout<input type="number" min="0" max="0.95" step="0.05" value={dropout} disabled={locked} onChange={event => setDropout(Number(event.target.value))} /></label>
      <label className="checkbox-row"><input type="checkbox" checked={bias} disabled={locked} onChange={event => setBias(event.target.checked)} />Dense biases</label>
      <p>Activation and dropout follow each hidden layer only. The output is linear. Applying replaces this internal module and trains it from scratch.</p>
      {error && <p role="alert">{error}</p>}
      <button className="secondary-button" disabled={locked} onClick={apply}>Apply internal edit</button>
      {initial && <button className="secondary-button" disabled={locked} onClick={onReset}>Restore original module</button>}
    </>}
  </details>
}

export default function InternalEditor({ layer, targets, locked, onLayer, selectedPath = null }: {
  layer: LayerSpec
  targets?: InternalTarget[]
  locked: boolean
  onLayer: (layer: LayerSpec) => void
  selectedPath?: string | null
}) {
  const [query, setQuery] = useState('')
  const edits = layer.internal_overrides ?? {}
  const update = (path: string, edit?: InternalOverride) => {
    const next = { ...edits }
    if (edit) next[path] = edit
    else delete next[path]
    onLayer({ ...layer, internal_overrides: next })
  }
  return <details className="internal-editor" open={Boolean(selectedPath) || undefined}>
    <summary>Open model internals{Object.keys(edits).length ? ' · custom variant' : ''}</summary>
    <p>Edit actual dense modules inside this model, not layers after it. Module paths describe containment, not execution order. Non-dense operations and connections stay unchanged.</p>
    {Object.keys(edits).length > 0 && <div><strong>Applied edits</strong>{Object.keys(edits).map(path => <div className="internal-reset" key={path}><code>{path}</code><button disabled={locked} onClick={() => update(path)}>Reset</button></div>)}<p>Reset also works if a configuration change invalidates a module path.</p></div>}
    {!targets ? <p>Internal structure appears after successful architecture validation.</p> : targets.length === 0 ? <p>This configuration has no editable Linear or pure dense Sequential modules. Convolutional projections are not dense layers.</p> : <>
      {selectedPath && <p>Selected on canvas: <code>{selectedPath}</code></p>}
      {!selectedPath && <label>Find internal layer<input value={query} onChange={event => setQuery(event.target.value)} placeholder="e.g. projection or temporal" /></label>}
      <p>{targets.length} editable modules (including containing dense stacks).</p>
      {selectedPath && !targets.some(t => t.path === selectedPath) && <p>This path is no longer available. Select a node in the updated graph or reset an applied edit.</p>}
      {targets.filter(target => selectedPath ? target.path === selectedPath : target.path.toLowerCase().includes(query.toLowerCase())).map(target => <TargetEditor
        key={`${target.path}:${target.in_features}:${target.out_features}:${JSON.stringify(target.override)}`}
        target={target} selected={target.path === selectedPath} locked={locked} onApply={edit => update(target.path, edit)} onReset={() => update(target.path)}
      />)}
    </>}
  </details>
}
