import { useMemo, useState } from 'react'
import {
  DndContext,
  DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  TouchSensor,
  closestCenter,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import * as Dialog from '@radix-ui/react-dialog'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  ArrowDown,
  ArrowUp,
  BarChart3,
  Blocks,
  Check,
  ChevronRight,
  CircleStop,
  Clock3,
  Copy,
  Download,
  FlaskConical,
  GripVertical,
  Layers3,
  LockKeyhole,
  Play,
  Plus,
  Save,
  Settings2,
  Trash2,
  Upload,
  X,
  Zap,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import YAML from 'yaml'
import { api } from './api'
import type {
  ArchitectureSpec,
  Catalog,
  CatalogLayer,
  EvaluationSpec,
  Job,
  LayerParams,
  LayerSpec,
  TraceEntry,
} from './types'

type View = 'builder' | 'runs' | 'compare'

const dataColumns = ['Copper', 'FCX', 'CLP', 'SCCO']

function clone<T>(value: T): T {
  return structuredClone(value)
}

function newId(type: string): string {
  return `${type}-${crypto.randomUUID().slice(0, 8)}`
}

function evaluationFromPreset(catalog: Catalog, presetName: 'quick' | 'standard' | 'robust'): EvaluationSpec {
  const preset = catalog.evaluation_presets[presetName]
  return {
    ...clone(catalog.evaluation_defaults),
    preset: presetName,
    cells: clone(preset.cells),
    seeds: [...preset.seeds],
    epochs: preset.epochs,
  }
}

function numberList(value: string): number[] {
  return value
    .split(',')
    .map((part) => Number(part.trim()))
    .filter(Number.isFinite)
}

function cellList(value: string) {
  return value
    .split(',')
    .map((part) => part.trim().split(/[x/]/).map(Number))
    .filter(([window, horizon]) => Number.isFinite(window) && Number.isFinite(horizon))
    .map(([window, horizon]) => ({ window, horizon }))
}

function formatMetric(value: number | null | undefined, digits = 4) {
  return value == null || Number.isNaN(value) ? '—' : value.toFixed(digits)
}

function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null
  return <div className="error-notice" role="alert">{error instanceof Error ? error.message : String(error)}</div>
}

function PaletteItem({ layer }: { layer: CatalogLayer }) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: `palette:${layer.type}`,
    data: { layer },
  })
  return (
    <button
      ref={setNodeRef}
      className={`palette-item ${isDragging ? 'dragging' : ''}`}
      style={{ transform: CSS.Translate.toString(transform) }}
      {...listeners}
      {...attributes}
    >
      <Plus size={15} />
      <span>{layer.label}</span>
      <GripVertical size={15} aria-hidden="true" />
    </button>
  )
}

function SortableLayerCard({
  layer,
  index,
  selected,
  trace,
  disabled,
  onSelect,
  onRemove,
  onMove,
}: {
  layer: LayerSpec
  index: number
  selected: boolean
  trace?: TraceEntry
  disabled: boolean
  onSelect: () => void
  onRemove: () => void
  onMove: (direction: -1 | 1) => void
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: layer.id,
    disabled,
  })
  return (
    <article
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={`layer-card ${selected ? 'selected' : ''} ${isDragging ? 'dragging' : ''}`}
      onClick={onSelect}
    >
      <button className="drag-handle" aria-label={`Drag ${layer.type}`} {...attributes} {...listeners} disabled={disabled}>
        <GripVertical size={17} />
      </button>
      <div className="layer-index">{String(index + 1).padStart(2, '0')}</div>
      <div className="layer-main">
        <strong>{layer.type.replaceAll('_', ' ')}</strong>
        <span>{Object.values(layer.params).map(String).join(' · ') || 'No parameters'}</span>
      </div>
      {trace && <div className="shape-badge">{trace.output_shape}</div>}
      <div className="layer-actions">
        <button aria-label="Move layer up" onClick={(event) => { event.stopPropagation(); onMove(-1) }} disabled={disabled || index === 0}><ArrowUp size={14} /></button>
        <button aria-label="Move layer down" onClick={(event) => { event.stopPropagation(); onMove(1) }} disabled={disabled}><ArrowDown size={14} /></button>
        <button aria-label="Remove layer" onClick={(event) => { event.stopPropagation(); onRemove() }} disabled={disabled}><Trash2 size={14} /></button>
      </div>
    </article>
  )
}

function LayerCanvas({
  architecture,
  selectedId,
  trace,
  onSelect,
  onRemove,
  onMove,
}: {
  architecture: ArchitectureSpec
  selectedId: string | null
  trace: TraceEntry[]
  onSelect: (id: string) => void
  onRemove: (id: string) => void
  onMove: (id: string, direction: -1 | 1) => void
}) {
  const { setNodeRef, isOver } = useDroppable({ id: 'canvas' })
  return (
    <div ref={setNodeRef} className={`layer-canvas ${isOver ? 'over' : ''}`}>
      <div className="fixed-node">
        <div className="node-icon"><Activity size={17} /></div>
        <div><span className="eyebrow">Input</span><strong>{architecture.input.representation.replace('_', ' ')}</strong></div>
        <span className="shape-badge">4 features</span>
      </div>
      <div className="connector" />
      <SortableContext items={architecture.layers.map((layer) => layer.id)} strategy={verticalListSortingStrategy}>
        {architecture.layers.map((layer, index) => (
          <div key={layer.id}>
            <SortableLayerCard
              layer={layer}
              index={index}
              selected={layer.id === selectedId}
              trace={trace.find((item) => item.layer_id === layer.id)}
              disabled={Boolean(architecture.locked)}
              onSelect={() => onSelect(layer.id)}
              onRemove={() => onRemove(layer.id)}
              onMove={(direction) => onMove(layer.id, direction)}
            />
            <div className="connector" />
          </div>
        ))}
      </SortableContext>
      {!architecture.layers.length && <div className="empty-drop">Drop layers here</div>}
      <div className="fixed-node head-node">
        <div className="node-icon"><Zap size={17} /></div>
        <div><span className="eyebrow">Forecast head</span><strong>{architecture.head.type.replaceAll('_', ' ')}</strong></div>
        {architecture.head.zero_initialize && <span className="zero-badge">zero init</span>}
      </div>
    </div>
  )
}

function Inspector({
  architecture,
  selectedLayer,
  catalog,
  featureNames,
  onArchitecture,
  onLayer,
}: {
  architecture: ArchitectureSpec
  selectedLayer?: LayerSpec
  catalog: Catalog
  featureNames: string[]
  onArchitecture: (next: ArchitectureSpec) => void
  onLayer: (next: LayerSpec) => void
}) {
  const locked = Boolean(architecture.locked)
  const updateParam = (key: string, raw: string, current: LayerParams[string]) => {
    if (!selectedLayer) return
    let value: string | number | number[] = raw
    if (Array.isArray(current)) value = numberList(raw)
    else if (typeof current === 'number') value = Number(raw)
    onLayer({ ...selectedLayer, params: { ...selectedLayer.params, [key]: value } })
  }
  const moveFeature = (position: number, direction: -1 | 1) => {
    const target = position + direction
    if (target < 0 || target >= architecture.input.feature_order.length) return
    const featureOrder = arrayMove(architecture.input.feature_order, position, target)
    onArchitecture({ ...architecture, input: { ...architecture.input, feature_order: featureOrder } })
  }
  return (
    <aside className="inspector panel-surface">
      <div className="panel-title"><Settings2 size={17} /><span>Inspector</span></div>
      <label>
        <span>Architecture name</span>
        <input value={architecture.name} disabled={locked} onChange={(event) => onArchitecture({ ...architecture, name: event.target.value })} />
      </label>
      <label>
        <span>Input representation</span>
        <select value={architecture.input.representation} disabled={locked} onChange={(event) => onArchitecture({
          ...architecture,
          input: { ...architecture.input, representation: event.target.value as ArchitectureSpec['input']['representation'] },
        })}>
          {catalog.input_representations.map((value) => <option key={value}>{value}</option>)}
        </select>
      </label>
      <div className="field-group">
        <span>Feature order</span>
        <div className="feature-list editable">{architecture.input.feature_order.map((index, position) => <span key={index}>{featureNames[index]}<button aria-label={`Move ${featureNames[index]} earlier`} disabled={locked || position === 0} onClick={() => moveFeature(position, -1)}><ArrowUp size={10} /></button><button aria-label={`Move ${featureNames[index]} later`} disabled={locked || position === architecture.input.feature_order.length - 1} onClick={() => moveFeature(position, 1)}><ArrowDown size={10} /></button><button aria-label={`Remove ${featureNames[index]}`} disabled={locked || architecture.input.feature_order.length === 1} onClick={() => onArchitecture({ ...architecture, input: { ...architecture.input, feature_order: architecture.input.feature_order.filter((item) => item !== index) } })}><X size={10} /></button></span>)}</div>
        {!locked && architecture.input.feature_order.length < featureNames.length && <select className="add-feature" value="" onChange={(event) => event.target.value && onArchitecture({ ...architecture, input: { ...architecture.input, feature_order: [...architecture.input.feature_order, Number(event.target.value)] } })}><option value="">Add feature…</option>{featureNames.map((name, index) => !architecture.input.feature_order.includes(index) && <option value={index} key={name}>{name}</option>)}</select>}
      </div>
      <label>
        <span>Forecast head</span>
        <select value={architecture.head.type} disabled={locked} onChange={(event) => {
          const type = event.target.value as ArchitectureSpec['head']['type']
          onArchitecture({ ...architecture, head: { type, zero_initialize: type !== 'direct' } })
        }}>
          {catalog.head_types.map((value) => <option key={value}>{value}</option>)}
        </select>
      </label>
      <label className="checkbox-row">
        <input type="checkbox" checked={architecture.head.zero_initialize} disabled={locked} onChange={(event) => onArchitecture({ ...architecture, head: { ...architecture.head, zero_initialize: event.target.checked } })} />
        <span>Zero-initialize output</span>
      </label>
      <div className="section-rule" />
      {selectedLayer ? (
        <>
          <div className="selected-label"><Layers3 size={15} /><strong>{selectedLayer.type.replaceAll('_', ' ')}</strong></div>
          {Object.entries(selectedLayer.params).map(([key, value]) => (
            <label key={key}>
              <span>{key.replaceAll('_', ' ')}</span>
              {key === 'algebra' ? (
                <select value={String(value)} disabled={locked} onChange={(event) => updateParam(key, event.target.value, value)}>
                  {catalog.algebras.map((item) => <option key={item}>{item}</option>)}
                </select>
              ) : key === 'kind' ? (
                <select value={String(value)} disabled={locked} onChange={(event) => updateParam(key, event.target.value, value)}>
                  {catalog.activations.map((item) => <option key={item}>{item}</option>)}
                </select>
              ) : (
                <input
                  type={typeof value === 'number' ? 'number' : 'text'}
                  step={key === 'dropout' || key === 'p' ? '0.05' : '1'}
                  value={Array.isArray(value) ? value.join(', ') : value}
                  disabled={locked}
                  onChange={(event) => updateParam(key, event.target.value, value)}
                />
              )}
            </label>
          ))}
        </>
      ) : <p className="muted-copy">Select a layer to edit its parameters.</p>}
    </aside>
  )
}

function EvaluationPanel({
  evaluation,
  catalog,
  onChange,
}: {
  evaluation: EvaluationSpec
  catalog: Catalog
  onChange: (next: EvaluationSpec) => void
}) {
  const loadPreset = (preset: EvaluationSpec['preset']) => onChange(evaluationFromPreset(catalog, preset))
  return (
    <details className="evaluation-panel panel-surface">
      <summary><FlaskConical size={16} />Evaluation settings<span>{evaluation.preset}</span></summary>
      <div className="protocol-note">
        <Check size={15} />
        <span><strong>Main run protocol</strong> · chronological split · training-only scaling · Adam defaults from <code>configs/evaluation.yaml</code></span>
      </div>
      <div className="evaluation-grid">
        <label><span>Preset</span><select value={evaluation.preset} onChange={(event) => loadPreset(event.target.value as EvaluationSpec['preset'])}><option value="quick">Quick</option><option value="standard">Standard</option><option value="robust">Robust</option></select></label>
        <label><span>Target</span><select value={evaluation.target_column} onChange={(event) => onChange({ ...evaluation, target_column: event.target.value })}>{dataColumns.map((column) => <option key={column}>{column}</option>)}</select></label>
        <label><span>Cells (window/horizon)</span><input value={evaluation.cells.map((cell) => `${cell.window}/${cell.horizon}`).join(', ')} onChange={(event) => onChange({ ...evaluation, cells: cellList(event.target.value) })} /></label>
        <label><span>Seeds</span><input value={evaluation.seeds.join(', ')} onChange={(event) => onChange({ ...evaluation, seeds: numberList(event.target.value) })} /></label>
        <label><span>Epochs</span><input type="number" value={evaluation.epochs} onChange={(event) => onChange({ ...evaluation, epochs: Number(event.target.value) })} /></label>
        <label><span>Batch size</span><input type="number" value={evaluation.batch_size} onChange={(event) => onChange({ ...evaluation, batch_size: Number(event.target.value) })} /></label>
        <label><span>Learning rate</span><input type="number" step="0.0001" value={evaluation.learning_rate} onChange={(event) => onChange({ ...evaluation, learning_rate: Number(event.target.value) })} /></label>
        <label><span>Loss</span><select value={evaluation.loss} onChange={(event) => onChange({ ...evaluation, loss: event.target.value as EvaluationSpec['loss'] })}><option value="mse">MSE</option><option value="mae">MAE</option><option value="huber">Huber</option></select></label>
        <label><span>Patience</span><input type="number" value={evaluation.early_stopping_patience ?? ''} onChange={(event) => onChange({ ...evaluation, early_stopping_patience: event.target.value ? Number(event.target.value) : null })} /></label>
        <label><span>Device</span><select value={evaluation.device} onChange={(event) => onChange({ ...evaluation, device: event.target.value as EvaluationSpec['device'] })}><option value="cpu">CPU</option><option value="auto">Auto</option><option value="mps">Apple MPS</option><option value="cuda">CUDA</option></select></label>
        <label className="checkbox-row"><input type="checkbox" checked={evaluation.shuffle} onChange={(event) => onChange({ ...evaluation, shuffle: event.target.checked })} /><span>Shuffle training batches</span></label>
        <label className="checkbox-row"><input type="checkbox" checked={evaluation.restore_best_weights} disabled={evaluation.early_stopping_patience == null} onChange={(event) => onChange({ ...evaluation, restore_best_weights: event.target.checked })} /><span>Restore best weights</span></label>
      </div>
    </details>
  )
}

function BuilderView({ catalog, jobs }: { catalog: Catalog; jobs: Job[] }) {
  const queryClient = useQueryClient()
  const initialPreset = catalog.presets.find((preset) => preset.preset_id === 'paper-quaternion') ?? catalog.presets[0]
  const [architecture, setArchitecture] = useState<ArchitectureSpec>(() => clone(initialPreset))
  const [selectedId, setSelectedId] = useState<string | null>(architecture.layers[0]?.id ?? null)
  const [evaluation, setEvaluation] = useState<EvaluationSpec>(() => evaluationFromPreset(catalog, 'quick'))
  const [savedId, setSavedId] = useState('')
  const [importError, setImportError] = useState<Error | null>(null)
  const savedArchitectures = useQuery({ queryKey: ['architectures'], queryFn: api.architectures })
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )
  const previewCell = evaluation.cells[0] ?? { window: 10, horizon: 1 }
  const validation = useQuery({
    queryKey: ['architecture-validation', architecture, previewCell],
    queryFn: () => api.validate(architecture, previewCell.window, previewCell.horizon),
    retry: false,
  })
  const saveMutation = useMutation({
    mutationFn: () => api.saveArchitecture({ ...architecture, id: savedId || undefined }),
    onSuccess: (record) => { setSavedId(record.id); queryClient.invalidateQueries({ queryKey: ['architectures'] }) },
  })
  const runMutation = useMutation({
    mutationFn: () => api.submit(architecture, evaluation),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
  })

  const onDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || architecture.locked) return
    const activeId = String(active.id)
    const overId = String(over.id)
    if (activeId.startsWith('palette:')) {
      const palette = active.data.current?.layer as CatalogLayer | undefined
      if (!palette) return
      const layer: LayerSpec = { id: newId(palette.type), type: palette.type, params: clone(palette.defaults) }
      const overIndex = architecture.layers.findIndex((item) => item.id === overId)
      const insertion = overIndex < 0 ? architecture.layers.length : overIndex
      const layers = [...architecture.layers]
      layers.splice(insertion, 0, layer)
      setArchitecture({ ...architecture, layers })
      setSelectedId(layer.id)
      return
    }
    const from = architecture.layers.findIndex((item) => item.id === activeId)
    const to = architecture.layers.findIndex((item) => item.id === overId)
    if (from >= 0 && to >= 0 && from !== to) setArchitecture({ ...architecture, layers: arrayMove(architecture.layers, from, to) })
  }
  const moveLayer = (id: string, direction: -1 | 1) => {
    const from = architecture.layers.findIndex((layer) => layer.id === id)
    const to = from + direction
    if (from >= 0 && to >= 0 && to < architecture.layers.length) setArchitecture({ ...architecture, layers: arrayMove(architecture.layers, from, to) })
  }
  const selectedLayer = architecture.layers.find((layer) => layer.id === selectedId)
  const featureNames = [evaluation.target_column, ...dataColumns.filter((column) => column !== evaluation.target_column)]
  const loadPreset = (id: string) => {
    const preset = catalog.presets.find((item) => item.preset_id === id)
    if (!preset) return
    setArchitecture(clone(preset))
    setSelectedId(preset.layers[0]?.id ?? null)
    setSavedId('')
  }
  const loadSaved = (id: string) => {
    const record = savedArchitectures.data?.find((item) => item.id === id)
    if (!record) return
    setSavedId(id)
    setArchitecture({ ...clone(record.spec), locked: false, preset_id: undefined })
    setSelectedId(record.spec.layers[0]?.id ?? null)
  }
  const cloneArchitecture = () => setArchitecture({ ...clone(architecture), name: `${architecture.name} — experiment`, locked: false, preset_id: undefined })
  const exportArchitecture = () => {
    const clean = clone(architecture)
    delete clean.locked
    delete clean.preset_id
    const blob = new Blob([YAML.stringify(clean)], { type: 'application/yaml' })
    const anchor = document.createElement('a')
    anchor.href = URL.createObjectURL(blob)
    anchor.download = `${architecture.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'architecture'}.yaml`
    anchor.click()
    URL.revokeObjectURL(anchor.href)
  }
  const importArchitecture = async (file: File) => {
    try {
      const parsed = YAML.parse(await file.text()) as ArchitectureSpec
      setArchitecture({ ...parsed, locked: false, preset_id: undefined })
      setSelectedId(parsed.layers?.[0]?.id ?? null)
      setSavedId('')
      setImportError(null)
    } catch (error) {
      setImportError(error instanceof Error ? error : new Error('Unable to parse architecture file'))
    }
  }
  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
      <section className="builder-toolbar panel-surface">
        <div className="starting-points"><div className="preset-control"><span>Preset</span><select value={architecture.preset_id ?? ''} onChange={(event) => loadPreset(event.target.value)}><option value="" disabled>Custom architecture</option>{catalog.presets.map((preset) => <option value={preset.preset_id} key={preset.preset_id}>{preset.name}</option>)}</select></div><div className="preset-control"><span>Saved</span><select value={savedId} onChange={(event) => loadSaved(event.target.value)}><option value="">Choose saved…</option>{savedArchitectures.data?.map((record) => <option value={record.id} key={record.id}>{record.spec.name}</option>)}</select></div></div>
        <div className="toolbar-actions">
          {architecture.locked && <button className="secondary-button" onClick={cloneArchitecture}><Copy size={15} />Clone to edit</button>}
          <label className="secondary-button file-button"><Upload size={15} />Import<input type="file" accept=".yaml,.yml,.json" onChange={(event) => event.target.files?.[0] && importArchitecture(event.target.files[0])} /></label>
          <button className="secondary-button" onClick={exportArchitecture}><Download size={15} />Export</button>
          <button className="secondary-button" disabled={Boolean(architecture.locked) || saveMutation.isPending || !validation.data?.valid} onClick={() => saveMutation.mutate()}><Save size={15} />Save</button>
          <button className="primary-button" disabled={!validation.data?.valid || runMutation.isPending || !evaluation.cells.length || !evaluation.seeds.length} onClick={() => runMutation.mutate()}><Play size={15} />Run validation</button>
        </div>
      </section>
      <ErrorNotice error={importError || validation.error || saveMutation.error || runMutation.error} />
      {runMutation.isSuccess && <div className="success-notice"><Check size={15} />Experiment queued. Track it in Runs.</div>}
      <section className="builder-grid">
        <aside className="palette panel-surface">
          <div className="panel-title"><Blocks size={17} /><span>Layer palette</span></div>
          <p className="muted-copy">Drag a block into the pipeline. Invalid combinations remain visible but cannot run.</p>
          {catalog.categories.map((category) => <div className="palette-group" key={category.name}><h3>{category.name}</h3>{category.layers.map((layer) => <PaletteItem key={layer.type} layer={layer} />)}</div>)}
        </aside>
        <main className="canvas-column">
          <div className="canvas-heading">
            <div><span className="eyebrow">Architecture canvas</span><h2>{architecture.name}</h2></div>
            {architecture.locked && <span className="locked-badge"><LockKeyhole size={13} />Paper reference</span>}
          </div>
          <LayerCanvas
            architecture={architecture}
            selectedId={selectedId}
            trace={validation.data?.trace ?? []}
            onSelect={setSelectedId}
            onRemove={(id) => { setArchitecture({ ...architecture, layers: architecture.layers.filter((layer) => layer.id !== id) }); if (selectedId === id) setSelectedId(null) }}
            onMove={moveLayer}
          />
          <div className="model-stats">
            <div><span>Parameters</span><strong>{validation.data?.parameters.toLocaleString() ?? '—'}</strong></div>
            <div><span>Receptive field</span><strong>{validation.data?.receptive_field ?? '—'} steps</strong></div>
            <div><span>Preview</span><strong>w{previewCell.window} / h{previewCell.horizon}</strong></div>
            <div><span>Queued/running</span><strong>{jobs.filter((job) => !['complete', 'failed', 'cancelled', 'interrupted'].includes(job.status.state)).length}</strong></div>
          </div>
          <EvaluationPanel evaluation={evaluation} catalog={catalog} onChange={setEvaluation} />
        </main>
        <Inspector
          architecture={architecture}
          selectedLayer={selectedLayer}
          catalog={catalog}
          featureNames={featureNames}
          onArchitecture={setArchitecture}
          onLayer={(next) => setArchitecture({ ...architecture, layers: architecture.layers.map((layer) => layer.id === next.id ? next : layer) })}
        />
      </section>
    </DndContext>
  )
}

function statusClass(state: Job['status']['state']) {
  return `status-pill status-${state.replace('_', '-')}`
}

function RunsView({ jobs, legacyRuns }: { jobs: Job[]; legacyRuns: Record<string, string | number | null>[] }) {
  const queryClient = useQueryClient()
  const [logJob, setLogJob] = useState<string | null>(null)
  const cancelMutation = useMutation({
    mutationFn: api.cancel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const log = useQuery({ queryKey: ['job-log', logJob], queryFn: () => api.log(logJob!), enabled: Boolean(logJob), refetchInterval: 2000 })
  const reproduction = legacyRuns.some((run) => run.candidate != null)
  return (
    <section className="content-view">
      <div className="view-heading"><div><span className="eyebrow">Experiment lifecycle</span><h2>Runs</h2></div><span>{jobs.length} persisted jobs</span></div>
      {!jobs.length && <div className="large-empty panel-surface"><Clock3 size={34} /><h3>No experiments yet</h3><p>Build an architecture and submit a validation run.</p></div>}
      <div className="jobs-list">
        {jobs.map((job) => {
          const percent = job.status.total ? Math.round(job.status.completed / job.status.total * 100) : 0
          const active = ['queued', 'starting', 'running'].includes(job.status.state)
          return <article className="job-card panel-surface" key={job.id}>
            <div className="job-top"><div><span className={statusClass(job.status.state)}>{job.status.state}</span><h3>{job.status.architecture_name}</h3><p>{job.status.phase.replace('_', ' ')} · {job.status.protocol ?? 'chronological-v1'} · {job.status.preset} · {job.id}</p></div><div className="job-actions"><button className="text-button" onClick={() => setLogJob(logJob === job.id ? null : job.id)}>View log</button>{active && <button className="danger-button" disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate(job.id)}><CircleStop size={14} />Cancel</button>}</div></div>
            <div className="progress-track"><div style={{ width: `${percent}%` }} /></div>
            <div className="job-meta"><span>{job.status.completed} / {job.status.total || '—'} runs</span><span>{percent}%</span>{job.status.current && <span>w{job.status.current.window}/h{job.status.current.horizon} · seed {job.status.current.seed} · epoch {job.status.current.epoch}/{job.status.current.epochs}</span>}<span>Updated {new Date(job.status.updated_at).toLocaleString()}</span></div>
            {job.status.error && <div className="inline-error">{job.status.error}</div>}
            {job.summary.length > 0 && <div className="result-strip"><span>Mean MAE ratio <strong>{formatMetric(job.summary.reduce((sum, row) => sum + row.mae_ratio, 0) / job.summary.length, 3)}</strong></span><span>Mean MSE ratio <strong>{formatMetric(job.summary.reduce((sum, row) => sum + row.mse_ratio, 0) / job.summary.length, 3)}</strong></span><span>Parameters <strong>{Math.round(job.summary[0].parameters).toLocaleString()}</strong></span></div>}
            {logJob === job.id && <pre className="training-log">{log.data || 'Waiting for worker output…'}</pre>}
          </article>
        })}
      </div>
      {legacyRuns.length > 0 && <section className="legacy-runs panel-surface"><div className="panel-title"><Activity size={17} /><span>{reproduction ? 'Released-notebook reproduction results' : 'Main evaluation results'}</span></div><p className="muted-copy">Read-only results from the directory supplied with <code>--results</code>. {reproduction ? 'MAE values are normalized cross-validation scores, matching the paper reproduction runner.' : 'MAE and MSE are in original target units, matching the leakage-safe main runner.'}</p><div className="comparison-table"><table><thead><tr><th>Model</th><th>Window</th><th>Horizon</th><th>{reproduction ? 'Candidate' : 'Seed'}</th><th>{reproduction ? 'CV MAE (scaled)' : 'MAE'}</th><th>{reproduction ? 'MAE std' : 'MSE'}</th><th>Parameters</th></tr></thead><tbody>{legacyRuns.slice(-12).reverse().map((run, index) => <tr key={`${run.model}-${run.seed ?? run.candidate}-${index}`}><td><strong>{run.model}</strong></td><td>{run.window}</td><td>{run.horizon}</td><td>{reproduction ? run.candidate : run.seed}</td><td>{formatMetric(Number(run.mae), 5)}</td><td>{formatMetric(reproduction ? Number(run.mae_std) : Number(run.mse), 5)}</td><td>{Number(run.parameters).toLocaleString()}</td></tr>)}</tbody></table></div></section>}
    </section>
  )
}

function FinalTestDialog({ job, onComplete }: { job: Job; onComplete: () => void }) {
  const [acknowledged, setAcknowledged] = useState(false)
  const mutation = useMutation({ mutationFn: () => api.finalTest(job.id), onSuccess: onComplete })
  return <Dialog.Root><Dialog.Trigger asChild><button className="text-button" disabled={job.status.preset === 'quick'}><FlaskConical size={14} />Final test</button></Dialog.Trigger><Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content"><Dialog.Title>Unlock the held-out test set?</Dialog.Title><Dialog.Description>This candidate can be tested only once. Use this after architecture selection is complete.</Dialog.Description><label className="checkbox-row acknowledgement"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /><span>I understand this consumes the final evaluation for this candidate.</span></label><ErrorNotice error={mutation.error} /><div className="dialog-actions"><Dialog.Close asChild><button className="secondary-button">Cancel</button></Dialog.Close><button className="primary-button" disabled={!acknowledged || mutation.isPending} onClick={() => mutation.mutate()}>Start final test</button></div><Dialog.Close className="dialog-close" aria-label="Close"><X size={17} /></Dialog.Close></Dialog.Content></Dialog.Portal></Dialog.Root>
}

function CompareView({ jobs }: { jobs: Job[] }) {
  const queryClient = useQueryClient()
  const candidates = jobs.filter((job) => job.status.state === 'complete' && job.status.phase === 'validation' && job.summary.length)
  const finalJobs = jobs.filter((job) => job.status.state === 'complete' && job.status.phase === 'final_test' && job.summary.length)
  const [selected, setSelected] = useState<string[]>([])
  const visible = selected.length ? candidates.filter((job) => selected.includes(job.id)) : candidates
  const data = visible.map((job) => ({
    id: job.id,
    name: job.status.architecture_name,
    mae: job.summary.reduce((sum, row) => sum + row.mae_ratio, 0) / job.summary.length,
    mse: job.summary.reduce((sum, row) => sum + row.mse_ratio, 0) / job.summary.length,
    parameters: job.summary.reduce((sum, row) => sum + row.parameters, 0) / job.summary.length,
  }))
  const leadSource = visible[0]
  const leadData = leadSource ? [...new Set(leadSource.per_lead.map((row) => Number(row.lead)))].sort((a, b) => a - b).map((lead) => {
    const rows = leadSource.per_lead.filter((row) => Number(row.lead) === lead)
    const mae = rows.reduce((sum, row) => sum + Number(row.mae), 0) / rows.length
    const baseline = rows.reduce((sum, row) => sum + Number(row.persistence_mae), 0) / rows.length
    return { lead, mae, persistence: baseline }
  }) : []
  const pareto = new Set(data.filter((candidate) => !data.some((other) => other.id !== candidate.id && other.mae <= candidate.mae && other.mse <= candidate.mse && (other.mae < candidate.mae || other.mse < candidate.mse))).map((item) => item.id))
  return (
    <section className="content-view">
      <div className="view-heading"><div><span className="eyebrow">Validation only</span><h2>Compare architectures</h2></div><span>Below 1.0 beats persistence</span></div>
      {!candidates.length && <div className="large-empty panel-surface"><BarChart3 size={34} /><h3>No completed validation runs</h3><p>Comparison results appear after a Standard or Robust experiment completes.</p></div>}
      {candidates.length > 0 && <>
        <div className="compare-selector panel-surface">{candidates.map((job) => <label className="checkbox-row" key={job.id}><input type="checkbox" checked={selected.includes(job.id)} onChange={(event) => setSelected(event.target.checked ? [...selected, job.id] : selected.filter((id) => id !== job.id))} /><span>{job.status.architecture_name}</span><small>{job.status.preset}</small></label>)}</div>
        <div className="charts-grid">
          <article className="chart-panel panel-surface"><h3>Error relative to persistence</h3><ResponsiveContainer width="100%" height={310}><BarChart data={data}><CartesianGrid stroke="#263956" vertical={false} /><XAxis dataKey="name" stroke="#91a5bc" tick={{ fontSize: 11 }} /><YAxis stroke="#91a5bc" /><Tooltip contentStyle={{ background: '#0d1a2a', border: '1px solid #2b4362' }} /><Legend /><Bar dataKey="mae" name="MAE ratio" fill="#48d7ae" radius={[4, 4, 0, 0]} /><Bar dataKey="mse" name="MSE ratio" fill="#58a6ff" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></article>
          <article className="chart-panel panel-surface"><h3>Accuracy–parameter trade-off</h3><ResponsiveContainer width="100%" height={310}><ScatterChart><CartesianGrid stroke="#263956" /><XAxis type="number" dataKey="parameters" name="Parameters" stroke="#91a5bc" /><YAxis type="number" dataKey="mae" name="MAE ratio" stroke="#91a5bc" /><Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#0d1a2a', border: '1px solid #2b4362' }} /><Scatter data={data} fill="#ffca5c" /></ScatterChart></ResponsiveContainer></article>
        </div>
        {leadSource && leadData.length > 0 && <article className="chart-panel panel-surface lead-chart"><h3>Per-lead MAE · {leadSource.status.architecture_name}</h3><ResponsiveContainer width="100%" height={250}><LineChart data={leadData}><CartesianGrid stroke="#263956" vertical={false} /><XAxis dataKey="lead" stroke="#91a5bc" label={{ value: 'Forecast lead', position: 'insideBottom', offset: -2 }} /><YAxis stroke="#91a5bc" /><Tooltip contentStyle={{ background: '#0d1a2a', border: '1px solid #2b4362' }} /><Legend /><Line type="monotone" dataKey="mae" name="Architecture MAE" stroke="#48d7ae" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="persistence" name="Persistence MAE" stroke="#58a6ff" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></article>}
        <div className="comparison-table panel-surface"><table><thead><tr><th>Architecture</th><th>Preset</th><th>MAE ratio</th><th>MSE ratio</th><th>Parameters</th><th>Selection</th><th /></tr></thead><tbody>{visible.map((job) => { const row = data.find((item) => item.id === job.id)!; return <tr key={job.id}><td><strong>{row.name}</strong></td><td>{job.status.preset}</td><td className={row.mae < 1 ? 'metric-good' : ''}>{formatMetric(row.mae, 3)}</td><td className={row.mse < 1 ? 'metric-good' : ''}>{formatMetric(row.mse, 3)}</td><td>{Math.round(row.parameters).toLocaleString()}</td><td>{pareto.has(job.id) ? <span className="pareto-badge">Pareto best</span> : 'Dominated'}</td><td><FinalTestDialog job={job} onComplete={() => queryClient.invalidateQueries({ queryKey: ['jobs'] })} /></td></tr> })}</tbody></table></div>
      </>}
      {finalJobs.length > 0 && <section className="final-results panel-surface"><div className="panel-title"><LockKeyhole size={17} /><span>Held-out final tests</span></div><p className="muted-copy">These results are separated from architecture selection and cannot be rerun for the same candidate configuration.</p><div className="comparison-table"><table><thead><tr><th>Architecture</th><th>Cell</th><th>Test MAE</th><th>Test MSE</th><th>MAE ratio</th><th>MSE ratio</th></tr></thead><tbody>{finalJobs.flatMap((job) => job.summary.map((row) => <tr key={`${job.id}-${row.window}-${row.horizon}`}><td><strong>{job.status.architecture_name}</strong></td><td>w{row.window}/h{row.horizon}</td><td>{formatMetric(row.mae_mean, 5)}</td><td>{formatMetric(row.mse_mean, 5)}</td><td className={row.mae_ratio < 1 ? 'metric-good' : ''}>{formatMetric(row.mae_ratio, 3)}</td><td className={row.mse_ratio < 1 ? 'metric-good' : ''}>{formatMetric(row.mse_ratio, 3)}</td></tr>))}</tbody></table></div></section>}
    </section>
  )
}

export default function App() {
  const [view, setView] = useState<View>('builder')
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: api.catalog, staleTime: Infinity })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 1500 })
  const legacyRuns = useQuery({ queryKey: ['legacy-runs'], queryFn: api.legacyRuns, staleTime: 5000 })
  const activeCount = useMemo(() => (jobs.data ?? []).filter((job) => ['queued', 'starting', 'running'].includes(job.status.state)).length, [jobs.data])
  if (catalog.isLoading) return <div className="app-loading"><div className="loading-mark"><Layers3 /></div><p>Loading architecture catalog…</p></div>
  if (catalog.error || !catalog.data) return <div className="app-loading"><ErrorNotice error={catalog.error ?? new Error('Catalog unavailable')} /></div>
  return (
    <div className="app-shell">
      <header className="app-header">
        <button className="brand" onClick={() => setView('builder')}><span className="brand-mark"><Layers3 size={19} /></span><span><strong>HyperCast<span>4D</span></strong><small>Architecture Playground</small></span></button>
        <nav aria-label="Primary navigation">
          <button className={view === 'builder' ? 'active' : ''} onClick={() => setView('builder')}><Blocks size={16} />Builder</button>
          <button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}><Clock3 size={16} />Runs{activeCount > 0 && <span className="nav-count">{activeCount}</span>}</button>
          <button className={view === 'compare' ? 'active' : ''} onClick={() => setView('compare')}><BarChart3 size={16} />Compare</button>
        </nav>
        <div className="local-badge"><span />Local workspace</div>
      </header>
      <div className="app-content">
        {view === 'builder' && <BuilderView catalog={catalog.data} jobs={jobs.data ?? []} />}
        {view === 'runs' && <RunsView jobs={jobs.data ?? []} legacyRuns={legacyRuns.data ?? []} />}
        {view === 'compare' && <CompareView jobs={jobs.data ?? []} />}
      </div>
      <footer className="app-footer"><span>Data and training stay on this machine</span><span><ChevronRight size={13} /> Test metrics remain locked during architecture search</span></footer>
    </div>
  )
}
