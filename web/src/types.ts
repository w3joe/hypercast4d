export type LayerParams = Record<string, string | number | number[]>

export type GraphNodeSpec = {
  id: string
  kind: string
  params: Record<string, unknown>
  source_ref?: { source: string; node: string }
  module_ref?: string
  group?: string
  label?: string
}
export type GraphEdgeSpec = { source: string; target: string; port: string }
export type GraphSpec = {
  schema_version: 2
  revision: string
  name: string
  sources: Record<string, ArchitectureSpec>
  nodes: GraphNodeSpec[]
  edges: GraphEdgeSpec[]
  groups: { id: string; label: string }[]
  output: string
  locked?: boolean
  preset_id?: string
}
export type GraphViewState = { positions: Record<string, { x: number; y: number }>; collapsed: string[] }
export type GraphNodeInfo = { label: string; ports: string[]; shape: unknown; settings: Record<string, unknown>; category: string; source_path?: string }
export type GraphValidation = { valid: boolean; spec: GraphSpec; parameters: number; graph_nodes: Record<string, GraphNodeInfo>; warnings: string[] }
export type GraphRecord = { id: string; spec: GraphSpec | ArchitectureSpec; view?: GraphViewState }

export type InternalOverride = {
  hidden_units: number[]
  activation: 'relu' | 'gelu' | 'silu' | 'tanh' | 'linear'
  dropout: number
  bias: boolean
}

export type InternalTarget = {
  path: string
  kind: 'dense' | 'dense_stack'
  in_features: number
  out_features: number
  hidden_units: number[]
  structure: string[]
  override: InternalOverride | null
  blocked_by: string | null
}

export type InternalGraph = {
  mode: 'observed'
  notice: string
  targets: InternalTarget[]
  nodes: { id: string; label: string; kind: string; path: string | null; target: string | null; shapes: number[][] }[]
  edges: { source: string; target: string }[]
}

export type LayerSpec = {
  id: string
  type: string
  params: LayerParams
  internal_overrides?: Record<string, InternalOverride>
}

export type ArchitectureSpec = {
  schema_version: number
  name: string
  input: {
    representation: 'levels' | 'centered' | 'differences'
    feature_order: number[]
  }
  layers: LayerSpec[]
  head: {
    type: 'direct' | 'persistence_residual' | 'cumulative_residual'
    zero_initialize: boolean
  }
  locked?: boolean
  preset_id?: string
}

export type CatalogLayer = {
  type: string
  label: string
  defaults: LayerParams
}

export type Catalog = {
  method_collection?: MethodCollection
  schema_version: number
  categories: { name: string; layers: CatalogLayer[] }[]
  input_representations: string[]
  head_types: string[]
  algebras: string[]
  activations: string[]
  presets: ArchitectureSpec[]
  evaluation_presets: Record<string, EvaluationPreset>
  evaluation_defaults: EvaluationDefaults
}

export type MethodCollection = {
  sources: Record<string, { title: string; venue: string; url: string; pages: string }>
  methods: {
    id: string
    name: string
    family: string
    kind: string
    status: 'reference' | 'adaptation'
    preset_id: string | null
    sources: { source_id: string; page: number }[]
    notes: string
  }[]
}

export type EvaluationCell = { window: number; horizon: number }

export type EvaluationPreset = {
  cells: EvaluationCell[]
  seeds: number[]
  epochs: number
  folds: { train_fraction: number; validation_fraction: number }[]
}

export type EvaluationDefaults = {
  protocol: 'chronological-v1'
  data_path: string
  target_column: string
  batch_size: number
  evaluation_batch_size: number
  learning_rate: number
  adam_beta1: number
  adam_beta2: number
  adam_epsilon: number
  adam_amsgrad: boolean
  loss: 'mse' | 'mae' | 'huber'
  shuffle: boolean
  early_stopping_patience: number | null
  early_stopping_min_delta: number
  restore_best_weights: boolean
  device: 'cpu' | 'auto' | 'mps' | 'cuda'
}

export type EvaluationSpec = EvaluationDefaults & {
  preset: 'quick' | 'standard' | 'robust'
  cells: EvaluationCell[]
  seeds: number[]
  epochs: number
}

export type ExecutionSpec = {
  target: 'local' | 'modal'
  gpu: string | null
}

export type ComputeCapabilities = {
  local: { available: boolean }
  modal: {
    available: boolean
    sdk_installed: boolean
    authenticated: boolean
    gpus: { id: string; label: string; description: string }[]
    setup_command: string
  }
}

export type TraceEntry = {
  layer_id: string
  layer_type: string
  input_shape: string
  output_shape: string
  parameters: number
  receptive_field: number
}

export type ValidationResult = {
  valid: boolean
  spec: ArchitectureSpec
  input_shape: string
  output_shape: string
  parameters: number
  receptive_field: number
  trace: TraceEntry[]
  internals?: Record<string, InternalTarget[]>
}

export type ResultSummary = {
  window: number
  horizon: number
  mae_mean: number
  mae_std: number | null
  mse_mean: number
  mse_std: number | null
  persistence_mae: number
  persistence_mse: number
  parameters: number
  train_seconds: number
  epochs_median: number
  mae_ratio: number
  mse_ratio: number
}

export type Job = {
  id: string
  request: {
    phase: 'validation' | 'final_test'
    candidate_hash: string
    architecture: ArchitectureSpec
    evaluation: EvaluationSpec
    execution: ExecutionSpec
    parent_job_id?: string
  }
  status: {
    id: string
    state: 'queued' | 'starting' | 'running' | 'complete' | 'failed' | 'cancelled' | 'interrupted'
    phase: 'validation' | 'final_test'
    completed: number
    total: number
    created_at: string
    updated_at: string
    architecture_name: string
    preset: string
    protocol: 'chronological-v1'
    execution_target: 'local' | 'modal'
    gpu?: string | null
    actual_gpu?: string | null
    modal_call_id?: string | null
    modal_dashboard_url?: string | null
    error?: string | null
    current?: {
      window: number
      horizon: number
      fold: number
      seed: number
      epoch: number
      epochs: number
      train_loss: number
      validation_loss: number
    } | null
  }
  summary: ResultSummary[]
  runs: Record<string, string | number | null>[]
  per_lead: Record<string, string | number | null>[]
}

export type ArchitectureRecord = {
  id: string
  created_at: string
  updated_at: string
  hash: string
  spec: ArchitectureSpec
}
