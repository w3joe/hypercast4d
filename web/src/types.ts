export type LayerParams = Record<string, string | number | number[]>

export type LayerSpec = {
  id: string
  type: string
  params: LayerParams
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
