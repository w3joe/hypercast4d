import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const paperPreset = {
  schema_version: 1,
  name: 'Paper Quaternion',
  input: { representation: 'levels', feature_order: [0, 1, 2, 3] },
  layers: [
    { id: 'hyper', type: 'hyper_dense', params: { units: 8, algebra: 'quaternion' } },
    { id: 'flatten', type: 'flatten', params: {} },
  ],
  head: { type: 'direct', zero_initialize: false },
  locked: true,
  preset_id: 'paper-quaternion',
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class { observe() {}; unobserve() {}; disconnect() {} })
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input)
    const body = path.includes('catalog') ? {
      schema_version: 1,
      categories: [{ name: 'Feature mixing', layers: [{ type: 'dense', label: 'Dense', defaults: { units: 32 } }] }],
      input_representations: ['levels', 'centered', 'differences'],
      head_types: ['direct', 'persistence_residual', 'cumulative_residual'],
      algebras: ['quaternion', 'coquaternion', 'cl11'],
      activations: ['relu', 'gelu', 'silu', 'tanh', 'linear'],
      presets: [paperPreset],
      evaluation_presets: {
        quick: { cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 3, folds: [] },
        standard: { cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 50, folds: [] },
        robust: { cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 50, folds: [] },
      },
      evaluation_defaults: {
        protocol: 'chronological-v1',
        data_path: 'data/raw/paper_data.xlsx',
        target_column: 'Copper',
        batch_size: 32,
        evaluation_batch_size: 256,
        learning_rate: 0.001,
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        adam_epsilon: 1e-7,
        adam_amsgrad: false,
        loss: 'mse',
        shuffle: true,
        early_stopping_patience: null,
        early_stopping_min_delta: 0,
        restore_best_weights: false,
        device: 'cpu',
      },
    } : path.includes('compute') ? {
      local: { available: true },
      modal: {
        available: true,
        sdk_installed: true,
        authenticated: true,
        gpus: [
          { id: 'T4', label: 'T4', description: 'Economy' },
          { id: 'L4', label: 'L4', description: 'Recommended' },
        ],
        setup_command: 'modal setup',
      },
    } : path.includes('validate') ? {
      valid: true,
      spec: paperPreset,
      input_shape: '[B, 10, 4]',
      output_shape: '[B, 1]',
      parameters: 225,
      receptive_field: 1,
      trace: [],
    } : []
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }))
})

describe('architecture playground', () => {
  it('opens on the locked paper builder with validation controls', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={queryClient}><App /></QueryClientProvider>)
    expect(await screen.findByText('Architecture canvas')).toBeInTheDocument()
    expect(screen.getByText('Paper reference')).toBeInTheDocument()
    expect(screen.getByText('Main run protocol')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /clone to edit/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run validation/i })).toBeInTheDocument()
  })

  it('offers local and Modal execution with GPU selection', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={queryClient}><App /></QueryClientProvider>)
    const method = await screen.findByRole('combobox', { name: /run method/i })
    expect(method).toHaveValue('local')
    expect(screen.getByRole('button', { name: /run validation/i })).toBeInTheDocument()
    fireEvent.change(method, { target: { value: 'modal' } })
    const gpu = await screen.findByRole('combobox', { name: /modal gpu/i })
    expect(gpu).toHaveValue('L4')
    expect(await screen.findByText(/usage charges may apply/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run validation/i })).toBeEnabled()
  })
})
