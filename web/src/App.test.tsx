import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
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
    expect(screen.getByRole('button', { name: /clone to edit/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run validation/i })).toBeInTheDocument()
  })
})
