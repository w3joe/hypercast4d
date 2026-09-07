import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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
      method_collection: {
        sources: { survey: { title: 'Survey', venue: 'FCS', url: 'https://example.com/paper.pdf', pages: '13' } },
        methods: [{ id: 'quaternion', name: 'Quaternion', family: 'Hypercomplex', kind: 'neural', status: 'adaptation', preset_id: 'paper-quaternion', sources: [{ source_id: 'survey', page: 13 }], notes: 'Test preset.' }],
      },
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
    } : path.includes('convert') ? executableGraph : path.includes('validate') ? {
      graph_nodes: graphInfo, warnings: [],
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


vi.mock('@xyflow/react', () => ({
  ReactFlowProvider: ({ children }: any) => children,
  useReactFlow: () => ({ fitView: vi.fn(), screenToFlowPosition: (p: any) => p }),
  Handle: () => null, Position: { Top: 'top', Bottom: 'bottom' }, MarkerType: { ArrowClosed: 'arrowclosed' }, Background: () => null, Controls: () => null, MiniMap: () => null,
  applyNodeChanges: (_: any, nodes: any) => nodes,
  ReactFlow: ({ nodes, nodeTypes, onSelectionChange, children }: any) => <div data-testid="flow">{nodes.map((n: any) => {
    const Component = nodeTypes[n.type]
    return <div key={n.id}><button onClick={() => onSelectionChange({ nodes: [n], edges: [] })}>Select {n.id}</button><Component data={n.data} /></div>
  })}{children}</div>,
}))
const executableGraph = { schema_version: 2, revision: 'test', name: 'Paper Quaternion', sources: { s0: paperPreset },
  nodes: [{ id: 'input', kind: 'source', params: {} }, { id: 'linear', kind: 'source', params: {}, group: 'core', source_ref: { source: 's0', node: 'linear' }, module_ref: 'weights' }, { id: 'output', kind: 'source', params: {} }],
  edges: [{ source: 'input', target: 'linear', port: 'args/0' }, { source: 'linear', target: 'output', port: 'args/0' }], groups: [{ id: 'core', label: 'Model core' }], output: 'output' }
const graphInfo = {
  input: { label: 'Input', category: 'placeholder', ports: [], shape: [2, 10, 4], settings: {} },
  linear: { label: 'Linear', category: 'call_module', ports: ['args/0'], shape: [2, 1], settings: { out_features: 1, bias: true }, source_path: 'core.projection' },
  output: { label: 'Output', category: 'output', ports: ['args/0'], shape: [2, 1], settings: {} },
}
function renderApp() { render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><App /></QueryClientProvider>) }

describe('graph-native playground', () => {
  it('switches Dense to HyperDense directly and saves matched units and algebra', async () => {
    const originalFetch = fetch
    const saved: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const path = String(input), payload = options?.body ? JSON.parse(String(options.body)) : null
      if (path.endsWith('/architectures') && payload) { saved.push(payload); return new Response(JSON.stringify({ id: 'saved', spec: payload })) }
      const response = await originalFetch(input, options)
      if (!path.endsWith('/validate')) return response
      const result = await response.json(), node = payload.architecture.nodes.find((n: any) => n.id === 'linear')
      result.graph_nodes.linear = { ...graphInfo.linear, shape: [2, 10, 32], label: node.kind === 'source' ? 'Linear' : node.kind, category: node.kind === 'source' ? 'call_module' : 'custom', settings: node.kind === 'source' ? { out_features: 32, bias: false } : node.params }
      return new Response(JSON.stringify(result))
    }))
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Expand' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Select linear' }))
    const selector = await screen.findByRole('combobox', { name: 'Layer type' })
    await waitFor(() => expect(selector).toBeEnabled())
    fireEvent.change(selector, { target: { value: 'hyper_dense' } })
    await waitFor(() => expect(selector).toHaveValue('hyper_dense'))
    const algebra = screen.getByRole('combobox', { name: 'Algebra' })
    fireEvent.change(algebra, { target: { value: 'cl11' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(saved[0].nodes.find((n: any) => n.id === 'linear').params).toEqual({ units: 8, bias: false, algebra: 'cl11' })
    expect(saved[0].edges).toEqual([{ source: 'input', target: 'linear', port: 'x' }, { source: 'linear', target: 'output', port: 'args/0' }])
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Layer type' })).toHaveValue('dense'))
  })
  it('keeps the original layer when a direct swap is incompatible', async () => {
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Expand' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Select linear' }))
    const selector = await screen.findByRole('combobox', { name: 'Layer type' })
    await waitFor(() => expect(selector).toBeEnabled())
    fireEvent.change(selector, { target: { value: 'hyper_dense' } })
    expect(await screen.findByText(/must both be divisible by 4/)).toBeVisible()
    expect(selector).toHaveValue('dense')
  })
  it('does not apply a swap that changes the layer shape in another evaluation cell', async () => {
    const originalFetch = fetch
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const response = await originalFetch(input, options)
      if (!String(input).endsWith('/validate')) return response
      const payload = JSON.parse(String(options?.body)), result = await response.json()
      const isHyper = payload.architecture.nodes.find((n: any) => n.id === 'linear').kind === 'hyper_dense'
      result.graph_nodes.linear = { ...graphInfo.linear, shape: [2, 10, isHyper && payload.horizon === 3 ? 16 : 32], settings: { out_features: 32, bias: true } }
      return new Response(JSON.stringify(result))
    }))
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(screen.getByText('Evaluation settings'))
    fireEvent.change(screen.getByLabelText('Cells (window/horizon)'), { target: { value: '10/1, 10/3' } })
    fireEvent.click(await screen.findByRole('button', { name: 'Expand' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Select linear' }))
    const selector = await screen.findByRole('combobox', { name: 'Layer type' })
    await waitFor(() => expect(selector).toBeEnabled())
    fireEvent.change(selector, { target: { value: 'hyper_dense' } })
    expect(await screen.findByText(/Original layer kept/)).toBeVisible()
    expect(selector).toHaveValue('dense')
  })
  it('keeps advanced tools and the layer palette out of the initial workspace', async () => {
    renderApp()
    await screen.findByRole('button', { name: 'Clone to edit' })
    expect(screen.getByRole('combobox', { name: 'Load graph preset' })).toHaveValue('paper-quaternion')
    expect(screen.getByRole('group', { name: 'Paper baselines' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Load saved graph' })).toBeDisabled()
    expect(screen.getByRole('option', { name: 'No saved experiments yet' })).toBeInTheDocument()
    expect(screen.getByText('2. Edit architecture')).toBeVisible()
    expect(screen.getByText('3. Run experiment')).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Share selected weights' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Export YAML' })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox', { name: 'Operation' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Add layers' }))
    expect(screen.getByRole('combobox', { name: 'Operation' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Advanced tools' }))
    expect(screen.getByRole('button', { name: 'Export YAML' })).toBeVisible()
  })
  it('edits actual layers on the same canvas and saves the executable graph', async () => {
    const originalFetch = fetch
    const requests: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      if (options?.body) requests.push({ path: String(input), body: JSON.parse(String(options.body)) })
      if (String(input).endsWith('/architectures') && options?.body) return new Response(JSON.stringify({ id: 'saved', spec: executableGraph }))
      return originalFetch(input, options)
    }))
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Expand' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Select linear' }))
    const width = await screen.findByRole('textbox', { name: 'out_features' })
    fireEvent.change(width, { target: { value: '64' } }); fireEvent.blur(width)
    await waitFor(() => expect(requests.some(r => r.path.endsWith('/validate') && r.body.architecture.nodes.some((n: any) => n.params.out_features === 64))).toBe(true))
    expect(screen.getAllByTestId('flow')).toHaveLength(1)
    expect(requests.some(r => r.path.includes('internal-graph'))).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(requests.some(r => r.path.endsWith('/architectures') && r.body.schema_version === 2 && r.body.view)).toBe(true))
  })
  it('preserves edits across navigation and supports undo and library loading', async () => {
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Graph name' }), { target: { value: 'My experiment' } })
    fireEvent.click(screen.getByRole('button', { name: /^Runs/ })); fireEvent.click(screen.getByRole('button', { name: /^Builder$/ }))
    expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('My experiment')
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('Paper Quaternion experiment')
    fireEvent.click(screen.getByRole('button', { name: 'Method collection' }))
    fireEvent.click(screen.getByRole('button', { name: 'Load Quaternion' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('Paper Quaternion'))
  })
  it('validates every cell and blocks training when any one fails', async () => {
    const originalFetch = fetch
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      if (String(input).endsWith('/validate') && options?.body && JSON.parse(String(options.body)).horizon === 3) return new Response(JSON.stringify({ detail: 'Output shape mismatch' }), { status: 422 })
      return originalFetch(input, options)
    }))
    renderApp(); await screen.findByRole('button', { name: 'Clone to edit' })
    await waitFor(() => expect(screen.getByRole('button', { name: /run validation/i })).toBeEnabled())
    fireEvent.click(screen.getByText('Evaluation settings'))
    fireEvent.change(screen.getByLabelText('Cells (window/horizon)'), { target: { value: '10/1, 10/3' } })
    await screen.findByText(/Output shape mismatch/)
    expect(screen.getByRole('button', { name: /run validation/i })).toBeDisabled()
  })
  it('retains local and Modal execution controls', async () => {
    renderApp()
    const method = await screen.findByRole('combobox', { name: /run method/i })
    expect(method).toHaveValue('local')
    fireEvent.change(method, { target: { value: 'modal' } })
    expect(await screen.findByRole('combobox', { name: /modal gpu/i })).toHaveValue('L4')
    expect(await screen.findByText(/usage charges may apply/i)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: /run validation/i })).toBeEnabled())
  })
})
