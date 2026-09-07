import type { GraphSpec, GraphNodeSpec, GraphNodeInfo } from './types'

export const DEFAULT_ALGEBRA_DIMENSIONS: Record<string, number> = {
  complex: 2,
  split_complex: 2,
  tricomplex: 3,
  quaternion: 4,
  coquaternion: 4,
  cl11: 4,
  octonion: 8,
}

export function denseKind(node: GraphNodeSpec, info?: GraphNodeInfo): 'dense' | 'hyper_dense' | null {
  if (node.kind === 'dense' || node.kind === 'hyper_dense') return node.kind
  if (node.kind === 'source' && info?.category === 'call_module') {
    if (info.label === 'Linear') return 'dense'
    if (info.label === 'HyperDense') return 'hyper_dense'
  }
  return null
}

export function prepareDenseSwap(
  graph: GraphSpec,
  id: string,
  target: 'dense' | 'hyper_dense',
  info: Record<string, GraphNodeInfo>,
  algebra?: string,
  algebraDimensions: Record<string, number> = DEFAULT_ALGEBRA_DIMENSIONS,
): GraphSpec {
  const node = graph.nodes.find(n => n.id === id)
  if (!node || !denseKind(node, info[id])) throw new Error('Select a Dense or HyperDense layer first.')
  const inputs = graph.edges.filter(e => e.target === id)
  if (inputs.length !== 1) throw new Error('Connect exactly one input before switching layer type.')
  const inputShape = info[inputs[0].source]?.shape
  const outputShape = info[id]?.shape
  if (!Array.isArray(inputShape) || !Array.isArray(outputShape) || inputShape.length < 2 || !inputShape.every(d => typeof d === 'number') || !outputShape.every(d => typeof d === 'number')) {
    throw new Error('Validate this layer’s input and output shapes before switching type.')
  }
  const inputWidth = inputShape.at(-1) as number, outputWidth = outputShape.at(-1) as number
  const settings = { ...info[id]?.settings, ...node.params }
  const selectedAlgebra = algebra ?? String(settings.algebra ?? 'quaternion')
  const dimension = algebraDimensions[selectedAlgebra]
  if (target === 'hyper_dense' && (!Number.isInteger(dimension) || dimension < 2)) {
    throw new Error(`Unknown HyperDense algebra ${selectedAlgebra}.`)
  }
  if (target === 'hyper_dense' && (inputWidth % dimension || outputWidth % dimension)) {
    throw new Error(`Cannot preserve this shape: input width ${inputWidth} and output width ${outputWidth} must both be divisible by ${dimension} for ${selectedAlgebra} HyperDense. No padding or projection has been added.`)
  }
  const units = target === 'hyper_dense' ? outputWidth / dimension : outputWidth
  if (outputWidth > 512) throw new Error('This width exceeds the supported replacement layer size.')
  const replacement: GraphNodeSpec = {
    id, kind: target, group: node.group, label: target === 'hyper_dense' ? 'HyperDense' : 'Dense',
    params: { units, bias: settings.bias ?? true, ...(target === 'hyper_dense' ? { algebra: selectedAlgebra } : {}) },
  }
  return { ...graph, nodes: graph.nodes.map(n => n.id === id ? replacement : n),
    edges: graph.edges.map(e => e.target === id ? { ...e, port: 'x' } : e) }
}
