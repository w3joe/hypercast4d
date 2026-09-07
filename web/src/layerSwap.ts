import type { GraphSpec, GraphNodeSpec, GraphNodeInfo } from './types'

export function denseKind(node: GraphNodeSpec, info?: GraphNodeInfo): 'dense' | 'hyper_dense' | null {
  if (node.kind === 'dense' || node.kind === 'hyper_dense') return node.kind
  if (node.kind === 'source' && info?.category === 'call_module') {
    if (info.label === 'Linear') return 'dense'
    if (info.label === 'HyperDense') return 'hyper_dense'
  }
  return null
}

export function prepareDenseSwap(graph: GraphSpec, id: string, target: 'dense' | 'hyper_dense', info: Record<string, GraphNodeInfo>): GraphSpec {
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
  if (target === 'hyper_dense' && (inputWidth % 4 || outputWidth % 4)) {
    throw new Error(`Cannot preserve this shape: input width ${inputWidth} and output width ${outputWidth} must both be divisible by 4 for HyperDense. No padding or projection has been added.`)
  }
  const units = target === 'hyper_dense' ? outputWidth / 4 : outputWidth
  if (units > (target === 'hyper_dense' ? 128 : 512)) throw new Error('This width exceeds the supported replacement layer size.')
  const settings = { ...info[id]?.settings, ...node.params }
  const replacement: GraphNodeSpec = {
    id, kind: target, group: node.group, label: target === 'hyper_dense' ? 'HyperDense' : 'Dense',
    params: { units, bias: settings.bias ?? true, ...(target === 'hyper_dense' ? { algebra: 'quaternion' } : {}) },
  }
  return { ...graph, nodes: graph.nodes.map(n => n.id === id ? replacement : n),
    edges: graph.edges.map(e => e.target === id ? { ...e, port: 'x' } : e) }
}
