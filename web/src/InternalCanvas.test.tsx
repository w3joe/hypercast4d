import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { GraphView } from './InternalCanvas'
import type { InternalGraph } from './types'

export const graph: InternalGraph = {
  mode: 'observed', notice: 'One synthetic forward pass.',
  targets: [{ path: 'model.0.temporal', kind: 'dense_stack', in_features: 32, out_features: 32,
    hidden_units: [32], structure: ['Linear', 'ReLU', 'Linear'], override: null, blocked_by: null },
    { path: 'model.0.temporal.0', kind: 'dense', in_features: 32, out_features: 32,
      hidden_units: [], structure: ['Linear'], override: null, blocked_by: null }],
  nodes: [
    { id: 'n0', label: 'History', kind: 'input', path: null, target: null, shapes: [[1, 10, 4]] },
    { id: 'n1', label: 'Linear', kind: 'module', path: 'model.0.temporal.0', target: 'model.0.temporal.0', shapes: [[1, 4, 32]] },
    { id: 'n2', label: 'Add', kind: 'operation', path: null, target: null, shapes: [[1, 32, 4]] },
    { id: 'n3', label: 'Core output', kind: 'output', path: null, target: null, shapes: [[1, 10, 4]] },
  ],
  edges: [{ source: 'n0', target: 'n1' }, { source: 'n0', target: 'n2' }, { source: 'n1', target: 'n2' }, { source: 'n2', target: 'n3' }],
}

describe('internal graph canvas', () => {
  it('draws branches and selects an actual dense node', () => {
    const onSelect = vi.fn()
    const { container } = render(<GraphView graph={graph} selectedPath={null} onSelect={onSelect} />)
    expect(screen.getByText('Add')).toBeVisible()
    expect(container.querySelectorAll('.internal-graph-edges > path')).toHaveLength(4)
    fireEvent.click(screen.getByRole('button', { name: 'Edit model.0.temporal.0' }))
    expect(onSelect).toHaveBeenCalledWith('model.0.temporal.0')
    expect(screen.queryByRole('button', { name: 'Add' })).not.toBeInTheDocument()
  })

  it('selects an entire MLP group and highlights its children', () => {
    const onSelect = vi.fn()
    render(<GraphView graph={graph} selectedPath="model.0.temporal" onSelect={onSelect} />)
    expect(screen.getByRole('button', { name: 'Edit model.0.temporal.0' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.change(screen.getByRole('combobox', { name: 'Edit a containing dense stack' }), { target: { value: 'model.0.temporal' } })
    expect(onSelect).toHaveBeenCalledWith('model.0.temporal')
  })
})
