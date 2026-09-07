import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import InternalEditor from './InternalEditor'
import type { InternalTarget, LayerSpec } from './types'

const layer: LayerSpec = { id: 'core', type: 'tslib_tsmixer', params: {} }
const target: InternalTarget = {
  path: 'model.0.temporal', kind: 'dense_stack', in_features: 32, out_features: 32,
  hidden_units: [32], structure: ['Linear(32 → 32)', 'ReLU()', 'Linear(32 → 32)'],
  override: null, blocked_by: null,
}

describe('internal dense editor', () => {
  it('edits the internal stack, leaving outer layers and params unchanged', () => {
    const onLayer = vi.fn()
    render(<InternalEditor layer={layer} targets={[target]} locked={false} onLayer={onLayer} />)
    fireEvent.click(screen.getByText('Open model internals'))
    fireEvent.click(screen.getByText(target.path))
    fireEvent.change(screen.getByLabelText(`Hidden widths for ${target.path}`), { target: { value: '64, 16' } })
    fireEvent.click(screen.getByText('Apply internal edit'))
    expect(onLayer).toHaveBeenCalledWith({ ...layer, internal_overrides: {
      [target.path]: { hidden_units: [64, 16], activation: 'relu', dropout: 0, bias: true },
    } })
  })

  it('rejects invalid widths before applying', () => {
    const onLayer = vi.fn()
    render(<InternalEditor layer={layer} targets={[target]} locked={false} onLayer={onLayer} />)
    fireEvent.click(screen.getByText('Open model internals'))
    fireEvent.click(screen.getByText(target.path))
    fireEvent.change(screen.getByLabelText(`Hidden widths for ${target.path}`), { target: { value: '0, 1.5' } })
    fireEvent.click(screen.getByText('Apply internal edit'))
    expect(screen.getByRole('alert')).toHaveTextContent('1 to 512')
    expect(onLayer).not.toHaveBeenCalled()
  })

  it('allows resetting edits even if validation cannot return a structure', () => {
    const onLayer = vi.fn()
    const modified = { ...layer, internal_overrides: { [target.path]: { hidden_units: [64], activation: 'relu' as const, dropout: 0, bias: true } } }
    render(<InternalEditor layer={modified} locked={false} onLayer={onLayer} />)
    fireEvent.click(screen.getByText('Open model internals · custom variant'))
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
    expect(onLayer).toHaveBeenCalledWith({ ...layer, internal_overrides: {} })
  })

  it('locks internal modifications and explains empty configurations', () => {
    const { rerender } = render(<InternalEditor layer={layer} targets={[target]} locked onLayer={vi.fn()} />)
    fireEvent.click(screen.getByText('Open model internals'))
    fireEvent.click(screen.getByText(target.path))
    expect(screen.getByText('Apply internal edit')).toBeDisabled()
    rerender(<InternalEditor layer={layer} targets={[]} locked={false} onLayer={vi.fn()} />)
    expect(screen.getByText(/no editable Linear/)).toBeInTheDocument()
  })
})
