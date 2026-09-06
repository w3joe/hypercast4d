import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import MethodCollection from './MethodCollection'
import type { Catalog } from './types'

const catalog = { method_collection: {
  sources: { survey: { title: 'Survey', venue: 'FCS', url: 'https://example.com/paper.pdf', pages: '13' } },
  methods: [
    { id: 'patchtst', name: 'PatchTST', family: 'Transformer', kind: 'neural', status: 'adaptation', preset_id: 'research-patchtst', sources: [{ source_id: 'survey', page: 13 }], notes: 'Runnable adaptation.' },
    { id: 'deformtime', name: 'DeformTime', family: 'Deformable attention', kind: 'neural', status: 'reference', preset_id: null, sources: [{ source_id: 'survey', page: 13 }], notes: 'Reference only.' },
  ],
} } as Catalog

describe('method collection', () => {
  it('filters references and loads only runnable methods', () => {
    const onLoad = vi.fn()
    render(<MethodCollection catalog={catalog} onLoad={onLoad} />)
    fireEvent.click(screen.getByText('Method collection'))
    expect(screen.queryByRole('button', { name: 'Load DeformTime' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Search methods'), { target: { value: 'deform' } })
    expect(screen.getByText('1 matching methods')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'PatchTST' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Search methods'), { target: { value: '' } })
    fireEvent.click(screen.getByLabelText('Runnable only'))
    expect(screen.queryByRole('heading', { name: 'DeformTime' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Load PatchTST' }))
    expect(onLoad).toHaveBeenCalledWith('research-patchtst')
    expect(screen.getByRole('link', { name: 'FCS · p. 13' })).toHaveAttribute('href', 'https://example.com/paper.pdf#page=13')
  })
})
