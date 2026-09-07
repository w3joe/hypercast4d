import { useMemo, useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, BarChart3, Blocks, ChevronRight, CircleStop, Clock3, ExternalLink, FlaskConical, Layers3, LockKeyhole, X } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import GraphBuilder from './GraphBuilder'
import { ErrorNotice } from './BuilderControls'
import ForecastDiagnostics from './ForecastDiagnostics'
import { api } from './api'
import type { Job } from './types'

type View = 'builder' | 'runs' | 'compare'

function formatMetric(value: number | null | undefined, digits = 4) {
  return value == null || Number.isNaN(value) ? '—' : value.toFixed(digits)
}

function statusClass(state: Job['status']['state']) {
  return `status-pill status-${state.replace('_', '-')}`
}

function RunsView({ jobs, legacyRuns }: { jobs: Job[]; legacyRuns: Record<string, string | number | null>[] }) {
  const queryClient = useQueryClient()
  const [logJob, setLogJob] = useState<string | null>(null)
  const cancelMutation = useMutation({
    mutationFn: api.cancel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const log = useQuery({ queryKey: ['job-log', logJob], queryFn: () => api.log(logJob!), enabled: Boolean(logJob), refetchInterval: 2000 })
  const reproduction = legacyRuns.some((run) => run.candidate != null)
  return (
    <section className="content-view">
      <div className="view-heading"><div><span className="eyebrow">Experiment lifecycle</span><h2>Runs</h2></div><span>{jobs.length} persisted jobs</span></div>
      {!jobs.length && <div className="large-empty panel-surface"><Clock3 size={34} /><h3>No experiments yet</h3><p>Build an architecture and submit a validation run.</p></div>}
      <div className="jobs-list">
        {jobs.map((job) => {
          const percent = job.status.total ? Math.round(job.status.completed / job.status.total * 100) : 0
          const active = ['queued', 'starting', 'running'].includes(job.status.state)
          return <article className="job-card panel-surface" key={job.id}>
            <div className="job-top"><div><span className={statusClass(job.status.state)}>{job.status.state}</span><h3>{job.status.architecture_name}</h3><p>{job.status.phase.replace('_', ' ')} · {job.status.protocol ?? 'chronological-v1'} · {job.status.preset} · {job.status.execution_target ?? 'local'}{job.status.gpu ? ` · ${job.status.gpu}` : ''} · {job.id}</p></div><div className="job-actions">{job.status.modal_dashboard_url && <a className="text-button" href={job.status.modal_dashboard_url} target="_blank" rel="noreferrer">Modal <ExternalLink size={13} /></a>}<button className="text-button" onClick={() => setLogJob(logJob === job.id ? null : job.id)}>View log</button>{active && <button className="danger-button" disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate(job.id)}><CircleStop size={14} />Cancel</button>}</div></div>
            <div className="progress-track"><div style={{ width: `${percent}%` }} /></div>
            <div className="job-meta"><span>{job.status.completed} / {job.status.total || '—'} runs</span><span>{percent}%</span>{job.status.current && <span>w{job.status.current.window}/h{job.status.current.horizon} · seed {job.status.current.seed} · epoch {job.status.current.epoch}/{job.status.current.epochs}</span>}<span>Updated {new Date(job.status.updated_at).toLocaleString()}</span></div>
            {job.status.error && <div className="inline-error">{job.status.error}</div>}
            {job.status.execution_target === 'gcp' && <p className="muted-copy">{job.status.gpu_count ?? 1} × GPU · {active ? 'VM startup / training; logs and results arrive when finished.' : 'GCP VM run'}{job.status.gcp_cleanup && job.status.gcp_cleanup !== 'complete' ? ` · Cleanup needs attention: ${job.status.gcp_cleanup}` : ''}</p>}
            {job.summary.length > 0 && <div className="result-strip"><span>Mean MAE ratio <strong>{formatMetric(job.summary.reduce((sum, row) => sum + row.mae_ratio, 0) / job.summary.length, 3)}</strong></span><span>Mean MSE ratio <strong>{formatMetric(job.summary.reduce((sum, row) => sum + row.mse_ratio, 0) / job.summary.length, 3)}</strong></span><span>Parameters <strong>{Math.round(job.summary[0].parameters).toLocaleString()}</strong></span></div>}
            {logJob === job.id && <pre className="training-log">{log.data || 'Waiting for worker output…'}</pre>}
          </article>
        })}
      </div>
      {legacyRuns.length > 0 && <section className="legacy-runs panel-surface"><div className="panel-title"><Activity size={17} /><span>{reproduction ? 'Released-notebook reproduction results' : 'Main evaluation results'}</span></div><p className="muted-copy">Read-only results from the directory supplied with <code>--results</code>. {reproduction ? 'MAE values are normalized cross-validation scores, matching the paper reproduction runner.' : 'MAE and MSE are in original target units, matching the leakage-safe main runner.'}</p><div className="comparison-table"><table><thead><tr><th>Model</th><th>Window</th><th>Horizon</th><th>{reproduction ? 'Candidate' : 'Seed'}</th><th>{reproduction ? 'CV MAE (scaled)' : 'MAE'}</th><th>{reproduction ? 'MAE std' : 'MSE'}</th><th>Parameters</th></tr></thead><tbody>{legacyRuns.slice(-12).reverse().map((run, index) => <tr key={`${run.model}-${run.seed ?? run.candidate}-${index}`}><td><strong>{run.model}</strong></td><td>{run.window}</td><td>{run.horizon}</td><td>{reproduction ? run.candidate : run.seed}</td><td>{formatMetric(Number(run.mae), 5)}</td><td>{formatMetric(reproduction ? Number(run.mae_std) : Number(run.mse), 5)}</td><td>{Number(run.parameters).toLocaleString()}</td></tr>)}</tbody></table></div></section>}
    </section>
  )
}

function FinalTestDialog({ job, onComplete }: { job: Job; onComplete: () => void }) {
  const [acknowledged, setAcknowledged] = useState(false)
  const mutation = useMutation({ mutationFn: () => api.finalTest(job.id), onSuccess: onComplete })
  return <Dialog.Root><Dialog.Trigger asChild><button className="text-button" disabled={job.status.preset === 'quick'}><FlaskConical size={14} />Final test</button></Dialog.Trigger><Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content"><Dialog.Title>Unlock the held-out test set?</Dialog.Title><Dialog.Description>This candidate can be tested only once. Use this after architecture selection is complete.</Dialog.Description><label className="checkbox-row acknowledgement"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /><span>I understand this consumes the final evaluation for this candidate.</span></label><ErrorNotice error={mutation.error} /><div className="dialog-actions"><Dialog.Close asChild><button className="secondary-button">Cancel</button></Dialog.Close><button className="primary-button" disabled={!acknowledged || mutation.isPending} onClick={() => mutation.mutate()}>Start final test</button></div><Dialog.Close className="dialog-close" aria-label="Close"><X size={17} /></Dialog.Close></Dialog.Content></Dialog.Portal></Dialog.Root>
}

function CompareView({ jobs }: { jobs: Job[] }) {
  const queryClient = useQueryClient()
  const candidates = jobs.filter((job) => job.status.state === 'complete' && job.status.phase === 'validation' && job.summary.length)
  const finalJobs = jobs.filter((job) => job.status.state === 'complete' && job.status.phase === 'final_test' && job.summary.length)
  const [selected, setSelected] = useState<string[]>([])
  const visible = selected.length ? candidates.filter((job) => selected.includes(job.id)) : candidates
  const data = visible.map((job) => ({
    id: job.id,
    name: job.status.architecture_name,
    mae: job.summary.reduce((sum, row) => sum + row.mae_ratio, 0) / job.summary.length,
    mse: job.summary.reduce((sum, row) => sum + row.mse_ratio, 0) / job.summary.length,
    parameters: job.summary.reduce((sum, row) => sum + row.parameters, 0) / job.summary.length,
  }))
  const pareto = new Set(data.filter((candidate) => !data.some((other) => other.id !== candidate.id && other.mae <= candidate.mae && other.mse <= candidate.mse && (other.mae < candidate.mae || other.mse < candidate.mse))).map((item) => item.id))
  return (
    <section className="content-view">
      <div className="view-heading"><div><span className="eyebrow">Validation only</span><h2>Compare architectures</h2></div><span>Below 1.0 beats persistence</span></div>
      {!candidates.length && <div className="large-empty panel-surface"><BarChart3 size={34} /><h3>No completed validation runs</h3><p>Comparison results appear after a Standard or Robust experiment completes.</p></div>}
      {candidates.length > 0 && <>
        <div className="compare-selector panel-surface">{candidates.map((job) => <label className="checkbox-row" key={job.id}><input type="checkbox" checked={selected.includes(job.id)} onChange={(event) => setSelected(event.target.checked ? [...selected, job.id] : selected.filter((id) => id !== job.id))} /><span>{job.status.architecture_name}</span><small>{job.status.preset}</small></label>)}</div>
        <div className="charts-grid">
          <article className="chart-panel panel-surface"><h3>Error relative to persistence</h3><ResponsiveContainer width="100%" height={310}><BarChart data={data}><CartesianGrid stroke="#e0e5de" vertical={false} /><XAxis dataKey="name" stroke="#68736d" tick={{ fontSize: 11 }} /><YAxis stroke="#68736d" /><Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e0e5de' }} /><Legend /><Bar dataKey="mae" name="MAE ratio" fill="#287455" radius={[4, 4, 0, 0]} /><Bar dataKey="mse" name="MSE ratio" fill="#477cb2" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></article>
          <article className="chart-panel panel-surface"><h3>Accuracy–parameter trade-off</h3><ResponsiveContainer width="100%" height={310}><ScatterChart><CartesianGrid stroke="#e0e5de" /><XAxis type="number" dataKey="parameters" name="Parameters" stroke="#68736d" /><YAxis type="number" dataKey="mae" name="MAE ratio" stroke="#68736d" /><Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#ffffff', border: '1px solid #e0e5de' }} /><Scatter data={data} fill="#b58e3d" /></ScatterChart></ResponsiveContainer></article>
        </div>
        <ForecastDiagnostics jobs={visible} />
        <div className="comparison-table panel-surface"><table><thead><tr><th>Architecture</th><th>Preset</th><th>MAE ratio</th><th>MSE ratio</th><th>Parameters</th><th>Selection</th><th /></tr></thead><tbody>{visible.map((job) => { const row = data.find((item) => item.id === job.id)!; return <tr key={job.id}><td><strong>{row.name}</strong></td><td>{job.status.preset}</td><td className={row.mae < 1 ? 'metric-good' : ''}>{formatMetric(row.mae, 3)}</td><td className={row.mse < 1 ? 'metric-good' : ''}>{formatMetric(row.mse, 3)}</td><td>{Math.round(row.parameters).toLocaleString()}</td><td>{pareto.has(job.id) ? <span className="pareto-badge">Pareto best</span> : 'Dominated'}</td><td><FinalTestDialog job={job} onComplete={() => queryClient.invalidateQueries({ queryKey: ['jobs'] })} /></td></tr> })}</tbody></table></div>
      </>}
      {finalJobs.length > 0 && <section className="final-results panel-surface"><div className="panel-title"><LockKeyhole size={17} /><span>Held-out final tests</span></div><p className="muted-copy">These results are separated from architecture selection and cannot be rerun for the same candidate configuration.</p><ForecastDiagnostics jobs={finalJobs} /><div className="comparison-table"><table><thead><tr><th>Architecture</th><th>Cell</th><th>Test MAE</th><th>Test MSE</th><th>MAE ratio</th><th>MSE ratio</th></tr></thead><tbody>{finalJobs.flatMap((job) => job.summary.map((row) => <tr key={`${job.id}-${row.window}-${row.horizon}`}><td><strong>{job.status.architecture_name}</strong></td><td>w{row.window}/h{row.horizon}</td><td>{formatMetric(row.mae_mean, 5)}</td><td>{formatMetric(row.mse_mean, 5)}</td><td className={row.mae_ratio < 1 ? 'metric-good' : ''}>{formatMetric(row.mae_ratio, 3)}</td><td className={row.mse_ratio < 1 ? 'metric-good' : ''}>{formatMetric(row.mse_ratio, 3)}</td></tr>))}</tbody></table></div></section>}
    </section>
  )
}

export default function App() {
  const [view, setView] = useState<View>('builder')
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: api.catalog, staleTime: Infinity })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 1500 })
  const legacyRuns = useQuery({ queryKey: ['legacy-runs'], queryFn: api.legacyRuns, staleTime: 5000 })
  const activeCount = useMemo(() => (jobs.data ?? []).filter((job) => ['queued', 'starting', 'running'].includes(job.status.state)).length, [jobs.data])
  if (catalog.isLoading) return <div className="app-loading"><div className="loading-mark"><Layers3 /></div><p>Loading architecture catalog…</p></div>
  if (catalog.error || !catalog.data) return <div className="app-loading"><ErrorNotice error={catalog.error ?? new Error('Catalog unavailable')} /></div>
  return (
    <div className="app-shell">
      <header className="app-header">
        <button className="brand" onClick={() => setView('builder')}><span className="brand-mark"><Layers3 size={19} /></span><span><strong>HyperCast<span>4D</span></strong><small>Architecture Playground</small></span></button>
        <nav aria-label="Primary navigation">
          <button className={view === 'builder' ? 'active' : ''} onClick={() => setView('builder')}><Blocks size={16} />Builder</button>
          <button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}><Clock3 size={16} />Runs{activeCount > 0 && <span className="nav-count">{activeCount}</span>}</button>
          <button className={view === 'compare' ? 'active' : ''} onClick={() => setView('compare')}><BarChart3 size={16} />Compare</button>
        </nav>
        <div className="local-badge"><span />Local workspace</div>
      </header>
      <div className="app-content">
            <div hidden={view !== 'builder'}><GraphBuilder catalog={catalog.data} /></div>
        {view === 'runs' && <RunsView jobs={jobs.data ?? []} legacyRuns={legacyRuns.data ?? []} />}
        {view === 'compare' && <CompareView jobs={jobs.data ?? []} />}
      </div>
      <footer className="app-footer"><span>Local by default · data goes to Modal or GCP only when selected</span><span><ChevronRight size={13} /> Test metrics remain locked during architecture search</span></footer>
    </div>
  )
}
