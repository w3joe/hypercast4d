import { useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Job } from './types'
import './forecast-diagnostics.css'

type Row = Record<string, string | number | null>
const colors = ['#287455', '#477cb2', '#a25837', '#8053a0', '#9b7c22', '#317f8b']
const score = (rows: Row[], key: string): number | null => {
  const values = rows.map(row => row[key]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  return values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : null
}
const format = (value: number | null, digits = 3) => value === null ? 'N/A' : value.toFixed(digits)
const protocolKey = (job: Job) => JSON.stringify([
  job.request.evaluation.protocol, job.request.evaluation.data_path, job.request.evaluation.target_column,
  job.status.phase === 'final_test' ? 'final-test' : job.request.evaluation.preset === 'robust' ? 'robust-folds' : 'standard-fold', job.status.phase,
])

export function diagnosticCells(jobs: Job[]) {
  const cells = new Map<string, { key: string; label: string; window: number; horizon: number; protocol: string }>()
  for (const job of jobs) for (const row of job.per_lead) {
    const window = Number(row.window), horizon = Number(row.horizon), protocol = protocolKey(job)
    const key = JSON.stringify([protocol, window, horizon])
    cells.set(key, { key, window, horizon, protocol,
      label: `${job.request.evaluation.target_column} · window ${window} / horizon ${horizon} · ${job.status.phase === 'final_test' ? 'Held-out test' : job.request.evaluation.preset === 'robust' ? 'Robust folds' : 'Standard split'} · ${job.request.evaluation.data_path}` })
  }
  return [...cells.values()].sort((a, b) => a.window - b.window || a.horizon - b.horizon)
}

const metrics: { key: string; title: string; caption: string; baseline?: string; baselineName?: string; reference?: number; domain: [number | 'auto', number | 'auto']; percent?: boolean }[] = [
  { key: 'mae', title: 'Absolute forecast error', caption: 'Mean absolute error in target units. Lower is better.', baseline: 'persistence_mae', baselineName: 'Persistence', reference: undefined, domain: [0, 'auto'] },
  { key: 'directional_accuracy', title: 'Direction accuracy', caption: 'Correct up / flat / down calls from the forecast origin. Dashed lines show the training-majority direction baseline.', baseline: 'direction_baseline_accuracy', baselineName: 'Direction baseline', reference: undefined, domain: [0, 100], percent: true },
  { key: 'return_correlation', title: 'Return correlation', caption: 'Predicted versus actual changes from the forecast origin. Higher is better; constant or near-constant returns are N/A.', reference: 0, domain: [-1, 1] },
  { key: 'bias', title: 'Forecast bias', caption: 'Mean predicted minus actual price. Above zero overpredicts; below zero underpredicts.', reference: 0, domain: ['auto', 'auto'] },
  { key: 'p95_abs_error', title: 'Large forecast errors (95th percentile)', caption: '95% of absolute errors fall at or below this value within each run. Lower is better.', reference: undefined, domain: [0, 'auto'] },
  { key: 'large_move_mae_ratio', title: 'Performance during large moves', caption: 'MAE / persistence MAE when the actual move exceeds the training 90th percentile for this lead. Below 1 beats persistence.', reference: 1, domain: [0, 'auto'] },
]

export default function ForecastDiagnostics({ jobs }: { jobs: Job[] }) {
  const cells = diagnosticCells(jobs)
  const [selected, setSelected] = useState('')
  const cell = cells.find(item => item.key === selected) ?? cells[0]
  const active = cell ? jobs.filter(job => protocolKey(job) === cell.protocol && job.per_lead.some(row => Number(row.window) === cell.window && Number(row.horizon) === cell.horizon)) : []
  const rowsFor = (job: Job, lead?: number) => job.per_lead.filter(row => Number(row.window) === cell?.window && Number(row.horizon) === cell?.horizon && (lead === undefined || Number(row.lead) === lead))
  const hasDiagnostics = active.some(job => rowsFor(job).some(row => 'bias' in row))
  const [selectedLead, setSelectedLead] = useState(1)
  const lead = Math.min(selectedLead, cell?.horizon ?? 1)
  if (!cell) return null
  return <section className="forecast-diagnostics" aria-label="Forecast diagnostics">
    <div className="diagnostic-heading"><div><span className="eyebrow">Forecast behaviour</span><h3>What the models get right</h3></div>
      <label>Evaluation cell<select aria-label="Diagnostic evaluation cell" value={cell.key} onChange={event => setSelected(event.target.value)}>{cells.map(item => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
    </div>
    <p className="muted-copy">Each point averages defined scores across seeds and folds for this cell. Lines keep forecast leads separate. N/A values are omitted, never replaced with zero. Quick runs are smoke tests.</p>
    {!hasDiagnostics && <p className="diagnostic-notice" role="status">These runs predate forecast diagnostics. Run validation again to populate the new charts and save individual forecasts.</p>}
    <div className="charts-grid">{metrics.filter(metric => hasDiagnostics || metric.key === 'mae').map(metric => {
      const chartData = Array.from({ length: cell.horizon }, (_, j) => {
        const point: Record<string, number | null> = { lead: j + 1 }
        active.forEach((job, index) => {
          const value = score(rowsFor(job, j + 1), metric.key)
          point[`model${index}`] = value === null ? null : value * (metric.percent ? 100 : 1)
          if (metric.baseline) {
            const baseline = score(rowsFor(job, j + 1), metric.baseline)
            point[`baseline${index}`] = baseline === null ? null : baseline * (metric.percent ? 100 : 1)
          }
        })
        return point
      })
      return <article className="chart-panel panel-surface" key={metric.key}>
        <h3>{metric.title}</h3><p className="diagnostic-caption">{metric.caption}</p>
        <ResponsiveContainer width="100%" height={285}><LineChart data={chartData} margin={{ top: 12, right: 18, bottom: 15, left: 6 }}>
          <CartesianGrid stroke="#e0e5de" vertical={false} />
          <XAxis dataKey="lead" type="number" domain={[1, Math.max(2, cell.horizon)]} allowDecimals={false} ticks={cell.horizon === 1 ? [1] : undefined} label={{ value: 'Forecast lead (trading observations)', position: 'insideBottom', offset: -10 }} />
          <YAxis domain={metric.domain} width={60} tickFormatter={value => `${Number(value).toFixed(metric.percent ? 0 : 2)}${metric.percent ? '%' : ''}`} />
          <Tooltip formatter={value => typeof value === 'number' ? `${value.toFixed(3)}${metric.percent ? '%' : ''}` : 'N/A'} labelFormatter={value => `Lead ${value}`} />
          <Legend wrapperStyle={{ paddingTop: 15 }} />
          {metric.reference !== undefined && <ReferenceLine ifOverflow="extendDomain" y={metric.reference} stroke="#778277" strokeDasharray="4 4" />}
          {active.flatMap((job, index) => [
            <Line key={job.id} dataKey={`model${index}`} name={`${job.status.architecture_name} · ${job.id.slice(-8)}`} stroke={colors[index % colors.length]} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} isAnimationActive={false} />,
            ...(metric.baseline ? [<Line key={`${job.id}-baseline`} dataKey={`baseline${index}`} name={`${metric.baselineName} · ${job.id.slice(-8)}`} stroke={colors[index % colors.length]} strokeDasharray="5 4" dot={false} connectNulls={false} isAnimationActive={false} />] : []),
          ])}
        </LineChart></ResponsiveContainer>
      </article>
    })}</div>
    {hasDiagnostics && <>
      <div className="diagnostic-heading"><h3>Scores and sample sizes</h3><label>Forecast lead<select aria-label="Diagnostic table lead" value={lead} onChange={event => setSelectedLead(Number(event.target.value))}>{Array.from({ length: cell.horizon }, (_, j) => <option key={j} value={j + 1}>{j + 1}</option>)}</select></label></div>
      <p className="diagnostic-caption">Counts are averages per seed/fold, not independent observations. Large-move MAE uses target units. Thresholds are absolute returns; 0.02 means 2%. Flat direction means a return within ±0.0001%.</p>
      <div className="comparison-table panel-surface"><table><thead><tr><th>Architecture</th><th>Direction</th><th>Baseline</th><th>Return r</th><th>Bias</th><th>P95 error</th><th>Large-move MAE</th><th>Large-move ratio</th><th>Large moves / samples</th><th>Threshold range</th><th>Forecasts</th></tr></thead><tbody>{active.map(job => {
        const rows = rowsFor(job, lead)
        const percent = (key: string) => { const value = score(rows, key); return value === null ? 'N/A' : `${(100 * value).toFixed(1)}%` }
        const thresholds = rows.map(row => row.large_move_threshold).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
        return <tr key={job.id}><td><strong>{job.status.architecture_name}</strong><small className="diagnostic-run-id">{job.id.slice(-8)} · {rows.length} runs</small></td><td>{percent('directional_accuracy')}</td><td>{percent('direction_baseline_accuracy')}</td><td>{format(score(rows, 'return_correlation'))}</td><td>{format(score(rows, 'bias'))}</td><td>{format(score(rows, 'p95_abs_error'))}</td><td>{format(score(rows, 'large_move_mae'))}</td><td>{format(score(rows, 'large_move_mae_ratio'))}</td><td>{format(score(rows, 'large_move_count'), 1)} / {format(score(rows, 'sample_count'), 1)}</td><td>{thresholds.length ? `${Math.min(...thresholds).toFixed(4)}–${Math.max(...thresholds).toFixed(4)}` : 'N/A'}</td><td>{rows.some(row => 'bias' in row) ? <a href={`/api/v1/jobs/${encodeURIComponent(job.id)}/predictions.csv`} download>Download CSV</a> : 'N/A'}</td></tr>
      })}</tbody></table></div>
    </>}
  </section>
}
