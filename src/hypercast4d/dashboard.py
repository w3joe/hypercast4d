"""Serve a dependency-free dashboard for live HyperCast4D results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


NUMERIC_FIELDS = {
    "window": int,
    "horizon": int,
    "seed": int,
    "mae": float,
    "mse": float,
    "parameters": int,
    "train_seconds": float,
    "process_peak_rss_mb": float,
    "epochs_ran": int,
    "best_validation_mse_scaled": float,
    "train_samples": int,
    "validation_samples": int,
    "test_samples": int,
}


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HyperCast4D Live Results</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: #0e1b2d;
      --panel-2: #13233a;
      --line: #263956;
      --text: #edf5ff;
      --muted: #99abc2;
      --accent: #47d7ac;
      --warning: #ffc857;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% -8%, #183b58 0, transparent 34rem),
        radial-gradient(circle at 92% 10%, #173853 0, transparent 28rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    main { width: min(1200px, calc(100% - 32px)); margin: 0 auto; padding: 38px 0 60px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }
    h1 { margin: 0; font-size: clamp(2rem, 5vw, 3.6rem); letter-spacing: -0.055em; }
    h1 span { color: var(--accent); }
    .subtitle { color: var(--muted); margin: 9px 0 0; max-width: 680px; line-height: 1.55; }
    .badge { border: 1px solid var(--line); background: #102139; padding: 9px 13px; border-radius: 999px; white-space: nowrap; }
    .badge.running::before { content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 5px #47d7ac22; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 30px 0 14px; }
    .card, .panel { border: 1px solid var(--line); background: linear-gradient(145deg, #12243aee, #0c1829ee); border-radius: 16px; box-shadow: 0 18px 45px #00000020; }
    .card { padding: 18px; }
    .label { color: var(--muted); font-size: .76rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
    .value { font-size: 1.7rem; font-weight: 750; margin-top: 7px; }
    .progress { height: 8px; border-radius: 9px; background: #07101d; margin-top: 12px; overflow: hidden; }
    .progress > div { height: 100%; width: 0; background: linear-gradient(90deg, #33b5e5, var(--accent)); transition: width .35s ease; }
    .panel { padding: 22px; margin-top: 14px; }
    .panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 18px; margin-bottom: 14px; }
    h2 { font-size: 1.05rem; margin: 0; }
    .note { color: var(--muted); font-size: .82rem; }
    #chart { width: 100%; min-height: 390px; display: block; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 8px; color: var(--muted); font-size: .78rem; }
    .legend span::before { content: ""; width: 9px; height: 9px; display: inline-block; border-radius: 3px; margin-right: 6px; background: var(--color); }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .86rem; }
    th { color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-size: .7rem; text-align: left; }
    th, td { border-bottom: 1px solid #253752aa; padding: 11px 9px; }
    td.number { font-variant-numeric: tabular-nums; }
    .empty { fill: var(--muted); font-size: 20px; }
    footer { color: var(--muted); text-align: center; margin-top: 22px; font-size: .8rem; }
    @media (max-width: 780px) {
      header { display: block; }
      .badge { display: inline-block; margin-top: 18px; }
      .grid { grid-template-columns: repeat(2, 1fr); }
      .panel { padding: 15px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>HyperCast<span>4D</span></h1>
      <p class="subtitle">Live leakage-safe forecasting results. The page reads the experiment's CSV output and refreshes automatically.</p>
    </div>
    <div id="status-badge" class="badge">Waiting for results</div>
  </header>

  <section class="grid">
    <article class="card">
      <div class="label">Progress</div>
      <div id="progress-value" class="value">0 / 0</div>
      <div class="progress"><div id="progress-bar"></div></div>
    </article>
    <article class="card">
      <div class="label">Forecast cells</div>
      <div id="cell-value" class="value">0</div>
    </article>
    <article class="card">
      <div class="label">Models observed</div>
      <div id="model-value" class="value">0</div>
    </article>
    <article class="card">
      <div class="label">Best mean MAE</div>
      <div id="best-value" class="value">—</div>
    </article>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Mean test MAE by model</h2>
      <span class="note">Lower is better · updates every 1.5 seconds</span>
    </div>
    <svg id="chart" viewBox="0 0 1000 420" role="img" aria-label="Live mean absolute error chart"></svg>
    <div id="legend" class="legend"></div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Latest completed runs</h2>
      <span id="updated" class="note">Not updated yet</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Window</th><th>Horizon</th><th>Seed</th><th>MAE</th><th>MSE</th><th>Parameters</th><th>Seconds</th></tr></thead>
        <tbody id="runs-body"></tbody>
      </table>
    </div>
  </section>
  <footer>Served locally by HyperCast4D · no experiment data leaves this machine</footer>
</main>
<script>
const order = ['persistence','linear','cnn','lstm','hyper_quaternion','hyper_coquaternion','hyper_cl11'];
const colors = {
  persistence:'#4cc9f0', linear:'#f7b267', cnn:'#ff6b6b', lstm:'#c77dff',
  hyper_quaternion:'#47d7ac', hyper_coquaternion:'#80ed99', hyper_cl11:'#ffd166'
};
const svgNS = 'http://www.w3.org/2000/svg';
function el(name, attrs={}, text='') {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key, value));
  if (text) node.textContent = text;
  return node;
}
function mean(values) { return values.reduce((a,b)=>a+b,0) / values.length; }
function summarize(rows) {
  const groups = new Map();
  rows.forEach(row => {
    const key = `${row.window}/${row.horizon}/${row.model}`;
    if (!groups.has(key)) groups.set(key, {window:row.window,horizon:row.horizon,model:row.model,values:[]});
    groups.get(key).values.push(row.mae);
  });
  return [...groups.values()].map(group => ({...group, mae:mean(group.values)}));
}
function drawChart(rows) {
  const svg = document.getElementById('chart');
  svg.replaceChildren();
  const summary = summarize(rows);
  if (!summary.length) {
    svg.appendChild(el('text',{x:500,y:205,'text-anchor':'middle',class:'empty'},'Run an experiment to populate this graph'));
    return;
  }
  const cells = [...new Set(summary.map(r=>`${r.window}/${r.horizon}`))].sort((a,b)=>Number(a.split('/')[0])-Number(b.split('/')[0]));
  const models = order.filter(model => summary.some(row=>row.model===model));
  const maxValue = Math.max(...summary.map(row=>row.mae)) * 1.15;
  const margin = {left:72,right:22,top:20,bottom:70};
  const width = 1000-margin.left-margin.right, height=420-margin.top-margin.bottom;
  for (let tick=0; tick<=5; tick++) {
    const y=margin.top+height-(tick/5)*height;
    svg.appendChild(el('line',{x1:margin.left,y1:y,x2:980,y2:y,stroke:'#2a3e5a','stroke-width':1}));
    svg.appendChild(el('text',{x:margin.left-12,y:y+4,'text-anchor':'end',fill:'#99abc2','font-size':12},(maxValue*tick/5).toFixed(2)));
  }
  const groupWidth=width/cells.length;
  const barWidth=Math.min(24,(groupWidth-24)/Math.max(models.length,1));
  cells.forEach((cell,cellIndex)=>{
    const center=margin.left+groupWidth*(cellIndex+.5);
    models.forEach((model,modelIndex)=>{
      const row=summary.find(item=>`${item.window}/${item.horizon}`===cell&&item.model===model);
      if (!row) return;
      const h=(row.mae/maxValue)*height;
      const x=center-(models.length*barWidth)/2+modelIndex*barWidth;
      const rect=el('rect',{x:x+1,y:margin.top+height-h,width:Math.max(barWidth-2,2),height:h,rx:3,fill:colors[model]||'#fff'});
      rect.appendChild(el('title',{},`${model} · w${row.window}/h${row.horizon} · mean MAE ${row.mae.toFixed(5)}`));
      svg.appendChild(rect);
    });
    svg.appendChild(el('text',{x:center,y:margin.top+height+27,'text-anchor':'middle',fill:'#edf5ff','font-size':13},`w${cell.split('/')[0]} / h${cell.split('/')[1]}`));
  });
  const legend=document.getElementById('legend'); legend.replaceChildren();
  models.forEach(model=>{const item=document.createElement('span');item.style.setProperty('--color',colors[model]);item.textContent=model;legend.appendChild(item);});
}
function renderTable(rows) {
  const body=document.getElementById('runs-body'); body.replaceChildren();
  rows.slice(-12).reverse().forEach(row=>{
    const tr=document.createElement('tr');
    [row.model,row.window,row.horizon,row.seed,row.mae.toFixed(5),row.mse.toFixed(5),row.parameters,row.train_seconds.toFixed(3)].forEach((value,index)=>{
      const td=document.createElement('td'); td.textContent=value; if(index>0)td.className='number'; tr.appendChild(td);
    }); body.appendChild(tr);
  });
}
async function poll() {
  try {
    const [statusResponse,runsResponse]=await Promise.all([fetch('/api/status',{cache:'no-store'}),fetch('/api/runs',{cache:'no-store'})]);
    const status=await statusResponse.json(), rows=await runsResponse.json();
    const completed=status.completed ?? rows.length, total=status.total ?? rows.length;
    document.getElementById('progress-value').textContent=`${completed} / ${total}`;
    document.getElementById('progress-bar').style.width=`${total?Math.min(100,completed/total*100):0}%`;
    const cells=new Set(rows.map(row=>`${row.window}/${row.horizon}`));
    const models=new Set(rows.map(row=>row.model));
    document.getElementById('cell-value').textContent=cells.size;
    document.getElementById('model-value').textContent=models.size;
    const summary=summarize(rows), best=summary.length?summary.reduce((a,b)=>a.mae<b.mae?a:b):null;
    document.getElementById('best-value').textContent=best?best.mae.toFixed(4):'—';
    const badge=document.getElementById('status-badge');
    badge.textContent=status.state==='running'?'Experiment running':status.state==='complete'?'Experiment complete':'Waiting for experiment';
    badge.className=`badge ${status.state||''}`;
    document.getElementById('updated').textContent=`Updated ${new Date().toLocaleTimeString()}`;
    drawChart(rows); renderTable(rows);
  } catch (error) {
    document.getElementById('status-badge').textContent='Waiting for result files';
  }
}
poll(); setInterval(poll,1500);
</script>
</body>
</html>
"""


def load_runs(results_directory: Path) -> list[dict[str, object]]:
    """Load live CSV rows and coerce known numeric fields for JSON clients."""
    path = results_directory / "runs.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field, converter in NUMERIC_FIELDS.items():
                value = raw.get(field, "")
                if value in ("", None):
                    row[field] = None
                    continue
                try:
                    converted = converter(value)
                    row[field] = (
                        converted
                        if not isinstance(converted, float) or math.isfinite(converted)
                        else None
                    )
                except (TypeError, ValueError):
                    row[field] = None
            rows.append(row)
    return rows


def load_status(results_directory: Path, row_count: int) -> dict[str, object]:
    """Load runner status, with a useful fallback for older completed outputs."""
    path = results_directory / "status.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if row_count:
        return {"state": "complete", "completed": row_count, "total": row_count}
    return {"state": "waiting", "completed": 0, "total": 0}


def handler_for(results_directory: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: object) -> None:
            self._send(
                json.dumps(value, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            rows = load_runs(results_directory)
            if path == "/api/runs":
                self._json(rows)
                return
            if path == "/api/status":
                self._json(load_status(results_directory, len(rows)))
                return
            if path == "/health":
                self._json({"ok": True})
                return
            self._json({"error": "not found"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), handler_for(args.results))
    url = f"http://{args.host}:{args.port}"
    print(f"HyperCast4D dashboard: {url}")
    print(f"Watching results in: {args.results.resolve()}")
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
