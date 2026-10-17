"""A page to look at one run of es.walk_forward: how the network learned, and how it did.

Reads the JSON that walk_forward(record=...) writes and makes one self-contained HTML file:

- the learning curves: the population's mean training Sharpe ratio at every generation, one
  line a test year;
- each test year: the network (at the rule's risk) and the rule, their cumulative return on
  QQQ day by day;
- a table of each year.

Hover a chart to read its values. Colours are the data-viz palette's first four categorical
slots (validated for lines in light and dark mode), every line labelled where it ends.

Usage: report.py RUN_JSON OUT_HTML
"""

import json
import sys
from pathlib import Path

run = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network run</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #dddcd5;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --grid: #e6e5df;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --grid: #2b2b29;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --grid: #2b2b29;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
}
body { margin: 0; background: var(--surface-0); }
.viz-root { background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 20px 16px 40px; }
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 18px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 18px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.tile .k { color: var(--text-secondary); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
.card h2 { font-size: 15px; margin: 0 0 2px; }
.card p { color: var(--text-secondary); margin: 0 0 8px; font-size: 13px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
.legend i { display: inline-block; width: 14px; height: 2px; vertical-align: middle; margin-right: 6px; }
.grid4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.panel h3 { font-size: 13px; margin: 0 0 2px; color: var(--text-secondary); font-weight: 600; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-muted); font-size: 11px; }
.tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; white-space: nowrap; z-index: 10; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
<h1>Network against the rule, year by year</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>How it learned</h2>
  <p>The population's mean Sharpe ratio on the training sessions, generation by generation; one line a test year.</p>
  <div class="legend" id="legend1"></div>
  <div id="learn"></div>
</div>
<div class="card">
  <h2>How it did on QQQ</h2>
  <p>Cumulative return in each test year, after costs: the network at the rule's risk, and the rule.</p>
  <div class="legend"><span><i style="background:var(--series-1)"></i>Network</span><span><i style="background:var(--series-2)"></i>Rule</span></div>
  <div class="grid4" id="years"></div>
</div>
<div class="card">
  <h2>Each year</h2>
  <table id="table"></table>
</div>
</div></div>
<div class="tip" id="tip"></div>
<script>
const RUN = __RUN__;
const tip = document.getElementById("tip");
const fmt = (x, d = 2) => (x >= 0 ? "+" : "") + x.toFixed(d);
const years = Object.keys(RUN.details);
const s = RUN.settings;
document.getElementById("sub").textContent =
  `Settings: ${Object.entries(s).map(([k, v]) => k + " " + v).join(", ")}`;
const tiles = [
  ["Network less rule, a day", fmt(RUN.mean_bps) + " bps"],
  ["t-statistic", fmt(RUN.t)],
  ["Days", RUN.days.toLocaleString()],
];
document.getElementById("tiles").innerHTML = tiles.map(([k, v]) =>
  `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

function chart(el, series, opts) {
  // series: [{name, color, ys, xs?, label}]; xs shared by index when absent
  const W = opts.width || 980, H = opts.height || 260, m = {l: 44, r: opts.right || 70, t: 8, b: 26};
  const n = Math.max(...series.map(d => d.ys.length));
  const all = series.flatMap(d => d.ys);
  let lo = Math.min(...all, opts.floor ?? Infinity), hi = Math.max(...all, opts.ceil ?? -Infinity);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const X = i => m.l + (W - m.l - m.r) * (n <= 1 ? 0 : i / (n - 1));
  const Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  const ticks = 4, grid = [];
  for (let k = 0; k <= ticks; k++) {
    const v = lo + (hi - lo) * k / ticks;
    grid.push(`<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>` +
      `<text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${opts.yfmt(v)}</text>`);
  }
  if (lo < 0 && hi > 0) grid.push(`<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--text-muted)" stroke-width="1"/>`);
  const xl = opts.xlabels ? opts.xlabels(n).map(([i, t]) =>
    `<text x="${X(i)}" y="${H - 6}" text-anchor="middle">${t}</text>`).join("") : "";
  const paths = series.map(d => {
    const pts = d.ys.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    const last = d.ys.length - 1;
    const label = opts.labelEnds ? `<text x="${X(last) + 6}" y="${Y(d.ys[last]) + 4}" style="fill:var(--text-secondary)">${d.label || d.name}</text>` : "";
    return `<polyline points="${pts}" fill="none" stroke="${d.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${label}`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.aria}">${grid.join("")}${xl}${paths}` +
    `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-muted)" stroke-width="1" visibility="hidden"/>` +
    `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
  const svg = el.querySelector("svg"), cross = svg.querySelector(".cross"), hit = svg.querySelector("rect");
  hit.addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((x - m.l) / (W - m.l - m.r) * (n - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>${opts.xname(i)}</b><br>` + series.map(d =>
      `<span style="color:${d.color}">&#9644;</span> ${d.name}: ${opts.yfmt(d.ys[i])}`).join("<br>");
    tip.style.display = "block"; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
}

const colors = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
document.getElementById("legend1").innerHTML = years.map((y, k) =>
  `<span><i style="background:${colors[k % 4]}"></i>Trained for ${y}</span>`).join("");
chart(document.getElementById("learn"), years.map((y, k) => ({
  name: "for " + y, label: y, color: colors[k % 4], ys: RUN.details[y].fitness })),
  { aria: "Learning curves", yfmt: v => v.toFixed(2), xname: i => "Generation " + (i + 1),
    labelEnds: true, xlabels: n => [[0, "1"], [n - 1, String(n)]] });

const box = document.getElementById("years");
years.forEach(y => {
  const d = RUN.details[y], cum = rs => { let c = 1; return rs.map(r => (c *= 1 + r) * 100 - 100); };
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h3>${y}</h3><div></div>`;
  box.appendChild(panel);
  chart(panel.querySelector("div"), [
    { name: "Network", color: "var(--series-1)", ys: cum(d.model) },
    { name: "Rule", color: "var(--series-2)", ys: cum(d.rule) },
  ], { width: 480, height: 220, right: 12, aria: "Cumulative return in " + y, yfmt: v => v.toFixed(1) + "%",
       xname: i => d.days[i], xlabels: n => [[0, d.days[0].slice(5)], [n - 1, d.days[n - 1].slice(5)]] });
});

const rows = years.map(y => {
  const r = RUN.years[y];
  return `<tr><td>${y}</td><td>${r.days}</td><td>${fmt(r.mean_bps)}</td><td>${fmt(r.t)}</td>` +
    `<td>${r.model_sharpe.toFixed(2)}</td><td>${r.rule_sharpe.toFixed(2)}</td><td>${r.scale.toFixed(2)}</td></tr>`;
}).join("");
document.getElementById("table").innerHTML =
  "<tr><th>Year</th><th>Days</th><th>Network less rule, bps a day</th><th>t</th><th>Network Sharpe</th><th>Rule Sharpe</th><th>Scale</th></tr>" + rows;
</script>
</body>
</html>
"""

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(PAGE.replace("__RUN__", json.dumps(run)))
print(f"written {out}")
