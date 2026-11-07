"""Build the interactive companion page (single HTML file with embedded result data)."""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT.parent / "artifact" / "knowability_atlas.html"
OUT.parent.mkdir(exist_ok=True)

Z = np.load(RES / "fig_lorenz_local.npz")
rng = np.random.default_rng(3)
idx = rng.choice(len(Z["X"]), 9000, replace=False)
lor = {"x": np.round(Z["X"][idx, 0], 2).tolist(), "y": np.round(Z["X"][idx, 1], 2).tolist(), "z": np.round(Z["X"][idx, 2], 2).tolist()}
for h in ["h0.5", "h1", "h2", "h3"]:
    lor[h] = np.round(Z[h][idx], 3).tolist()

law = json.loads((RES / "law.json").read_text())
keys = sorted(law["nn_fits"], key=float)
land = {"h": [float(k) for k in keys], "logn": np.log10(law["nn_fits"][keys[0]]["n"]).round(3).tolist(),
        "loggap": [np.log10(np.clip(law["nn_fits"][k]["gaps"], 1e-3, None)).round(3).tolist() for k in keys],
        "nstar": law["nn_law_all"]["0.05"]["points"], "gamma": law["nn_law_all"]["0.05"]["gamma"],
        "r2": law["nn_law_all"]["0.05"]["r2"]}
atlas = [{"label": r["short"], "shares": [round(s, 4) for s in r["shares"]], "LSS": round(r["LSS"], 4),
          "acc": round(r["acc"], 3), "acc0": round(r["acc0"], 3)} for r in json.loads((RES / "atlas.json").read_text())["rows"]]
data = json.dumps({"lorenz": lor, "land": land, "atlas": atlas}, separators=(",", ":"))

html = r"""<title>Knowability Atlas</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>
/* Layout: a single dark observing-room column; each exhibit is a figure with a caption strip and its own controls. */
:root{
  --bg:#0d1117; --panel:#141b24; --ink:#e6e9ee; --muted:#9aa5b4; --line:#263241; --accent:#f2a541; --cool:#5fb3d9;
  --display:"Newsreader", Georgia, "Times New Roman", serif;
  --body:"Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, Consolas, monospace;
  color-scheme:dark;
}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding-block:40px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:44px}
header{display:grid;gap:14px;max-width:70ch}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font-family:var(--display);font-weight:600;font-size:clamp(34px,5vw,54px);line-height:1.05;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-weight:600;font-size:26px;margin:0;text-wrap:balance}
p{margin:0;color:var(--ink)}
.lede{font-size:18px;color:var(--muted)}
figure{margin:0;display:grid;gap:12px}
.head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:10px 24px}
.plot{width:100%;height:520px;background:var(--panel);border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--muted);font-size:15px;max-width:72ch}
.ctl{display:flex;align-items:center;gap:12px;font-family:var(--mono);font-size:13px;color:var(--muted)}
.ctl input[type=range]{accent-color:var(--accent);width:180px}
.ctl output{color:var(--ink);min-width:6ch}
.stat{font-family:var(--mono);font-size:13px;color:var(--cool)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:28px}
.grid2 > *{min-width:0}
footer{color:var(--muted);font-size:14px;border-top:1px solid var(--line);padding-top:18px;max-width:80ch}
input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
@media (max-width:600px){.plot{height:400px}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Companion to “How Much of the Future Is Knowable?”</div>
  <h1>How much of the future can be known?</h1>
  <p class="lede">Interactive versions of the paper's main models. Every point comes from the study's own runs: GPU-integrated Lorenz ensembles, 1-NN learners trained on up to a million samples, and walk-forward forecasts in eleven settings.</p>
</header>

<figure>
  <div class="head"><h2>Where chaos keeps its secrets</h2>
    <label class="ctl" for="hsel">horizon h <input id="hsel" type="range" min="0" max="3" step="1" value="1"> <output id="hout">1</output></label></div>
  <div id="lorenz" class="plot" role="img" aria-label="3D Lorenz attractor coloured by Bayes certainty"></div>
  <figcaption>9,000 points on the Lorenz attractor, coloured by the Bayes certainty 2|η − ½| that the sign of x after h time units can be called from a noisy observation (σ = 0.1, 256-particle ensembles). Uncertainty is not spread evenly: it lives on thin filaments that multiply as the horizon grows. Drag to rotate.</figcaption>
</figure>

<div class="grid2">
<figure>
  <div class="head"><h2>The learnability landscape</h2><span class="stat" id="lawstat"></span></div>
  <div id="land" class="plot" role="img" aria-label="Surface of log learnability gap over sample size and horizon"></div>
  <figcaption>Gap between the physical ceiling and what a 1-NN learner reaches, over training samples and horizon. The orange path marks where the gap falls to 0.05: the samples needed grow exponentially with the horizon, close to the rate λD₂ set by the Lyapunov exponent and the attractor's dimension.</figcaption>
</figure>
<figure>
  <div class="head"><h2>A qubit's ceiling</h2>
    <label class="ctl" for="lam">depolarizing λ <input id="lam" type="range" min="0" max="0.9" step="0.05" value="0"> <output id="lamout">0.00</output></label></div>
  <div id="bloch" class="plot" role="img" aria-label="Bloch sphere coloured by prediction ceiling"></div>
  <figcaption>Each pure state's best possible accuracy for a Z measurement, (1 + |z|)/2. Noise shrinks the sphere and the ceiling with it. For a uniformly random rotation the knowable fraction is exactly 1/ln 2 − 1 ≈ 44.3%, even with perfect knowledge of the state.</figcaption>
  <span class="stat" id="qstat"></span>
</figure>
</div>

<figure>
  <div class="head"><h2>The predictability atlas</h2></div>
  <div id="atlas" class="plot" style="height:560px" role="img" aria-label="Stacked bars of explained entropy share per domain"></div>
  <figcaption>Share of each outcome's entropy explained out of sample by entity history, domain features and public benchmarks (betting odds, weather models). The rest is unexplained. Raw accuracy and knowability disagree often: House seats are called correctly 90% of the time, yet the coloured bar is short.</figcaption>
</figure>

<footer>Data: Meteostat, Open-Meteo, football-data.co.uk, MIT Election Data and Science Lab, Yahoo Finance, USGS ComCat, SEC EDGAR and IBM device noise models. All numbers are generated by the study's released code; nothing here is hand-entered.</footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<script>
const D = __DATA__;
const C = {bg:"#141b24", ink:"#e6e9ee", muted:"#9aa5b4", line:"#263241", accent:"#f2a541", cool:"#5fb3d9"};
const font = {family:"Source Sans 3, Segoe UI, sans-serif", color:C.ink, size:13};
const axis3 = (t) => ({title:{text:t,font:{color:C.muted}}, color:C.muted, gridcolor:C.line, zerolinecolor:C.line, backgroundcolor:C.bg, showbackground:true});
const cfg = {displaylogo:false, responsive:true, modeBarButtonsToRemove:["toImage"]};
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Lorenz
const H = ["h0.5","h1","h2","h3"], HL = ["0.5","1","2","3"];
function lorenzTrace(k){ return {type:"scatter3d", mode:"markers", x:D.lorenz.x, y:D.lorenz.y, z:D.lorenz.z,
  marker:{size:1.8, color:D.lorenz[H[k]], colorscale:"Inferno", cmin:0, cmax:1, opacity:0.9,
  colorbar:{title:{text:"certainty",font:{color:C.muted}}, tickfont:{color:C.muted}, len:0.6, thickness:12}},
  hovertemplate:"certainty %{marker.color:.2f}<extra></extra>"}; }
Plotly.newPlot("lorenz", [lorenzTrace(1)], {paper_bgcolor:C.bg, font, margin:{l:0,r:0,t:10,b:0},
  scene:{xaxis:axis3("x"), yaxis:axis3("y"), zaxis:axis3("z"), camera:{eye:{x:1.5,y:-1.5,z:0.6}}, aspectmode:"cube"}}, cfg);
const hsel = document.getElementById("hsel"), hout = document.getElementById("hout");
hsel.addEventListener("input", () => { const k = +hsel.value; hout.value = HL[k];
  Plotly.restyle("lorenz", {"marker.color":[D.lorenz[H[k]]]}); });

// Landscape
const L = D.land;
const fitA = Math.log(L.nstar[0][1]) - L.gamma * L.nstar[0][0];
Plotly.newPlot("land", [
  {type:"surface", x:L.logn, y:L.h, z:L.loggap, colorscale:"Plasma", reversescale:true, showscale:false, opacity:0.95,
   contours:{z:{show:true, start:Math.log10(0.05), end:Math.log10(0.05), size:1, color:"#ffffff", width:4}},
   hovertemplate:"log10 n %{x:.1f}<br>h %{y}<br>log10 gap %{z:.2f}<extra></extra>"},
  {type:"scatter3d", mode:"lines+markers", x:L.nstar.map(p=>Math.log10(p[1])), y:L.nstar.map(p=>p[0]), z:L.nstar.map(()=>Math.log10(0.05)),
   line:{color:C.accent, width:6}, marker:{size:3, color:C.accent}, name:"n*(h)", hovertemplate:"h %{y}: n* ≈ 10^%{x:.1f}<extra></extra>"}
], {paper_bgcolor:C.bg, font, margin:{l:0,r:0,t:10,b:0}, showlegend:false,
  scene:{xaxis:axis3("log10 samples"), yaxis:axis3("horizon h"), zaxis:axis3("log10 gap"), camera:{eye:{x:-1.6,y:-1.4,z:0.8}}}}, cfg);
document.getElementById("lawstat").textContent = `n*(h) ∝ e^{${L.gamma.toFixed(2)} h},  R² = ${L.r2.toFixed(3)}`;

// Bloch sphere
const nu = 60, nv = 30, xs=[], ys=[], zs=[], cs=[];
for (let i=0;i<=nv;i++){ const v=Math.PI*i/nv; const rx=[],ry=[],rz=[],rc=[];
  for (let j=0;j<=nu;j++){ const u=2*Math.PI*j/nu; rx.push(Math.cos(u)*Math.sin(v)); ry.push(Math.sin(u)*Math.sin(v)); rz.push(Math.cos(v)); rc.push((1+Math.abs(Math.cos(v)))/2); }
  xs.push(rx); ys.push(ry); zs.push(rz); cs.push(rc); }
function bloch(l){ const s=1-l; return {type:"surface", x:xs.map(r=>r.map(a=>a*s)), y:ys.map(r=>r.map(a=>a*s)), z:zs.map(r=>r.map(a=>a*s)),
  surfacecolor:cs.map(r=>r.map(a=>0.5+(a-0.5)*s)), cmin:0.5, cmax:1, colorscale:"Viridis",
  colorbar:{title:{text:"ceiling",font:{color:C.muted}}, tickfont:{color:C.muted}, len:0.6, thickness:12},
  hovertemplate:"ceiling %{surfacecolor:.3f}<extra></extra>"}; }
const ghost = {type:"surface", x:xs, y:ys, z:zs, opacity:0.08, showscale:false, colorscale:[[0,"#ffffff"],[1,"#ffffff"]], hoverinfo:"skip"};
Plotly.newPlot("bloch", [ghost, bloch(0)], {paper_bgcolor:C.bg, font, margin:{l:0,r:0,t:10,b:0},
  scene:{xaxis:axis3("x"), yaxis:axis3("y"), zaxis:axis3("z"), aspectmode:"cube", camera:{eye:{x:1.4,y:1.2,z:0.7}}}}, cfg);
const lam = document.getElementById("lam"), lamout = document.getElementById("lamout"), qstat = document.getElementById("qstat");
function qtext(l){ qstat.textContent = `average ceiling for a random rotation: ½ + (1 − λ)/π = ${(0.5+(1-l)/Math.PI).toFixed(4)}`; }
qtext(0);
lam.addEventListener("input", () => { const l=+lam.value; lamout.value=l.toFixed(2); Plotly.react("bloch", [ghost, bloch(l)], document.getElementById("bloch").layout); qtext(l); });

// Atlas
const A = D.atlas.slice().reverse();
const names = ["entity history","features","benchmark","not explained"], cols = ["#5fb3d9","#f2a541","#d96f5f","#2a3644"];
Plotly.newPlot("atlas", names.map((n,k)=>({type:"bar", orientation:"h", name:n, y:A.map(r=>r.label), x:A.map(r=>r.shares[k]),
  marker:{color:cols[k]}, hovertemplate:"%{y}: %{x:.3f} of H(Y)<extra>"+n+"</extra>"})),
  {barmode:"stack", paper_bgcolor:C.bg, plot_bgcolor:C.bg, font, margin:{l:90,r:20,t:10,b:50},
   legend:{orientation:"h", y:-0.12, font:{color:C.muted}},
   xaxis:{range:[0,1], title:{text:"share of outcome entropy", font:{color:C.muted}}, gridcolor:C.line, color:C.muted},
   yaxis:{color:C.ink}}, cfg);
</script>
"""
OUT.write_text(html.replace("__DATA__", data), encoding="utf-8")
print(OUT, round(OUT.stat().st_size / 1024), "KB")
