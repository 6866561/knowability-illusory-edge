"""Render the interactive companion's 3D figures (same data and styling) as high-resolution PNGs for the paper."""
import json, math, pathlib
import numpy as np
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT.parent / "paper" / "figs"
C = dict(bg="#141b24", ink="#e6e9ee", muted="#9aa5b4", line="#263241", accent="#f2a541")
FONT = dict(family="Source Sans 3, Segoe UI, Arial, sans-serif", color=C["ink"], size=22)


def ax3(t):
    return dict(title=dict(text=t, font=dict(color=C["muted"], size=22)), color=C["muted"], gridcolor=C["line"],
                zerolinecolor=C["line"], backgroundcolor=C["bg"], showbackground=True, tickfont=dict(size=16))


def layout(scene, title):
    return dict(paper_bgcolor=C["bg"], font=FONT, margin=dict(l=0, r=0, t=50, b=0), showlegend=False,
                title=dict(text=title, x=0.03, y=0.97, font=dict(size=26, color=C["ink"])), scene=scene)


def save(fig, name):
    # static export ignores 3D axis-title sizes, so axes are named in the paper's caption instead
    fig.update_scenes(xaxis_title_text="", yaxis_title_text="", zaxis_title_text="")
    cam = fig.layout.scene.camera.eye
    fig.update_scenes(camera_eye=dict(x=cam.x * 0.78, y=cam.y * 0.78, z=cam.z * 0.85))
    path = OUT / f"{name}.png"
    fig.write_image(path, width=1000, height=820, scale=2.5)
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (0x14, 0x1b, 0x24))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        pad = 30
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))
        im.crop(box).save(path)
    print("saved", name, im.size if not box else (box[2] - box[0], box[3] - box[1]))


Z = np.load(RES / "fig_lorenz_local.npz")
idx = np.random.default_rng(3).choice(len(Z["X"]), 16000, replace=False)
X = Z["X"][idx]
for h, tag in [("h2", "a"), ("h3", "b")]:
    fig = go.Figure(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
                                 marker=dict(size=1.9, color=Z[h][idx], colorscale="Inferno", cmin=0, cmax=1, opacity=0.92,
                                             colorbar=dict(title=dict(text="certainty", font=dict(color=C["muted"], size=20)),
                                                           tickfont=dict(color=C["muted"], size=16), len=0.6, thickness=18))))
    fig.update_layout(**layout(dict(xaxis=ax3("x"), yaxis=ax3("y"), zaxis=ax3("z"), aspectmode="cube",
                                    camera=dict(eye=dict(x=1.5, y=-1.5, z=0.6))),
                               f"({tag}) Lorenz knowability, h = {h[1:]}"))
    save(fig, f"comp_lorenz_{h}")

law = json.loads((RES / "law.json").read_text())
keys = sorted(law["nn_fits"], key=float)
logn = np.log10(law["nn_fits"][keys[0]]["n"]); hs = [float(k) for k in keys]
G = np.log10(np.clip([law["nn_fits"][k]["gaps"] for k in keys], 1e-3, None))
P = law["nn_law_all"]["0.05"]["points"]
fig = go.Figure([
    go.Surface(x=logn, y=hs, z=G, colorscale="Plasma", reversescale=True, showscale=False, opacity=0.96,
               contours=dict(z=dict(show=True, start=math.log10(0.05), end=math.log10(0.05), size=1, color="#ffffff", width=6))),
    go.Scatter3d(x=[math.log10(p[1]) for p in P], y=[p[0] for p in P], z=[math.log10(0.05)] * len(P), mode="lines+markers",
                 line=dict(color=C["accent"], width=9), marker=dict(size=5, color=C["accent"]))])
fig.update_layout(**layout(dict(xaxis=ax3("log10 samples"), yaxis=ax3("horizon h"), zaxis=ax3("log10 gap"),
                                camera=dict(eye=dict(x=-1.6, y=-1.4, z=0.8))), "(c) Learnability landscape"))
save(fig, "comp_landscape")

nu, nv = 80, 40
u = np.linspace(0, 2 * np.pi, nu + 1); v = np.linspace(0, np.pi, nv + 1)
U, V = np.meshgrid(u, v)
xs, ys, zs = np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V)
pd_ = 0.35
s = 1 - pd_
fig = go.Figure([
    go.Surface(x=xs, y=ys, z=zs, opacity=0.10, showscale=False, colorscale=[[0, "#ffffff"], [1, "#ffffff"]], hoverinfo="skip"),
    go.Surface(x=xs * s, y=ys * s, z=zs * s, surfacecolor=0.5 + ((1 + np.abs(zs)) / 2 - 0.5) * s, cmin=0.5, cmax=1,
               colorscale="Viridis", colorbar=dict(title=dict(text="ceiling", font=dict(color=C["muted"], size=20)),
                                                   tickfont=dict(color=C["muted"], size=16), len=0.6, thickness=18))])
fig.update_layout(**layout(dict(xaxis=ax3("x"), yaxis=ax3("y"), zaxis=ax3("z"), aspectmode="cube",
                                camera=dict(eye=dict(x=1.4, y=1.2, z=0.7))), f"(d) Qubit ceiling, depolarizing p = {pd_}"))
save(fig, "comp_bloch")
