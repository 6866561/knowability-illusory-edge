import json, sys, pathlib
import pandas as pd, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kn import core
R = pathlib.Path(__file__).resolve().parents[1]
p = pd.read_parquet(R / "results/d9_quakes_preds.parquet")
f = pd.read_parquet(R.parent / "data/d9_quakes.parquet")[["t", "entity", "x_c30"]]
d = p.merge(f, on=["t", "entity"], how="left"); assert d.x_c30.notna().all() and len(d) == len(p)
out = {}
for name, g in (("active", d[d.x_c30 > 0]), ("quiet", d[d.x_c30 == 0])):
    o = {"n": int(len(g)), "base_rate": float(g.y.mean())}
    for m in ("p_M0", "p_M2", "p_M4"):
        o[m] = {"acc": float(core.accuracy(g.y, g[m])), "logloss": float(core.log_loss(g.y, g[m]))}
    for m in ("p_M2", "p_M4"):
        o[m]["LSS_vs_M0"] = float(core.skill(g.y, g[m], g["p_M0"])["LSS"])
    out[name] = o
(R / "results/d9_quakes_strata.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
