"""Sustained mid-season STANCE-ANGLE changes (same detector as depth/width: K = 15 games, persists over +-30 games).
Threshold 10 deg. Sign: + = more CLOSED, - = more OPEN (Savant angle: negative = open)."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
import events as E

s = pd.read_parquet("stance_adj.parquet").sort_values(["bsy", "game_date"])
s["angle_adj"] = s.angle_adj.fillna(s.angle)
rows = []
for _, g in s.groupby("bsy"):
    r = E.detect(g, "angle", 10.0)
    if r:
        rows.append(r)
A = pd.DataFrame(rows)
A["dir"] = np.where(A.move > 0, "CLOSED", "OPENED")
A.to_csv("angle_events.csv", index=False)
print(f"angle events: {len(A)}")
print(A.groupby(["dir", "year"]).size().unstack(fill_value=0).to_string())
print(f"median |change| {A.persist.abs().median():.1f} deg")
w = pd.read_csv("width_events.csv")
d = pd.read_csv("events_thr4.csv")
print(
    f"(for comparison: width {len(w)}, depth {len(d)}); hitter-seasons with an angle event that also have a width/depth event: "
    f"{A.bsy.isin(set(w.bsy) | set(d.bsy)).sum()}"
)
