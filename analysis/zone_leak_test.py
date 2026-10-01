"""Statcast's per-pitch strike-zone bounds leak whether the batter swung (paper Section 3.3).

Out-of-zone pitches (zone rebuilt from the crossing point and the hitter-season MEDIAN zone, stance_lib.pitches) in the
pre-ABS seasons. Model: P(swing) by gradient boosting, trained on 2024, scored on 2025 (AUC).
  base      plate location, pitch type, pitch speed
  + sz      base + the pitch's own sz_top / sz_bot
  + sz dev  base + the pitch's sz_top / sz_bot minus the hitter-season median (what the hitter's body did on that pitch)
Also: SD of the per-pitch zone top / bottom around the hitter-season median, on takes vs swings."""

import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from stance_lib import pitches

warnings.filterwarnings("ignore")

p = pitches(("2024", "2025"))
p = p[p.out_zone & p.plate_x.notna() & p.plate_z.notna() & p.release_speed.notna() & p.sz_top.notna()].copy()
key = p.batter.astype(str) + p.side + p.year
p["top_dev"] = p.sz_top - p.groupby(key).sz_top.transform("median")
p["bot_dev"] = p.sz_bot - p.groupby(key).sz_bot.transform("median")
p["pt"] = p.pitch_type.astype("category").cat.codes
p["y"] = p.swing.astype(int)
tr, te = p[p.year == "2024"], p[p.year == "2025"]
print(f"out-of-zone pitches: train 2024 {len(tr):,}, test 2025 {len(te):,}; swing rate {te.y.mean():.3f}")
BASE = ["plate_x", "plate_z", "pt", "release_speed"]
for lab, cols in (
    ("base: location + type + speed", BASE),
    ("+ per-pitch sz_top / sz_bot", BASE + ["sz_top", "sz_bot"]),
    ("+ deviation from hitter-season median", BASE + ["top_dev", "bot_dev"]),
):
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, categorical_features=[cols.index("pt")], random_state=0
    )
    m.fit(tr[cols], tr.y)
    print(f"  {lab:40s} test AUC {roc_auc_score(te.y, m.predict_proba(te[cols])[:, 1]):.3f}")
for lab, sel in (("takes", te.y == 0), ("swings", te.y == 1)):
    print(
        f"  SD of zone top / bottom around the hitter-season median on {lab:6s}: "
        f"{12 * te.loc[sel, 'top_dev'].std():.2f} / {12 * te.loc[sel, 'bot_dev'].std():.2f} in"
    )
