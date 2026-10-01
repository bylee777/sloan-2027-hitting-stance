"""Bat speed -> contact and power, using INTENDED changes (after Powers & Yurko 2025). Hitters deliberately slow down with two
strikes, and by different amounts. For each hitter-season: two-strike minus 0-1-strike difference in bat speed, whiff rate,
exit velocity and collision efficiency q, each on the SAME pitch mix (pitch class x attack zone cells, weights = the hitter's
own 0-1-strike mix). Across hitters, slope of each difference on the bat-speed difference = effect per mph of a deliberate
change (the common two-strike effect sits in the intercept). Compared with the naive swing-level slopes."""

import numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

p = pitches()
for c in ("bat_speed", "launch_speed", "release_speed", "plate_x", "plate_z", "sz_top", "sz_bot"):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
key = p.batter.astype(str) + p.side + p.year
top = p.groupby(key).sz_top.transform("median")
bot = p.groupby(key).sz_bot.transform("median")
d = np.maximum((p.plate_x.abs() - 17 / 24) * 12, np.maximum(bot - p.plate_z, p.plate_z - top) * 12)
p["azone"] = np.select([d <= -2.9, d <= 2.9, d <= 8.7], ["heart", "shadow", "chase"], "waste")
p["cls"] = np.select(
    [p.pitch_type.isin(["FF", "SI", "FC"]), p.pitch_type.isin(["SL", "ST", "CU", "KC", "SV", "CS"])],
    ["FB", "BRK"],
    "OFF",
)
s = p[p.swing.astype(bool) & (p.bat_speed >= 50)].copy()
s["two"] = s.strikes.astype(int) == 2
s["cell"] = s.cls + s.azone
s["whiff100"] = s.whiff.astype(float) * 100
s["bip"] = s.type == "X"
s["q100"] = np.where(s.bip, 100 * (s.launch_speed - s.bat_speed) / (0.92 * s.release_speed + s.bat_speed), np.nan)
s["ev"] = np.where(s.bip, s.launch_speed, np.nan)
s["bsy"] = s.batter.astype(str) + s.side + s.year
rows = []
for b, g in s.groupby("bsy"):
    a, t = g[~g.two], g[g.two]
    if len(t) < 120 or len(a) < 250:
        continue
    w = a.cell.value_counts(normalize=True)
    out = {"bsy": b, "n2": len(t)}
    for y in ("bat_speed", "whiff100", "ev", "q100"):
        ma, mt = a.groupby("cell")[y].mean(), t.groupby("cell")[y].mean()
        cells = [c for c in w.index if c in mt.index and pd.notna(mt[c]) and pd.notna(ma.get(c))]
        ww = w[cells] / w[cells].sum()
        out["d_" + y] = float((ww * (mt[cells] - ma[cells])).sum())
    rows.append(out)
R = pd.DataFrame(rows)
print(
    f"hitter-seasons: {len(R)}; average two-strike change (same pitch mix): bat speed {R.d_bat_speed.mean():+.2f} mph "
    f"(SD across hitters {R.d_bat_speed.std():.2f}), whiff {R.d_whiff100.mean():+.2f} pp, EV {R.d_ev.mean():+.2f} mph, q x100 {R.d_q100.mean():+.2f}"
)
print("\nper 1 mph of DELIBERATE bat speed (across hitters; slope of two-strike change on bat-speed change):")
for y, lab, naive in (
    ("d_whiff100", "whiff % of swings", "-0.07 (z -3.2)"),
    ("d_ev", "exit velocity (mph)", "+0.76 (z +86)"),
    ("d_q100", "collision efficiency q x100", "-0.22 (z -38)"),
):
    z = R.dropna(subset=[y])
    m = sm.WLS(z[y], sm.add_constant(z.d_bat_speed), weights=z.n2).fit(cov_type="HC1")
    print(f"   {lab:30s} {m.params.iloc[1]:+.2f} (z {m.tvalues.iloc[1]:+.1f})   | naive swing-level slope was {naive}")
R.to_csv("intended_speed.csv", index=False)
