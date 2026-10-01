"""PCA-style event study: what happens to a hitter's contact after a sustained mid-season move BACK
(or UP) in the box?

Treated: batter-seasons with a detected depth event (events.py; threshold via argv, default 4 in).
Controls: every batter-season without a depth event (>= 4 in either way).
Event time in calendar days; bins: B0 [-90,-31] = baseline (reference), B1 [-30,-1] = run-up (a slump
here = selection on a slump -> mean reversion), A0 [0,29], A1 [30,59], A2 [60,90].
Pitch level: y ~ sum_bins beta_{dir,bin} + batter-season FE + date FE + count x zone FE; SE by batter.
Day level (stance data): depth (first stage), contact point vs plate, same bins.
Reading: an effect of moving back = post bins above BASELINE (not just above the run-up) and BACK above UP.
"""

import sys, numpy as np, pandas as pd, warnings
from stance_lib import pitches
from fe import feols

warnings.filterwarnings("ignore")
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
BINS = [("B1", -30, -1), ("A0", 0, 29), ("A1", 30, 59), ("A2", 60, 90)]  # B0 [-90,-31] = reference

import events as E  # re-use detection at the chosen threshold

s = pd.read_parquet("stance_adj.parquet")
ev = []
for _, g in s.sort_values(["bsy", "game_date"]).groupby("bsy"):
    r = E.detect(g, "depth", THR)
    if r:
        ev.append(r)
ev = pd.DataFrame(ev)
ev["dir"] = np.where(ev.move > 0, "BACK", "UP")
print(
    f"threshold {THR} in: {len(ev)} events ({ev.dir.value_counts().to_dict()}), mean |move| {ev.move.abs().mean():.1f} in"
)


def add_bins(df, date_col):
    df = df.merge(ev[["bsy", "event_date", "dir"]], on="bsy", how="left")
    t = (pd.to_datetime(df[date_col]) - pd.to_datetime(df.event_date)).dt.days
    df["t"] = t
    treated = df.dir.notna()
    df = df[~treated | t.between(-90, 90)].copy()  # treated: keep the +-90 day window only
    cols = []
    for d in ("BACK", "UP"):
        for b, lo, hi in BINS:
            c = f"{d}_{b}"
            df[c] = ((df.dir == d) & df.t.between(lo, hi)).astype(float)
            cols.append(c)
    return df, cols


def show(title, b, V, cols, scale, unit):
    print(f"\n  {title}  (vs baseline [-90,-31]; {unit})")
    for d in ("BACK", "UP"):
        cells = []
        for bn, lo, hi in BINS:
            c = f"{d}_{bn}"
            se = np.sqrt(V.loc[c, c])
            cells.append(f"{bn}[{lo},{hi}] {b[c]*scale:+6.2f} ({se*scale:.2f})")
        print(f"    {d:4s} " + " | ".join(cells))
    # post (A0+A1) minus baseline, BACK minus UP
    for d in ("BACK", "UP"):
        L = pd.Series(0.0, index=cols)
        L[f"{d}_A0"] = 0.5
        L[f"{d}_A1"] = 0.5
        est = L @ b
        se = np.sqrt(L @ V @ L)
        print(f"    {d:4s} post[0,59] vs baseline: {est*scale:+.2f} (SE {se*scale:.2f}, z {est/se:+.2f})")
    L = pd.Series(0.0, index=cols)
    for c, w in (("BACK_A0", 0.5), ("BACK_A1", 0.5), ("UP_A0", -0.5), ("UP_A1", -0.5)):
        L[c] = w
    est = L @ b
    se = np.sqrt(L @ V @ L)
    print(f"    BACK minus UP, post:          {est*scale:+.2f} (SE {se*scale:.2f}, z {est/se:+.2f})")


# ---- day level: first stage + contact point
s2, cols = add_bins(s.copy(), "game_date")
for y, lab in (
    ("depth_adj", "depth in box (in, + = deeper)"),
    ("avg_intercept_y_vs_plate", "contact point vs plate (in, + = out front)"),
):
    d = s2.dropna(subset=[y])
    b, V = feols(d, y, cols, ["bsy", "game_date"], "batter")
    show(lab, b, V, cols, 1, "inches")

# ---- pitch level
p = pitches()
p["bsy"] = p.batter.astype(str) + p.side + p.year
p["count"] = p.balls.astype(str) + p.strikes.astype(str)
p["zone_i"] = p.zone.fillna(0).astype(int)
p["brk"] = p.pitch_type.isin({"SL", "ST", "CU", "KC", "SV"})
p, cols = add_bins(p, "game_date")
p["whiff_f"] = p.whiff.astype(float)
p["swing_f"] = p.swing.astype(float)
p["xwoba"] = p.estimated_woba_using_speedangle
tests = (
    ("whiff per swing (pp)", p[p.swing], "whiff_f", 100),
    ("zone whiff per swing (pp)", p[p.swing & p.in_zone], "whiff_f", 100),
    ("breaking-ball whiff per swing (pp)", p[p.swing & p.brk], "whiff_f", 100),
    ("chase: swing | out of zone (pp)", p[p.out_zone], "swing_f", 100),
    ("breaking-ball chase (pp)", p[p.out_zone & p.brk], "swing_f", 100),
    ("xwOBA on contact (points)", p[p.xwoba.notna() & (p.type == "X")], "xwoba", 1000),
)
for lab, d, y, sc in tests:
    b, V = feols(d, y, cols, ["bsy", "game_date", ("count", "zone_i")], "batter")
    show(lab, b, V, cols, sc, "")
ev.to_csv(f"events_thr{int(THR)}.csv", index=False)
