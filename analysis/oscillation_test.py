"""Do hitters who flip their stance back and forth ('oscillators') do worse than hitters who commit to one direction?
(Martin, RotoGraphs 2026: oscillators declined more, 73% vs 53%.) Monthly stance (angle, width, depth), 2024-26.
Per hitter-season: path = sum |month-to-month change|, net = |last - first|, oscillation = 1 - net / path.
Tinkering = z-scored path (averaged over the three measures); primary measure = largest z.
(1) Replication: movers (top half of tinkering) split into oscillators (oscillation >= 0.5 on the primary measure) vs
    directional; outcome = 2nd-half minus 1st-half xwOBA (Jul-Sep vs Mar-Jun).
(2) Bounce-back control: regression of that change on oscillation, tinkering and the 1st-half level vs the hitter's prior
    season (a hot or cold 1st half regresses), season FE, PA-weighted.
(3) Prospective: oscillation measured Apr-Jul only; outcome Aug-Sep minus Apr-Jul, same controls."""

import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
G = pd.read_parquet("front_foot_monthly_geometry.parquet").dropna(subset=["stance_angle", "width_in", "depth_in"])
G["mo"] = G.month.str[5:7].astype(int)
G = G[G.mo.between(3, 9)]
D = pd.read_parquet("hitter_day_outcomes.parquet")
D["mo"] = D.game_date.str[5:7].astype(int)
MEAS = {"stance_angle": "angle", "width_in": "width", "depth_in": "depth"}


def osc_stats(g):
    out = {}
    for c, k in MEAS.items():
        x = g.sort_values("mo")[c].to_numpy()
        path = np.abs(np.diff(x)).sum()
        net = abs(x[-1] - x[0])
        out[f"path_{k}"] = path
        out[f"osc_{k}"] = 1 - net / path if path > 0 else 0.0
    return pd.Series(out)


def xw(d):
    return 1000 * d.xw_sum.sum() / d.PA.sum() if d.PA.sum() else np.nan


prev = (
    D.groupby(["batter", "side", "year"])
    .apply(lambda d: pd.Series({"xw_prev": xw(d), "PA_prev": d.PA.sum()}))
    .reset_index()
)
prev["year"] = (prev.year.astype(int) + 1).astype(str)


def build(months_osc, h1, h2):
    g = G[G.mo.isin(months_osc)]
    ok = g.groupby(["batter", "side", "year"]).mo.transform("nunique") >= len(months_osc) - 1
    S = g[ok].groupby(["batter", "side", "year", "name"]).apply(osc_stats).reset_index()
    for k in MEAS.values():
        S[f"z_{k}"] = (S[f"path_{k}"] - S[f"path_{k}"].mean()) / S[f"path_{k}"].std()
    S["tinker"] = S[[f"z_{k}" for k in MEAS.values()]].mean(1)
    S["primary"] = S[[f"z_{k}" for k in MEAS.values()]].idxmax(1).str[2:]
    S["osc"] = S.apply(lambda r: r[f"osc_{r.primary}"], axis=1)
    o = (
        D.groupby(["batter", "side", "year"])
        .apply(
            lambda d: pd.Series(
                {
                    "xw1": xw(d[d.mo.isin(h1)]),
                    "PA1": d[d.mo.isin(h1)].PA.sum(),
                    "xw2": xw(d[d.mo.isin(h2)]),
                    "PA2": d[d.mo.isin(h2)].PA.sum(),
                }
            )
        )
        .reset_index()
    )
    X = S.merge(o, on=["batter", "side", "year"]).merge(prev, on=["batter", "side", "year"], how="left")
    X = X[(X.PA1 >= 150) & (X.PA2 >= 100)].copy()
    X["dxw"] = X.xw2 - X.xw1
    X["hot1"] = X.xw1 - X.xw_prev.where(X.PA_prev >= 200)
    return X


for lab, mo_osc, h1, h2 in (
    ("FULL SEASON (replication)", range(3, 10), range(3, 7), range(7, 10)),
    ("PROSPECTIVE (oscillation Apr-Jul -> Aug-Sep)", range(4, 8), range(3, 8), range(8, 10)),
):
    X = build(list(mo_osc), list(h1), list(h2))
    mv = X[X.tinker >= X.tinker.median()]
    osc, dirn = mv[mv.osc >= 0.5], mv[mv.osc < 0.5]
    print(f"\n=== {lab}: {len(X)} hitter-seasons; movers {len(mv)} (oscillators {len(osc)}, directional {len(dirn)})")
    for nm, z in (("oscillators", osc), ("directional", dirn), ("non-movers", X[X.tinker < X.tinker.median()])):
        print(
            f"   {nm:12s} share declined {100*(z.dxw < 0).mean():4.0f}% | mean change {np.average(z.dxw, weights=z.PA2):+5.1f} xwOBA pts | 1st half vs prior season {z.hot1.mean():+5.1f}"
        )
    Y = X.dropna(subset=["hot1"])
    r = smf.wls("dxw ~ osc + tinker + hot1 + C(year)", Y, weights=Y.PA2).fit(cov_type="HC1")
    r2 = smf.wls("dxw ~ osc * tinker + hot1 + C(year)", Y, weights=Y.PA2).fit(cov_type="HC1")
    print(
        f"   regression (n {len(Y)}): oscillation 0 -> 1 {r.params['osc']:+.1f} pts (z {r.tvalues['osc']:+.1f}); tinkering +1 SD {r.params['tinker']:+.1f} "
        f"(z {r.tvalues['tinker']:+.1f}); hot 1st half +10 pts -> {10*r.params['hot1']:+.1f} (bounce-back); osc x tinker {r2.params['osc:tinker']:+.1f} (z {r2.tvalues['osc:tinker']:+.1f})"
    )
    v = X[X.name.str.contains("Guerrero Jr.") & (X.year == "2026")]
    if len(v):
        print(
            f"   Guerrero 2026: tinkering {v.tinker.iloc[0]:+.2f} SD, primary {v.primary.iloc[0]}, oscillation {v.osc.iloc[0]:.2f}, change {v.dxw.iloc[0]:+.0f}"
        )
