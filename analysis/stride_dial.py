"""Is stance width really a STRIDE-LENGTH dial? Mediation, within hitter.
Monthly panel (hitter FE + calendar-month FE, swing-weighted, batter-clustered):
   outcome ~ setup width                      (total effect)
   outcome ~ setup width + stride length      (does the stride carry it? setup coefficient should shrink)
Outcomes: whiff %, bat speed, swing length, K %, xwOBA on contact, chase %.
Event check: 177 sustained width changes, 2 months before vs after: change in outcome ~ change in setup + change in stride.
"""

import numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from fe import demean

g = pd.read_parquet("front_foot_monthly_geometry.parquet").dropna(subset=["width_in", "width_land_in", "stride_len_in"])
D = pd.read_parquet("hitter_day_outcomes.parquet")
D["month"] = D.game_date.str[:7]
M = (
    D.groupby(["batter", "side", "month"])[
        ["PA", "xw_sum", "K", "sw", "wh", "oz", "oz_sw", "xwc_sum", "xwc_n", "bs_sum", "bs_n", "sl_sum"]
    ]
    .sum()
    .reset_index()
)
M["whiff"] = 100 * M.wh / M.sw
M["bat_speed"] = M.bs_sum / M.bs_n
M["swing_len"] = M.sl_sum / M.bs_n
M["K_pct"] = 100 * M.K / M.PA
M["xwcon"] = 1000 * M.xwc_sum / M.xwc_n
M["chase"] = 100 * M.oz_sw / M.oz
X = g.merge(M, on=["batter", "side", "month"])
X = X[(X.sw >= 40) & (X.bs_n >= 30)].copy()
X["bs"] = X.batter.astype(str) + X.side
X["cal"] = X.month.str[5:]
X["setup3"] = X.width_in / 3
X["stride3"] = X.stride_len_in / 3
X["land3"] = X.width_land_in / 3
OUT = {
    "whiff": "whiff % (pp)",
    "bat_speed": "bat speed (mph)",
    "swing_len": "swing length (ft)",
    "K_pct": "strikeout % (pp)",
    "xwcon": "xwOBA on contact",
    "chase": "chase % (pp)",
}


def fe_fit(d, y, xs, w):
    codes = [pd.factorize(d.bs)[0], pd.factorize(d.month)[0]]
    Mx = demean(
        np.column_stack([d[y].to_numpy(float)] + [d[x].to_numpy(float) for x in xs]), codes, iters=200, tol=1e-9
    )
    ww = d[w].to_numpy(float)
    r = sm.WLS(Mx[:, 0], Mx[:, 1:], weights=ww).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.bs)[0]})
    return r.params, r.bse


print(f"monthly panel: {len(X):,} hitter-months, {X.bs.nunique()} hitters (per 3 in; within hitter)")
print(f"{'':20s} {'setup alone':>18s} | {'setup (w/ stride)':>18s} {'stride':>18s} | share carried by stride")
rows = []
for y, lab in OUT.items():
    d = X.dropna(subset=[y])
    b1, s1 = fe_fit(d, y, ["setup3"], "sw")
    b2, s2 = fe_fit(d, y, ["setup3", "stride3"], "sw")
    share = 1 - b2[0] / b1[0] if abs(b1[0]) > 1e-9 else np.nan
    print(
        f"{lab:20s} {b1[0]:+7.2f} (z {b1[0]/s1[0]:+5.1f}) | {b2[0]:+7.2f} (z {b2[0]/s2[0]:+5.1f}) {b2[1]:+7.2f} (z {b2[1]/s2[1]:+5.1f}) | {share:5.0%}"
    )
    rows.append(
        dict(
            outcome=y,
            setup_alone=b1[0],
            z_alone=b1[0] / s1[0],
            setup_joint=b2[0],
            z_setup_joint=b2[0] / s2[0],
            stride=b2[1],
            z_stride=b2[1] / s2[1],
            share=share,
        )
    )
pd.DataFrame(rows).to_csv("stride_dial_monthly.csv", index=False)
# ---- event check
E = pd.read_csv("equalizer_events.csv") if False else None
ev = pd.read_csv("width_events.csv")
ev["ym"] = ev.event_date.str[:7]
X["t"] = pd.PeriodIndex(X.month, freq="M")
rows = []
for _, e in ev.iterrows():
    h = X[(X.batter == e.batter) & (X.side == e.side)]
    if not len(h):
        continue
    k = (h.t - pd.Period(e.ym, freq="M")).apply(lambda x: x.n)
    pre, post = h[k.between(-2, -1)], h[k.between(1, 2)]
    if len(pre) and len(post):
        r = dict(name=e["name"], dir=e.dir)
        for c in ["width_in", "stride_len_in", "width_land_in"] + list(OUT):
            wa, wb = post.sw, pre.sw
            r["d_" + c] = np.average(post[c], weights=wa) - np.average(pre[c], weights=wb)
        rows.append(r)
E = pd.DataFrame(rows)
print(f"\nevent check: {len(E)} sustained width changes (per 3 in; 2 months after minus before)")
for y, lab in OUT.items():
    e = E.dropna(subset=["d_" + y])
    a = sm.OLS(e["d_" + y], sm.add_constant(e[["d_width_in"]] / 3)).fit(cov_type="HC1")
    b = sm.OLS(e["d_" + y], sm.add_constant(e[["d_width_in", "d_stride_len_in"]] / 3)).fit(cov_type="HC1")
    print(
        f"   {lab:20s} setup alone {a.params.iloc[1]:+6.2f} (z {a.tvalues.iloc[1]:+.1f}) | with stride: setup {b.params.iloc[1]:+6.2f} (z {b.tvalues.iloc[1]:+.1f}), "
        f"stride {b.params.iloc[2]:+6.2f} (z {b.tvalues.iloc[2]:+.1f})"
    )
E.to_csv("stride_dial_events.csv", index=False)
