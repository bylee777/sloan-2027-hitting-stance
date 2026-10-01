"""Stride equalizer, deeper: (1) WITHIN hitter: when a hitter changes his setup width, how much of it reaches his
landing width? (month to month, hitter FE; and the 201 sustained width events, pre vs post months).
(2) Is the common landing (~57% of height) a SWEET SPOT? landing/height vs bat speed, whiff, xwOBA on contact,
between hitter-seasons (quadratic; peak vs population mean) and within hitter."""

import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
g = pd.read_parquet("front_foot_monthly_geometry.parquet").dropna(subset=["width_in", "width_land_in", "stride_len_in"])
g["bs"] = g.batter.astype(str) + g.side
# (1a) month to month within hitter (deviation from the hitter's own mean)
for c in ("width_in", "width_land_in", "stride_len_in"):
    g[c + "_dm"] = g[c] - g.groupby("bs")[c].transform("mean")
r = smf.ols("width_land_in_dm ~ width_in_dm - 1", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.bs)[0]})
r2 = smf.ols("stride_len_in_dm ~ width_in_dm - 1", g).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(g.bs)[0]}
)
print(f"(1a) month to month, same hitter ({len(g):,} hitter-months, {g.bs.nunique()} hitters):")
print(
    f"     +1 in wider setup -> landing {r.params.iloc[0]:+.2f} in (SE {r.bse.iloc[0]:.2f}), stride {r2.params.iloc[0]:+.2f} in (SE {r2.bse.iloc[0]:.2f})"
)
print(
    f"     within-hitter SD: setup width {g.width_in_dm.std():.2f} in, landing width {g.width_land_in_dm.std():.2f} in"
)
# (1b) sustained width events: 2 months before vs 2 months after (skip the event month)
ev = pd.read_csv("width_events.csv")
ev["ym"] = ev.event_date.str[:7]
g["t"] = pd.PeriodIndex(g.month, freq="M")
rows = []
for _, e in ev.iterrows():
    h = g[(g.batter == e.batter) & (g.side == e.side)]
    t0 = pd.Period(e.ym, freq="M")
    k = (h.t - t0).apply(lambda x: x.n)
    pre, post = h[k.between(-2, -1)], h[k.between(1, 2)]
    if len(pre) and len(post):
        rows.append(
            dict(
                name=e["name"],
                dir=e.dir,
                d_setup=post.width_in.mean() - pre.width_in.mean(),
                d_land=post.width_land_in.mean() - pre.width_land_in.mean(),
                d_stride=post.stride_len_in.mean() - pre.stride_len_in.mean(),
            )
        )
E = pd.DataFrame(rows)
b = np.polyfit(E.d_setup, E.d_land, 1)
print(f"\n(1b) {len(E)} sustained width changes (2 months before vs after):")
for d, z in E.groupby("dir"):
    print(
        f"     {d:9s} n {len(z):3d}: setup {z.d_setup.mean():+5.1f} in -> landing {z.d_land.mean():+5.1f} in, stride {z.d_stride.mean():+5.1f} in "
        f"(share of the setup change that reaches landing {z.d_land.mean() / z.d_setup.mean():.0%})"
    )
print(
    f"     slope landing-change on setup-change {b[0]:+.2f} (1 = landing moves with the setup, 0 = landing fully defended)"
)
E.to_csv("equalizer_events.csv", index=False)
# (2) sweet spot?
H = pd.read_parquet("taxonomy/hitters_pca.parquet").dropna(subset=["width_land_in", "height_in", "bat_speed"])
H["land_rel"] = 100 * H.width_land_in / H.height_in
H["xwcon"] = 1000 * H.xwoba_con
H["whiff100"] = 100 * H.whiff
H["xw"] = 1000 * H.xwoba
print(
    f"\n(2) landing width as % of height: mean {H.land_rel.mean():.1f}%, SD {H.land_rel.std():.1f} ({len(H)} hitter-seasons)"
)
for y, lab in (
    ("bat_speed", "bat speed (mph)"),
    ("whiff100", "whiff %"),
    ("xwcon", "xwOBA on contact"),
    ("xw", "xwOBA"),
):
    m = smf.ols(f"{y} ~ land_rel + I(land_rel**2) + C(year) + C(side)", H).fit(
        cov_type="cluster", cov_kwds={"groups": H.batter}
    )
    a, q = m.params["land_rel"], m.params["I(land_rel ** 2)"]
    peak = -a / (2 * q) if q != 0 else np.nan
    lin = smf.ols(f"{y} ~ land_rel + C(year) + C(side)", H).fit(cov_type="cluster", cov_kwds={"groups": H.batter})
    print(
        f"     {lab:18s} curvature {q:+.4f} (z {m.tvalues['I(land_rel ** 2)']:+.1f}) -> {'peak' if q < 0 else 'trough'} at {peak:5.1f}% | "
        f"linear per +5% of height {5*lin.params['land_rel']:+.2f} (z {lin.tvalues['land_rel']:+.1f})"
    )
# within hitter: seasons landing wider (relative to height) vs his own average
H["bs"] = H.batter.astype(str) + H.side
for c in ("land_rel", "bat_speed", "whiff100", "xwcon"):
    H[c + "_dm"] = H[c] - H.groupby("bs")[c].transform("mean")
Hm = H[H.groupby("bs").bs.transform("size") >= 2]
for y, lab in (("bat_speed", "bat speed"), ("whiff100", "whiff %"), ("xwcon", "xwOBA on contact")):
    m = smf.ols(f"{y}_dm ~ land_rel_dm - 1", Hm).fit(cov_type="cluster", cov_kwds={"groups": Hm.batter})
    print(
        f"     within hitter, per +5% of height wider landing: {lab:16s} {5*m.params.iloc[0]:+.2f} (z {m.tvalues.iloc[0]:+.1f})"
    )
