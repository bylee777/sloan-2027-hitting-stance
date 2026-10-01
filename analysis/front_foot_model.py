"""Model support for the front-foot finding + foot position vs left/right-handed pitchers.
A. Do hitters set their front foot differently vs same- vs opposite-hand pitchers? (season geometry by pitcher hand)
B. Monthly panel (hitter-month, >= 40 PA): does opening the front foot at SETUP / at LANDING change K, chase, whiff, xwOBA?
   (a) within-hitter FE: hitter-season FE + calendar-month FE, controls = rest of stance + bat speed/length + last
       month's outcomes; SE clustered by hitter.  (b) double ML: gradient-boosted nuisance models, 5-fold
       cross-fitting grouped by hitter, controls incl. hitter-season means (Mundlak) and lagged outcomes.
   (c) placebo: NEXT month's foot turn added to (a).  (d) outcomes on same- vs opposite-hand pitches."""

import numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from fe import feols
from stance_lib import pitches

rng = np.random.default_rng(0)
# ---------------- A. by pitcher hand
fh = pd.read_parquet("front_foot_by_pitcher_hand.parquet")
fh["same"] = np.where(fh.side == fh.p_throws, "same", "opp")
p = pitches()
p["same_b"] = p.stand == p.p_throws
sw_n = (
    p[p.swing]
    .groupby(["batter", "side", "year", "same_b"])
    .size()
    .unstack()
    .rename(columns={True: "n_same", False: "n_opp"})
    .reset_index()
)
wide = fh.pivot_table(
    index=["batter", "side", "year"],
    columns="same",
    values=["front_turn0", "front_turn2", "stance_angle", "depth_in", "width_in"],
).reset_index()
wide.columns = ["_".join(c).strip("_") for c in wide.columns]
wide = wide.merge(sw_n, on=["batter", "side", "year"])
wide = wide[(wide.n_same >= 60) & (wide.n_opp >= 60)]
print(f"A. {len(wide)} hitter-seasons with >= 60 swings vs each hand: SAME-hand minus OPPOSITE-hand")
for c, lab in (
    ("front_turn0", "front-foot turn at setup (deg)"),
    ("front_turn2", "front-foot turn at landing (deg)"),
    ("stance_angle", "stance-line angle (deg)"),
    ("depth_in", "depth (in)"),
    ("width_in", "width (in)"),
):
    dlt = wide[f"{c}_same"] - wide[f"{c}_opp"]
    nx = wide.assign(year=(wide.year.astype(int) - 1).astype(str)).assign(d=dlt)[["batter", "side", "year", "d"]]
    rel = wide.assign(d=dlt).merge(nx, on=["batter", "side", "year"], suffixes=("", "_n"))
    print(
        f"   {lab:34s} mean {dlt.mean():+.2f}, SD {dlt.std():.2f}, |diff| >= 10 deg or 3 in: "
        f"{(dlt.abs() >= (10 if 'deg' in lab else 3)).mean():.0%}; same hitter next season r = {rel.d.corr(rel.d_n):+.2f}"
    )
# ---------------- B. monthly panel
g = pd.read_parquet("front_foot_monthly_geometry.parquet")
for c in ("front_turn0", "front_turn2"):
    lo, hi = g[c].quantile([0.01, 0.99])
    g[c] = g[c].clip(lo, hi)
p["month"] = p.game_date.str[:7]
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
p["is_pa"] = p.events.notna() & ~p.events.fillna("").astype(str).str.startswith(NON_PA)
p["K"] = p.events.isin(["strikeout", "strikeout_double_play"]) & p.is_pa
p["xw"] = np.where(
    p.estimated_woba_using_speedangle.notna(),
    p.estimated_woba_using_speedangle.astype(float),
    p.woba_value.astype(float),
)


def agg(d):
    pa = d[d.is_pa]
    return pd.Series(
        {
            "PA": len(pa),
            "K": 100 * pa.K.mean() if len(pa) else np.nan,
            "chase": 100 * d[d.out_zone].swing.mean(),
            "whiff": 100 * d[d.swing].whiff.mean(),
            "xwoba": 1000 * np.nanmean(pa.xw) if len(pa) else np.nan,
            "bat_speed": d.bat_speed.astype(float).mean(),
            "swing_len": d.swing_length.astype(float).mean(),
        }
    )


key = ["batter", "side", "month"]
o_all = p.groupby(key).apply(agg).reset_index()
o_same = p[p.same_b].groupby(key).apply(agg).reset_index().rename(columns=lambda c: c if c in key else c + "_same")
o_opp = p[~p.same_b].groupby(key).apply(agg).reset_index().rename(columns=lambda c: c if c in key else c + "_opp")
D = g.merge(o_all, on=key).merge(o_same, on=key, how="left").merge(o_opp, on=key, how="left")
D = D.merge(pd.read_csv("bio_all.csv")[["batter", "height_in"]], on="batter", how="left")
D = D[D.PA >= 40].sort_values(key).reset_index(drop=True)
D["bsy"] = D.batter.astype(str) + D.side + D.year
for c in ("K", "chase", "whiff", "xwoba"):
    D[c + "_lag"] = D.groupby("bsy")[c].shift(1)
D["t0"] = D.front_turn0 / 10
D["t2"] = D.front_turn2 / 10
D["t0_next"] = D.groupby("bsy").t0.shift(-1)
D["t2_next"] = D.groupby("bsy").t2.shift(-1)
print(
    f"\nB. monthly panel: {len(D):,} hitter-months ({D.bsy.nunique()} hitter-seasons); within-hitter month-to-month SD: "
    f"setup turn {10*(D.t0 - D.groupby('bsy').t0.transform('mean')).std():.1f} deg, landing turn {10*(D.t2 - D.groupby('bsy').t2.transform('mean')).std():.1f} deg"
)
CTRL = ["stance_angle", "width_in", "depth_in", "off_plate_in", "stride_len_in", "bat_speed", "swing_len"]
OUT = [("K", "strikeout % (pp)"), ("chase", "chase % (pp)"), ("whiff", "whiff % (pp)"), ("xwoba", "xwOBA (pts)")]


def fe_fit(d, y, xs):
    d = d.dropna(subset=[y] + xs)
    b, V = feols(d, y, xs, ["bsy", "month"], "batter")
    return b, np.sqrt(np.diag(V)), len(d)


print("\n(a) within hitter, per 10 deg MORE OPEN front foot (month to month; hitter-season + month FE)")
for y, lab in OUT:
    b, se, n = fe_fit(D, y, ["t0", "t2"] + CTRL + [y + "_lag"])
    print(
        f"   {lab:20s} setup {b['t0']:+.2f} (z {b['t0']/se[0]:+.1f}) | landing {b['t2']:+.2f} (z {b['t2']/se[1]:+.1f})   n {n:,}"
    )
print("\n(c) placebo: NEXT month's foot turn added (should be ~0)")
for y, lab in OUT:
    xs = ["t0", "t2", "t0_next", "t2_next"] + CTRL + [y + "_lag"]
    d = D.dropna(subset=[y] + xs)
    b, V = feols(d, y, xs, ["bsy", "month"], "batter")
    se = pd.Series(np.sqrt(np.diag(V)), index=xs)
    print(
        f"   {lab:20s} next-month setup {b['t0_next']:+.2f} (z {b['t0_next']/se['t0_next']:+.1f}) | next-month landing {b['t2_next']:+.2f} (z {b['t2_next']/se['t2_next']:+.1f})"
        f"  || this month setup {b['t0']:+.2f} (z {b['t0']/se['t0']:+.1f})"
    )
print("\n(b) double machine learning (gradient boosting, 5-fold cross-fitting by hitter)")
for c in ["t0", "t2"] + CTRL + ["K", "chase", "whiff", "xwoba"]:
    D[c + "_hm"] = D.groupby("bsy")[c].transform("mean")  # Mundlak: hitter-season means
XC = (
    CTRL
    + ["height_in", "t0_hm", "t2_hm"]
    + [c + "_hm" for c in CTRL]
    + ["K_lag", "chase_lag", "whiff_lag", "xwoba_lag", "K_hm", "chase_hm", "whiff_hm", "xwoba_hm"]
)
D["mo"] = D.month.str[5:].astype(int)
D["yr"] = D.year.astype(int)
D["isR"] = (D.side == "R").astype(int)
XC += ["mo", "yr", "isR"]


def dml(y):
    d = D.dropna(subset=[y, "t0", "t2"]).reset_index(drop=True)
    X = d[XC].to_numpy(float)
    res = {}
    folds = GroupKFold(5).split(X, groups=d.batter)
    r = {k: np.zeros(len(d)) for k in (y, "t0", "t2")}
    for tr, te in folds:
        for k in (y, "t0", "t2"):
            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, random_state=0
            )
            m.fit(X[tr], d[k].to_numpy()[tr])
            r[k][te] = d[k].to_numpy()[te] - m.predict(X[te])
    Z = np.column_stack([r["t0"], r["t2"]])
    fit = sm.OLS(r[y], Z).fit(cov_type="cluster", cov_kwds={"groups": d.batter.to_numpy()})
    return fit.params, fit.bse, len(d)


for y, lab in OUT:
    b, se, n = dml(y)
    print(f"   {lab:20s} setup {b[0]:+.2f} (z {b[0]/se[0]:+.1f}) | landing {b[1]:+.2f} (z {b[1]/se[1]:+.1f})   n {n:,}")
print("\n(d) by matchup, within hitter (as (a)): outcome on SAME-hand vs OPPOSITE-hand pitches")
for y, lab in OUT:
    cells = []
    for suf, nm in (("_same", "same-hand"), ("_opp", "opp-hand")):
        dd = D[D["PA" + suf] >= 15]
        b, se, n = fe_fit(dd, y + suf, ["t0", "t2"] + CTRL + [y + "_lag"])
        cells.append(
            f"{nm}: setup {b['t0']:+.2f} (z {b['t0']/se[0]:+.1f}), landing {b['t2']:+.2f} (z {b['t2']/se[1]:+.1f})"
        )
    print(f"   {lab:20s} " + " | ".join(cells))
D.to_parquet("front_foot_panel.parquet", index=False)
