"""Which hitters set their FRONT FOOT differently vs left- and right-handed pitchers, and how do they differ?
diff = front-foot turn at setup vs SAME-hand minus vs OPPOSITE-hand pitchers (deg; + = more open vs same-hand).
Habitual adjusters: |diff| >= 3 deg with the same sign in two consecutive seasons. Compare with everyone else:
profile (height, age, side, bat speed, fingerprint) and results (platoon gap in xwOBA / K / chase), plus
'does adjusting pay?': between hitters and within hitter season to season."""

import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

fh = pd.read_parquet("front_foot_by_pitcher_hand.parquet")
fh["m"] = np.where(fh.side == fh.p_throws, "same", "opp")
W = fh.pivot_table(
    index=["batter", "name", "side", "year"],
    columns="m",
    values=["front_turn0", "front_turn2", "stance_angle", "depth_in", "width_in"],
).reset_index()
W.columns = ["_".join(c).strip("_") for c in W.columns]
p = pitches()
p["same"] = p.stand == p.p_throws
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
p["is_pa"] = p.events.notna() & ~p.events.fillna("").astype(str).str.startswith(NON_PA)
p["xw"] = np.where(
    p.estimated_woba_using_speedangle.notna(),
    p.estimated_woba_using_speedangle.astype(float),
    p.woba_value.astype(float),
)


def res(d):
    pa = d[d.is_pa]
    return pd.Series(
        {
            "sw": d.swing.sum(),
            "PA": len(pa),
            "xw": 1000 * pa.xw.mean(),
            "K": 100 * pa.events.isin(["strikeout", "strikeout_double_play"]).mean(),
            "chase": 100 * d[d.out_zone].swing.mean(),
            "whiff": 100 * d[d.swing].whiff.mean(),
        }
    )


R = p.groupby(["batter", "side", "year", "same"]).apply(res).unstack("same")
R.columns = [f"{a}_{'same' if b else 'opp'}" for a, b in R.columns]
R = R.reset_index()
X = W.merge(R, on=["batter", "side", "year"])
X = X[(X.sw_same >= 60) & (X.sw_opp >= 60)].copy()
X["diff"] = X.front_turn0_same - X.front_turn0_opp
X["gap_xw"] = X.xw_same - X.xw_opp
X["gap_K"] = X.K_same - X.K_opp
X["gap_chase"] = X.chase_same - X.chase_opp
X["gap_whiff"] = X.whiff_same - X.whiff_opp
X["xw_all"] = (X.xw_same * X.PA_same + X.xw_opp * X.PA_opp) / (X.PA_same + X.PA_opp)
nx = X[["batter", "side", "year", "diff"]].assign(year=(X.year.astype(int) - 1).astype(str))
pair = X.merge(nx, on=["batter", "side", "year"], suffixes=("", "_next"))
hab = pair[(pair["diff"].abs() >= 3) & (pair.diff_next.abs() >= 3) & (np.sign(pair["diff"]) == np.sign(pair.diff_next))]
adj_ids = set(zip(hab.batter, hab.side))
X["adjuster"] = [(b, s) in adj_ids for b, s in zip(X.batter, X.side)]
print(
    f"{len(X)} hitter-seasons (60+ swings vs each hand); diff SD {X['diff'].std():.1f} deg; |diff| >= 3 deg: {(X['diff'].abs() >= 3).mean():.0%}, >= 5 deg: {(X['diff'].abs() >= 5).mean():.0%}"
)
print(
    f"habitual adjusters (|diff| >= 3 deg, same direction, 2 seasons running): {len(adj_ids)} hitters "
    f"({(hab['diff'] > 0).sum()} open MORE vs same-hand, {(hab['diff'] < 0).sum()} open LESS)"
)
nm = (
    X[X.adjuster]
    .groupby(["name", "side"])
    .agg(seasons=("year", lambda y: ", ".join(sorted(y))), diff=("diff", "mean"), gap_xw=("gap_xw", "mean"))
    .reset_index()
    .sort_values("diff")
)
print("\nthe adjusters (avg diff; + = more open vs same-hand) and their same-minus-opposite xwOBA gap:")
print(nm.round(1).to_string(index=False))
# profile comparison
H = pd.read_parquet("taxonomy/hitters_pca.parquet")[
    ["batter", "side", "year", "height_in", "bat_speed", "PC1", "PC2", "PC3", "PC4", "chase", "whiff", "K"]
]
bio = pd.read_csv("bio_all.csv")[["batter", "birth"]]
X = X.merge(H, on=["batter", "side", "year"], how="left").merge(bio, on="batter", how="left")
X["age"] = X.year.astype(int) - pd.to_datetime(X.birth).dt.year
comp = X.groupby("adjuster").agg(
    n=("batter", "size"),
    height=("height_in", "mean"),
    age=("age", "mean"),
    share_R=("side", lambda s: (s == "R").mean()),
    bat_speed=("bat_speed", "mean"),
    chase=("chase", lambda c: 100 * c.mean()),
    whiff=("whiff", lambda c: 100 * c.mean()),
    xw_all=("xw_all", "mean"),
    gap_xw=("gap_xw", "mean"),
    gap_K=("gap_K", "mean"),
    gap_chase=("gap_chase", "mean"),
    setup_turn=("front_turn0_opp", "mean"),
    setup_turn_sd=("front_turn0_opp", "std"),
)
print("\nadjusters vs everyone else (hitter-season averages):")
print(comp.round(2).T.to_string())
# does adjusting pay?  between hitters: gap ~ diff ; within: change in gap ~ change in diff
print("\ndoes opening MORE vs same-hand pitchers shrink the same-hand penalty? (per 5 deg of diff)")
X["d5"] = X["diff"] / 5
for y, lab in (
    ("gap_xw", "same-hand xwOBA gap (pts; + = better vs same-hand)"),
    ("gap_K", "same-hand K% gap (pp)"),
    ("gap_chase", "same-hand chase gap (pp)"),
):
    r = smf.ols(f"{y} ~ d5 + C(side) + C(year)", X).fit(cov_type="HC1")
    pr = X.merge(
        X[["batter", "side", "year", "d5", y]].assign(year=lambda z: (z.year.astype(int) - 1).astype(str)),
        on=["batter", "side", "year"],
        suffixes=("", "_n"),
    )
    pr["dd"] = pr.d5_n - pr.d5
    pr["dy"] = pr[y + "_n"] - pr[y]
    r2 = smf.ols(f"dy ~ dd + {y}", pr).fit(cov_type="HC1")
    print(
        f"   {lab:48s} between hitters {r.params.d5:+.2f} (z {r.tvalues.d5:+.1f}) | same hitter season to season {r2.params.dd:+.2f} (z {r2.tvalues.dd:+.1f})"
    )
X.to_csv("foot_adjusters.csv", index=False)
