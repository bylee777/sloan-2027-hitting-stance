"""Hitter-day POWER outcomes from raw Statcast (for 'are we trading contact for home runs?').
Per batter x side x game_date: PA, HR, balls in play, tracked BIP, barrels (MLB definition: EV >= 98 mph and a launch-angle window
that widens from 26-30 deg at 98 mph to 8-50 deg at 116+), hard-hit (EV >= 95), expected HR (P(HR | EV, LA) from a gradient-boosting
model fit on all 2024-26 tracked BIP, cross-fitted by season so no ball predicts itself), EV sum, fly balls, extra-base hits, at-bats.
Writes hitter_day_power.parquet (joins hitter_day_outcomes.parquet on batter, side, game_date)."""

import glob, numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier

COLS = ["game_date", "game_type", "batter", "stand", "events", "type", "bb_type", "launch_speed", "launch_angle"]
R = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("raw/*.parquet"))], ignore_index=True)
R = R[R.game_type == "R"].copy()
for c in ("launch_speed", "launch_angle"):
    R[c] = pd.to_numeric(R[c], errors="coerce").astype(float)
for c in ("events", "type", "bb_type"):
    R[c] = R[c].astype(object).where(R[c].notna(), None)
R["game_date"] = pd.to_datetime(R.game_date).dt.strftime("%Y-%m-%d")
R["year"] = R.game_date.str[:4]
R["pa"] = R.events.notna() & ~R.events.isin(["truncated_pa"])
R["hr"] = R.events == "home_run"
R["bip"] = R.type == "X"
R["trk"] = R.bip & R.launch_speed.notna() & R.launch_angle.notna()
ev, la = R.launch_speed, R.launch_angle
lo = np.clip(26 - (ev - 98), 8, 26)
hi = np.where(ev < 99, 30, np.where(ev < 100, 31, np.clip(33 + (ev - 100) * (17 / 16), 33, 50)))
R["barrel"] = R.trk & (ev >= 98) & (la >= lo) & (la <= hi)
R["hard"] = R.trk & (ev >= 95)
R["fb"] = R.bip & (R.bb_type == "fly_ball")
R["xbh"] = R.events.isin(["double", "triple", "home_run"])
R["tb_extra"] = R.events.map({"double": 1, "triple": 2, "home_run": 3}).fillna(0)  # ISO numerator
NON_AB = [
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_fly",
    "sac_bunt",
    "catcher_interf",
    "sac_fly_double_play",
    "sac_bunt_double_play",
]
R["ab"] = R.pa & ~R.events.isin(NON_AB)
# expected HR per tracked ball, cross-fitted by season
T = R[R.trk].copy()
T["xhr"] = np.nan
for y in sorted(T.year.unique()):
    tr, te = T.year != y, T.year == y
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=7)
    m.fit(T.loc[tr, ["launch_speed", "launch_angle"]], T.loc[tr, "hr"])
    T.loc[te, "xhr"] = m.predict_proba(T.loc[te, ["launch_speed", "launch_angle"]])[:, 1]
R["xhr"] = 0.0
R.loc[T.index, "xhr"] = T.xhr
R["ev_sum"] = np.where(R.trk, R.launch_speed, 0.0)
R = R.rename(columns={"stand": "side"})
agg = dict(
    PA_p=("pa", "sum"),
    HR=("hr", "sum"),
    BIP=("bip", "sum"),
    TRK=("trk", "sum"),
    BRL=("barrel", "sum"),
    HARD=("hard", "sum"),
    FB=("fb", "sum"),
    XBH=("xbh", "sum"),
    TBX=("tb_extra", "sum"),
    AB=("ab", "sum"),
    XHR=("xhr", "sum"),
    EV_SUM=("ev_sum", "sum"),
)
P = R.groupby(["batter", "side", "game_date"]).agg(**agg).reset_index()
P.to_parquet("hitter_day_power.parquet", index=False)
D = pd.read_parquet("hitter_day_outcomes.parquet")
J = D.merge(P, on=["batter", "side", "game_date"], how="left")
print(f"raw regular-season pitches {len(R):,}; hitter-days {len(P):,}; joined to outcomes {J.PA_p.notna().mean():.1%}")
print(
    f"PA agreement (power vs outcomes table): corr {J[['PA', 'PA_p']].corr().iloc[0, 1]:.4f}, mean diff {(J.PA_p - J.PA).mean():+.3f}"
)
print(
    f"league: HR/PA {P.HR.sum() / P.PA_p.sum():.4f}, barrel/BIP {P.BRL.sum() / P.TRK.sum():.3f}, xHR/HR {P.XHR.sum() / P.HR.sum():.3f}, "
    f"hard-hit {P.HARD.sum() / P.TRK.sum():.3f}"
)
