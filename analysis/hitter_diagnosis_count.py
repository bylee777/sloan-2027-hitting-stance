"""[COUNT VARIANT of hitter_diagnosis_intended.py — does 'pitch count' belong in the contact model f?]
f = league model of xwOBA on contact from pitch + intended swing + strikes (paper §4.5). Variants, same 5 hitter-grouped folds:
  V0 original inputs | V1 + balls (full ball-strike count) | V2 + pitcher's pitch count so far in the game and times this pitcher
  has faced this hitter in the game | V3 + pitch number within the plate appearance.
Reports out-of-fold R^2 and a hitter-clustered paired test on squared error, the model's partial dependence on the new inputs, and how the
accounting (inputs vs squaring up) changes with the richest variant. Writes hitter_diagnosis_count.csv (does NOT touch the paper's files).
"""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from stance_lib import pitches
from physics import state_at_y, Y_FRONT

p = pitches()
num = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "delta_run_exp",
    "sz_top",
    "sz_bot",
]
for c in num:
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
p = p.dropna(subset=["delta_run_exp", "plate_x", "plate_z"]).copy()
p["rv"] = p.delta_run_exp
# ---- attack zone from the clean zone (hitter-season median top/bottom)
key = p.batter.astype(str) + p.side + p.year
top = p.groupby(key).sz_top.transform("median")
bot = p.groupby(key).sz_bot.transform("median")
dx = (p.plate_x.abs() - 17 / 24) * 12
dz = np.maximum(bot - p.plate_z, p.plate_z - top) * 12
d = np.maximum(dx, dz)
p["azone"] = np.select([d <= -2.9, d <= 2.9, d <= 8.7], ["heart", "shadow", "chase"], "waste")
p["cls"] = np.select(
    [p.pitch_type.isin(["FF", "SI", "FC"]), p.pitch_type.isin(["SL", "ST", "CU", "KC", "SV", "CS"])],
    ["FB", "BRK"],
    "OFF",
)
p["cnt"] = p.balls.astype(str) + p.strikes.astype(str)
p["cell"] = p.year + "|" + p.cnt + "|" + p.azone + "|" + p.cls
p["sw"] = p.swing.astype(bool)
p["wh"] = p.whiff.astype(bool)
p["bip"] = p.type == "X"
p["contact"] = p.sw & ~p.wh
g = p.groupby("cell")
L = pd.DataFrame(
    {
        "rv_all": g.rv.mean(),
        "rv_sw": p[p.sw].groupby("cell").rv.mean(),
        "rv_tk": p[~p.sw].groupby("cell").rv.mean(),
        "pw": p[p.sw].groupby("cell").wh.mean(),
        "rv_wh": p[p.wh].groupby("cell").rv.mean(),
        "rv_ct": p[p.contact].groupby("cell").rv.mean(),
        "rv_bip": p[p.bip].groupby("cell").rv.mean(),
    }
).fillna(0)
p = p.join(L, on="cell")
p["DECISIONS"] = np.where(p.sw, p.rv_sw, p.rv_tk) - p.rv_all
p["WHIFFS"] = np.where(p.sw, (p.wh.astype(float) - p.pw) * (p.rv_wh - p.rv_ct), 0.0)
# ---- xwOBA -> run value map (season x count, balls in play)
bb = p[p.bip & p.estimated_woba_using_speedangle.notna()]
mp = {}
for (y, c), z in bb.groupby(["year", "cnt"]):
    A = np.column_stack([np.ones(len(z)), z.estimated_woba_using_speedangle])
    mp[(y, c)] = np.linalg.lstsq(A, z.rv.to_numpy(), rcond=None)[0]
coef = np.array([mp.get((y, c), (np.nan, np.nan)) for y, c in zip(p.year, p.cnt)])
RVx = lambda x: coef[:, 0] + coef[:, 1] * x
# ---- league contact-quality model (cross-fitted by hitter)
st = state_at_y(p, Y_FRONT)
p["vaa"] = np.degrees(np.arctan2(st["vz"], -st["vy"]))
sg = np.where(p.stand == "R", 1.0, -1.0)
p["loc_x"] = p.plate_x * sg
p["mov_x"] = p.pfx_x * 12 * sg
p["mov_z"] = p.pfx_z * 12
p["contact_y"] = p.intercept_ball_minus_batter_pos_y_inches
p["contact_x"] = p.intercept_ball_minus_batter_pos_x_inches
p["cls_i"] = p.cls.map({"FB": 0, "BRK": 1, "OFF": 2})
# ---- intended swing (squared-up contact vs fastballs), hitter-season x strikes; fallbacks for thin cells
p["bsy"] = p.batter.astype(str) + p.side + p.year
p["k"] = p.strikes.astype(int).clip(0, 2)
sq = (
    p.bip
    & (p.bat_speed >= 50)
    & (p.launch_speed > 0.8 * (1.23 * p.bat_speed + 0.23 * p.release_speed))
    & (p.cls == "FB")
)
S = p[sq]
cellm = (
    S.groupby(["bsy", "k"])
    .agg(ibs=("bat_speed", "mean"), isl=("swing_length", "mean"), n=("bat_speed", "size"))
    .reset_index()
)
seas = (
    S.groupby("bsy")
    .agg(
        ibs_s=("bat_speed", "mean"),
        isl_s=("swing_length", "mean"),
        iaa=("attack_angle", "mean"),
        itl=("swing_path_tilt", "mean"),
        iad=("attack_direction", "mean"),
        n_s=("bat_speed", "size"),
    )
    .reset_index()
)
comp = (
    p[p.sw & (p.bat_speed >= 50)]
    .groupby("bsy")
    .agg(
        cbs=("bat_speed", "mean"),
        csl=("swing_length", "mean"),
        caa=("attack_angle", "mean"),
        ctl=("swing_path_tilt", "mean"),
        cad=("attack_direction", "mean"),
    )
    .reset_index()
)
p = p.merge(cellm, on=["bsy", "k"], how="left").merge(seas, on="bsy", how="left").merge(comp, on="bsy", how="left")
use_cell = p.n >= 8
use_seas = p.n_s >= 8
p["int_bs"] = np.where(use_cell, p.ibs, np.where(use_seas, p.ibs_s, p.cbs))
p["int_sl"] = np.where(use_cell, p.isl, np.where(use_seas, p.isl_s, p.csl))
p["int_aa"] = np.where(use_seas, p.iaa, p.caa)
p["int_tl"] = np.where(use_seas, p.itl, p.ctl)
p["int_ad"] = np.where(use_seas, p.iad, p.cad)

# ---- 'pitch count' inputs
p = p.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"]).reset_index(drop=True)
coef = np.array(
    [mp.get((y_, c_), (np.nan, np.nan)) for y_, c_ in zip(p.year, p.cnt)]
)  # RVx is positional: rebuild after the sort
RVx = lambda x: coef[:, 0] + coef[:, 1] * x
p["b"] = p.balls.astype(int).clip(0, 3)
p["pc"] = p.groupby(["game_pk", "pitcher"]).cumcount()  # pitcher's pitches before this one (this game)
p["tto"] = (
    p.groupby(["game_pk", "pitcher", "batter"]).at_bat_number.rank(method="dense").clip(1, 4)
)  # times faced this hitter today
p["pnum"] = pd.to_numeric(p.pitch_number, errors="coerce").clip(1, 12)
F0 = [
    "int_bs",
    "int_sl",
    "int_aa",
    "int_tl",
    "int_ad",
    "k",
    "loc_x",
    "plate_z",
    "release_speed",
    "mov_x",
    "mov_z",
    "vaa",
    "cls_i",
]
VAR = {
    "V0 original": F0,
    "V1 + balls": F0 + ["b"],
    "V2 + pitcher count, times faced": F0 + ["b", "pc", "tto"],
    "V3 + pitch # in PA": F0 + ["b", "pc", "tto", "pnum"],
}
trk = (
    p.bip
    & p.estimated_woba_using_speedangle.notna()
    & p[VAR["V3 + pitch # in PA"]].notna().all(1)
    & (p.bat_speed >= 50)
)
T = p[trk]
y = T.estimated_woba_using_speedangle.to_numpy()
folds = list(GroupKFold(5).split(T, groups=T.batter))
PRED = {}
for name, F in VAR.items():
    pred = np.full(len(T), np.nan)
    for tr, te in folds:
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=80, random_state=0
        )
        m.fit(T.iloc[tr][F], y[tr])
        pred[te] = m.predict(T.iloc[te][F])
    PRED[name] = pred
print(f"balls in play modelled: {len(T):,}")
e0 = (y - PRED["V0 original"]) ** 2
bat = T.batter.to_numpy()
for name, pred in PRED.items():
    e = (y - pred) ** 2
    r2 = 1 - e.mean() / y.var()
    d = pd.Series(e0 - e).groupby(bat).sum()
    n = pd.Series(np.ones(len(e))).groupby(bat).sum()
    gain = d.sum() / n.sum()
    se = np.sqrt(((d - gain * n) ** 2).sum()) / n.sum()  # hitter-clustered SE of the mean per-ball gain
    print(f"   {name:34s} out-of-fold R^2 {r2:.4f}   gain vs V0 {gain / y.var():+.4f} R^2 (z {gain / se:+.1f})")
# ---- partial dependence: same pitch and intended swing, change only the new input (model refit on all balls in play, V3 inputs)
F = VAR["V3 + pitch # in PA"]
m = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=80, random_state=0
).fit(T[F], y)
Sx = T.sample(60000, random_state=1)


def pdp(col, vals, label):
    out = []
    for v in vals:
        Z = Sx[F].copy()
        Z[col] = v
        out.append(1000 * m.predict(Z).mean())
    base = out[0]
    print(f"   {label:34s} " + "  ".join(f"{v}: {o - base:+5.1f}" for v, o in zip(vals, out)))


print("\nmodel's view, xwOBA-on-contact points vs the first value (everything else as observed):")
pdp("b", [0, 1, 2, 3], "balls")
pdp("k", [0, 1, 2], "strikes")
pdp("pc", [0, 25, 50, 75, 100], "pitcher's pitches so far")
pdp("tto", [1, 2, 3, 4], "times facing this pitcher today")
pdp("pnum", [1, 3, 5, 7, 9], "pitch number in the PA")
# ---- accounting with the richest variant vs the original
xw = p.estimated_woba_using_speedangle.to_numpy()
p["xw_pred"] = np.nan
p.loc[trk, "xw_pred"] = PRED["V3 + pitch # in PA"]
p["INPUTS"] = np.where(trk, RVx(p.xw_pred.to_numpy()) - p.rv_bip, 0.0)
p["SQUARING"] = np.where(trk, RVx(xw) - RVx(p.xw_pred.to_numpy()), 0.0)
p["LUCK"] = np.where(p.bip & p.estimated_woba_using_speedangle.notna(), p.rv - RVx(xw), 0.0)
p["TOTAL"] = p.rv - p.rv_all
PARTS = ["DECISIONS", "WHIFFS", "INPUTS", "SQUARING", "LUCK"]
p["OTHER"] = p.TOTAL - p[PARTS].sum(1)
PARTS = PARTS + ["OTHER"]
NON = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "other_advance")
ev = p.events.fillna("").astype(str)
p["pa"] = (ev != "") & ~ev.str.startswith(NON)
H = (
    p.groupby(["batter", "side", "year"])
    .agg(PA=("pa", "sum"), **{k: (k, "sum") for k in PARTS + ["TOTAL"]})
    .reset_index()
)
H = H[H.PA >= 150].copy()
for k in PARTS + ["TOTAL"]:
    H[k] = H[k] / H.PA * 600
a = H[H.PA >= 300]
nx = a.assign(year=(a.year.astype(int) - 1).astype(str))
pr = a.merge(nx, on=["batter", "side", "year"], suffixes=("", "_n"))
O = pd.read_csv("hitter_diagnosis_intended.csv")
O["year"] = O.year.astype(str)
C = H.merge(O, on=["batter", "side", "year"], suffixes=("", "_orig"))
print(f"\naccounting with V3 inputs vs the paper's version ({len(H)} hitter-seasons, 150+ PA):")
print(f"   {'part':10s} {'share (V3)':>10s} {'year-to-year r (V3)':>20s} {'corr with paper version':>24s}")
for k in ["SQUARING", "INPUTS", "LUCK", "DECISIONS", "OTHER", "WHIFFS"]:
    share = np.cov(a[k], a.TOTAL)[0, 1] / a.TOTAL.var()
    print(f"   {k:10s} {100 * share:9.0f}% {pr[k].corr(pr[k + '_n']):20.2f} {C[k].corr(C[k + '_orig']):24.3f}")
H.to_csv("hitter_diagnosis_count.csv", index=False)
