"""Hitter run-value ACCOUNTING: every pitch's run value above league average, split into parts that add up exactly.

Cell c = season x count x attack zone (heart / shadow / chase / waste, from the clean zone) x pitch class.
League values per cell: RV_L(c) (all pitches), RV_L(c, swing), RV_L(c, take), p_whiff_L(c), RV_L(c, whiff), RV_L(c, contact).
Per pitch:  RV - RV_L(c) = DECISIONS  [RV_L(c, action) - RV_L(c)]
                         + WHIFFS     [(whiff - p_whiff_L) x (RV_L(c, whiff) - RV_L(c, contact))]            (swings)
                         + CONTACT    [RV - RV_L(c, contact)]  on balls in play, split into
                               INPUTS      RVx(pred xwOBA) - RV_L(c, in play)   pred = league model of the swing + pitch
                               SQUARING UP RVx(xwOBA) - RVx(pred xwOBA)         how squarely he met it, given the inputs
                               LUCK        RV - RVx(xwOBA)                      result vs expected
                         + OTHER      in-play vs foul mix, fouls, whiff residual, takes (called balls/strikes), untracked balls in play
RVx(.) = league map from xwOBA to run value (per season x count). Pred xwOBA: gradient boosting on bat speed, swing
length, attack angle, swing-path tilt, attack direction, contact point, pitch location / speed / movement / descent angle,
cross-fitted by hitter (a hitter never predicts himself). Output: runs per 600 PA per hitter-season, reliability
(year-to-year r) per part, shrunk values, and each 2026 regular's biggest gaps."""

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
F = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "swing_path_tilt",
    "attack_direction",
    "contact_y",
    "contact_x",
    "loc_x",
    "plate_z",
    "release_speed",
    "mov_x",
    "mov_z",
    "vaa",
    "cls_i",
]
trk = p.bip & p.estimated_woba_using_speedangle.notna() & p[F].notna().all(1) & (p.bat_speed >= 50)
T = p[trk]
pred = np.full(len(T), np.nan)
for tr, te in GroupKFold(5).split(T, groups=T.batter):
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=80, random_state=0
    )
    m.fit(T.iloc[tr][F], T.iloc[tr].estimated_woba_using_speedangle)
    pred[te] = m.predict(T.iloc[te][F])
p["xw_pred"] = np.nan
p.loc[trk, "xw_pred"] = pred
xw = p.estimated_woba_using_speedangle.to_numpy()
p["INPUTS"] = np.where(trk, RVx(p.xw_pred.to_numpy()) - p.rv_bip, 0.0)  # vs league balls in play in the same cell
p["SQUARING"] = np.where(trk, RVx(xw) - RVx(p.xw_pred.to_numpy()), 0.0)
p["LUCK"] = np.where(p.bip & p.estimated_woba_using_speedangle.notna(), p.rv - RVx(xw), 0.0)
p["TOTAL"] = p.rv - p.rv_all
PARTS = ["DECISIONS", "WHIFFS", "INPUTS", "SQUARING", "LUCK"]
p["OTHER"] = p.TOTAL - p[PARTS].sum(1)
PARTS = PARTS + ["OTHER"]
# ---- hitter-season totals per 600 PA
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
print(
    f"hitter-seasons (150+ PA): {len(H)}; parts add up exactly: max |sum - total| = {(H[PARTS].sum(1) - H.TOTAL).abs().max():.1e}"
)
print(
    f"xwOBA-on-contact model (cross-fitted by hitter): R^2 {1 - np.nanmean((xw[trk] - pred) ** 2) / np.nanvar(xw[trk]):.3f}"
)
# ---- reliability: year-to-year correlation (both seasons 300+ PA)
a = H[H.PA >= 300]
nx = a.assign(year=(a.year.astype(int) - 1).astype(str))
pr = a.merge(nx, on=["batter", "side", "year"], suffixes=("", "_n"))
print(f"\n=== what each part is worth and how much of it is SKILL (year-to-year r, {len(pr)} hitter pairs, 300+ PA)")
print(f"{'part':10s} {'SD across hitters':>18s} {'year-to-year r':>15s} {'share of total variance':>24s}")
REL = {}
for k in PARTS + ["TOTAL"]:
    r = pr[k].corr(pr[k + "_n"])
    REL[k] = max(r, 0.0)
    cov = np.cov(a[k], a.TOTAL)[0, 1] / a.TOTAL.var() if k != "TOTAL" else 1.0
    print(f"{k:10s} {a[k].std():15.1f} runs {r:15.2f} {100*cov:22.0f}%")
# ---- shrink toward the league mean by reliability scaled to PA (EB): w = PA / (PA + k), k from r at the median PA
med = pr.PA.median()
for k in PARTS:
    r = min(max(REL[k], 0.01), 0.95)
    kk = med * (1 - r) / r
    w = H.PA / (H.PA + kk)
    H[k + "_s"] = w * H[k]  # league mean of each part is ~0 by construction
H["name"] = H.batter.map(
    pd.read_parquet("stance_adj.parquet", columns=["batter", "name"]).drop_duplicates("batter").set_index("batter").name
)
H.to_csv("hitter_diagnosis.csv", index=False)
p[
    ["batter", "side", "year", "game_date", "cell", "azone", "cls", "cnt", "sw", "wh", "bip", "rv"]
    + PARTS
    + ["TOTAL", "xw_pred"]
].to_parquet("hitter_diagnosis_pitches.parquet")
pd.to_pickle(REL, "hitter_diagnosis_rel.pkl")
S = H[H.year == "2026"].sort_values("TOTAL", ascending=False)
pd.set_option("display.width", 220)
show = ["name", "side", "PA", "TOTAL"] + [k + "_s" for k in PARTS[:-1]]
print("\n=== 2026 top and bottom 8 (runs per 600 PA vs league average; parts shrunk by reliability)")
print(S[show].head(8).round(1).to_string(index=False))
print(S[show].tail(8).round(1).to_string(index=False))
for nm in ("Guerrero Jr., Vladimir", "Crow-Armstrong, Pete", "Clement, Ernie", "Okamoto, Kazuma", "Kirk, Alejandro"):
    z = H[H.name == nm].sort_values("year")
    print(f"\n{nm}:")
    print(z[["year", "PA", "TOTAL"] + PARTS].round(1).to_string(index=False))
