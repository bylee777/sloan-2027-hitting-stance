"""Hitter setup/movement taxonomy, 2024-26 hitter-seasons (>= 300 tracked swings).
Stance language = START (depth, off-plate, width, stance angle, front/back foot turn at setup)
                + MOVE (stride length, stride direction, landing width, front-foot turn at landing, back-foot travel).
Gaussian mixture on robust-scaled features; k by BIC (2-10) + interpretability. Archetypes described by swing
(bat speed, swing length, attack angle/direction, path tilt, contact point) and outcomes; year-to-year stability."""

import sys, numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from sklearn.mixture import GaussianMixture
from stance_lib import pitches

g = pd.read_parquet("taxonomy/geometry_all.parquet")
p = pitches()
p["bsy"] = p.batter.astype(str) + p.side + p.year
sw = p[p.bat_speed.astype(float) >= 50]
ad = sw.attack_direction.astype(float)
print("attack_direction mean by side (sign check):", sw.assign(ad=ad).groupby("side").ad.mean().round(2).to_dict())
swing = (
    sw.groupby(["batter", "side", "year"])
    .agg(
        swings=("bat_speed", "size"),
        bat_speed=("bat_speed", "mean"),
        swing_len=("swing_length", "mean"),
        attack_angle=("attack_angle", "mean"),
        attack_dir=("attack_direction", "mean"),
        path_tilt=("swing_path_tilt", "mean"),
    )
    .reset_index()
)
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
pa = p[p.events.notna() & ~p.events.fillna("").astype(str).str.startswith(NON_PA)].copy()
pa["same"] = pa.stand == pa.p_throws
pa["xw"] = np.where(
    pa.estimated_woba_using_speedangle.notna(),
    pa.estimated_woba_using_speedangle.astype(float),
    pa.woba_value.astype(float),
)


def outc(d):
    oz, s_ = d[d.out_zone], d[d.swing]
    bip = d[(d.type == "X") & d.launch_speed.notna()]
    return pd.Series(
        {
            "chase": (oz.swing.mean()),
            "whiff": s_.whiff.mean(),
            "zone_contact": 1 - d[d.swing & d.in_zone].whiff.mean(),
            "ev": bip.launch_speed.astype(float).mean(),
            "sweet": bip.launch_angle.between(8, 32).mean(),
            "xwoba_con": bip.estimated_woba_using_speedangle.astype(float).mean(),
        }
    )


o1 = p.groupby(["batter", "side", "year"]).apply(outc).reset_index()
o2 = (
    pa.groupby(["batter", "side", "year"])
    .agg(
        PA=("xw", "size"),
        K=("events", lambda e: e.isin(["strikeout", "strikeout_double_play"]).mean()),
        BB=("events", lambda e: e.isin(["walk", "intent_walk"]).mean()),
        xwoba=("xw", "mean"),
    )
    .reset_index()
)
plat = (
    pa.groupby(["batter", "side", "year", "same"])
    .xw.mean()
    .unstack()
    .rename(columns={True: "xw_same", False: "xw_opp"})
    .reset_index()
)
H = (
    g.merge(swing, on=["batter", "side", "year"])
    .merge(o1, on=["batter", "side", "year"])
    .merge(o2, on=["batter", "side", "year"])
    .merge(plat, on=["batter", "side", "year"], how="left")
)
H = H.merge(pd.read_csv("bio_all.csv")[["batter", "height_in"]], on="batter", how="left")
H = H[H.swings >= 300].reset_index(drop=True)
H["platoon_gap"] = H.xw_opp - H.xw_same
START = ["depth_in", "off_plate_in", "width_in", "stance_angle", "front_turn0", "back_turn0"]
MOVE = ["stride_len_in", "stride_dir", "width_land_in", "front_turn2", "back_foot_move_in"]
F = START + MOVE
X = H[F].copy()
for c in F:
    lo, hi = X[c].quantile([0.01, 0.99])
    X[c] = X[c].clip(lo, hi)
Z = ((X - X.median()) / (X.quantile(0.75) - X.quantile(0.25))).to_numpy()
print(f"\n{len(H)} hitter-seasons (>= 300 tracked swings)")
bic = {}
for k in range(2, 11):
    bic[k] = np.mean(
        [GaussianMixture(k, covariance_type="full", n_init=3, random_state=s).fit(Z).bic(Z) for s in range(3)]
    )
print("BIC by k:", {k: round(v) for k, v in bic.items()})
pd.to_pickle((H, Z, F, bic), "taxonomy/_stage.pkl")
