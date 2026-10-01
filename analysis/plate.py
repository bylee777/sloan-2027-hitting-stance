"""Plate distance (distance off the plate) and handedness, 2024-26, clean zone. close_ft = -(off-plate - mean)/12,
so + = standing CLOSER to the plate. Depth kept in every model (both are hitter-season constants).
A. Same-hand penalty per ft closer: K types per PA by prior-chase tercile (batter-season + pitcher-season FE).
B. Mechanism, pitch level: whiff per swing and xwOBA on contact on AWAY vs IN pitches (hitter-centric thirds of
   the plate) relative to MIDDLE, per ft closer (batter-season FE absorbs position; count x zone x class FE).
C. Who benefits: same-hand run value per ft closer / deeper interacted with height, reach, exit velocity (z-scored).
   reach = 90th pct of lateral contact distance from the body (bat tracking) = usable arm/bat reach."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from fe import feols
from stance_lib import pitches

p = pitches()
p["bsy"] = p.batter.astype(str) + p.side + p.year
s = pd.read_parquet("stance_adj.parquet")
hs = (
    s.groupby("bsy")
    .agg(
        batter=("batter", "first"), depth=("depth_adj", "mean"), off_plate=("off_plate", "mean"), days=("depth", "size")
    )
    .reset_index()
)
con = p[p.description.isin(["foul", "hit_into_play"]) & p.intercept_ball_minus_batter_pos_x_inches.notna()]
hs = hs.merge(
    con.groupby("bsy").intercept_ball_minus_batter_pos_x_inches.quantile(0.9).rename("reach").reset_index(),
    on="bsy",
    how="left",
)
hs = hs.merge(p[p.type == "X"].groupby("bsy").launch_speed.mean().rename("ev").reset_index(), on="bsy", how="left")
hs = hs.merge(pd.read_csv("bio_all.csv")[["batter", "height_in"]], on="batter", how="left")
hs = hs[hs.days >= 20].dropna()
print(
    f"hitter-seasons {len(hs)}; off-plate mean {hs.off_plate.mean():.1f} in (SD {hs.off_plate.std():.1f}); "
    f"reach p90 mean {hs.reach.mean():.1f} in (SD {hs.reach.std():.1f}); corr(height, reach) {hs.height_in.corr(hs.reach):+.2f}; "
    f"corr(off-plate, reach) {hs.off_plate.corr(hs.reach):+.2f}; corr(depth, off-plate) {hs.depth.corr(hs.off_plate):+.2f}"
)
for c in ("height_in", "reach", "ev"):
    hs[c + "_z"] = (hs[c] - hs[c].mean()) / hs[c].std()
hs["close_ft"] = -(hs.off_plate - hs.off_plate.mean()) / 12
hs["deep_ft"] = (hs.depth - hs.depth.mean()) / 12
p = p.merge(hs[["bsy", "close_ft", "deep_ft", "height_in_z", "reach_z", "ev_z"]], on="bsy")
p["same"] = (p.stand == p.p_throws).astype(float)

# ---------------- A. K types
prior = (
    p.groupby(["batter", "side", "year"])
    .apply(lambda d: (d.swing & d.out_zone).sum() / max(d.out_zone.sum(), 1))
    .rename("prior_chase")
    .reset_index()
)
prior["year"] = (prior.year.astype(int) + 1).astype(str)
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
pa = p[p.events.notna() & ~p.events.fillna("").astype(str).str.startswith(NON_PA)].copy()
K = pa.events.isin(["strikeout", "strikeout_double_play"])
swK = K & pa.description.isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
pa["K"] = K * 100.0
pa["K_chase"] = (swK & pa.out_zone) * 100.0
pa["K_zone_whiff"] = (swK & ~pa.out_zone) * 100.0
pa["K_looking"] = (K & (pa.description == "called_strike")) * 100.0
pa["psy"] = pa.pitcher.astype(str) + pa.year
pa["close_same"] = pa.close_ft * pa.same
pa["deep_same"] = pa.deep_ft * pa.same
pa2 = pa.merge(prior, on=["batter", "side", "year"])
cut = pa2.groupby("bsy").prior_chase.first().quantile([1 / 3, 2 / 3]).to_numpy()
pa2["terc"] = np.select([pa2.prior_chase < cut[0], pa2.prior_chase < cut[1]], ["low chase", "mid chase"], "HIGH chase")
print("\nA. per ft CLOSER to the plate: change in the same-hand strikeout penalty (pp per PA)  [- = helps]")
for lab, d in (
    ("ALL 2024-26", pa),
    ("low chase", pa2[pa2.terc == "low chase"]),
    ("mid chase", pa2[pa2.terc == "mid chase"]),
    ("HIGH chase", pa2[pa2.terc == "HIGH chase"]),
):
    cells = []
    for k in ("K", "K_chase", "K_zone_whiff", "K_looking"):
        b, V = feols(d, k, ["same", "close_same", "deep_same"], ["bsy", "psy"], "batter")
        se = np.sqrt(np.diag(V))
        cells.append(f"{k.replace('K_', '')} {b['close_same']:+.2f} (z {b['close_same']/se[1]:+.1f})")
    print(f"   {lab:11s} " + " | ".join(cells))

# ---------------- B. location mechanism
sgn = np.where(p.stand == "R", 1.0, -1.0)
xa = p.plate_x.astype(float) * sgn  # + = away from the hitter
p["band"] = np.select([xa > 0.236, xa < -0.236], ["away", "in"], "middle")
p["cls"] = np.select(
    [
        p.pitch_type.isin(["FF", "SI", "FC"]),
        p.pitch_type.isin(["SL", "ST", "CU", "KC", "SV"]),
        p.pitch_type.isin(["CH", "FS"]),
    ],
    ["FB", "BRK", "OFF"],
    "X",
)
p = p[p.cls != "X"]
p["count"] = p.balls.astype(str) + p.strikes.astype(str)
p["zone_i"] = p.zone.fillna(0).astype(int)
p["close_away"] = p.close_ft * (p.band == "away")
p["close_in"] = p.close_ft * (p.band == "in")
p["whiff_f"] = p.whiff.astype(float)
p["xw"] = p.estimated_woba_using_speedangle
print("\nB. per ft CLOSER: outcome on AWAY and IN pitches relative to MIDDLE pitches (hitter-centric thirds)")
print(
    f"   share of same-hand pitches away/in: {(p[p.same==1].band=='away').mean():.0%}/{(p[p.same==1].band=='in').mean():.0%};"
    f" opposite-hand: {(p[p.same==0].band=='away').mean():.0%}/{(p[p.same==0].band=='in').mean():.0%}"
)
for lab, d, y, sc in (
    ("whiff per swing (pp; - helps)", p[p.swing], "whiff_f", 100),
    ("xwOBA on contact (pts; + helps)", p[(p.type == "X") & p.xw.notna()], "xw", 1000),
):
    b, V = feols(
        d, y, ["close_away", "close_in"], ["bsy", ("cls", "count", "year"), ("cls", "zone_i", "year")], "batter"
    )
    se = np.sqrt(np.diag(V))
    print(
        f"   {lab:32s} AWAY {b['close_away']*sc:+.2f} (z {b['close_away']/se[0]:+.1f}) | IN {b['close_in']*sc:+.2f} (z {b['close_in']/se[1]:+.1f})"
    )

# ---------------- C. heterogeneity: same-hand run value per ft closer / deeper x traits
p["rv"] = p.delta_run_exp.astype(float) * 100
p["close_same"] = p.close_ft * p.same
p["deep_same"] = p.deep_ft * p.same
print("\nC. same-hand run value per 100 pitches (+ helps): base effect and how it changes per 1 SD of each trait")
for tr in ("height_in_z", "reach_z", "ev_z"):
    d = p.dropna(subset=["rv"]).copy()
    d["same_tr"] = d.same * d[tr]
    d["close_same_tr"] = d.close_same * d[tr]
    d["deep_same_tr"] = d.deep_same * d[tr]
    cols = ["close_same", "deep_same", "same_tr", "close_same_tr", "deep_same_tr"]
    b, V = feols(d, "rv", cols, ["bsy", ("same", "cls", "count", "year"), ("same", "cls", "zone_i", "year")], "batter")
    se = pd.Series(np.sqrt(np.diag(V)), index=cols)
    print(
        f"   {tr[:-2]:9s} closer: {b['close_same']:+.2f} (z {b['close_same']/se['close_same']:+.1f}), x trait {b['close_same_tr']:+.2f} (z {b['close_same_tr']/se['close_same_tr']:+.1f}) | "
        f"deeper: {b['deep_same']:+.2f} (z {b['deep_same']/se['deep_same']:+.1f}), x trait {b['deep_same_tr']:+.2f} (z {b['deep_same_tr']/se['deep_same_tr']:+.1f})"
    )
hs.to_csv("hitter_traits.csv", index=False)
