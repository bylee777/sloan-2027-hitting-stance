"""Vladimir Guerrero Jr. 2024-26: what changed in 2026? Results, luck, swing decisions, contact, bat tracking,
pitch-type splits, how pitchers attacked him, stance, and the 2026 month-by-month path."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

V = 665489
p = pitches()
for c in (
    "launch_speed",
    "launch_angle",
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "release_speed",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "intercept_ball_minus_batter_pos_y_inches",
    "plate_x",
    "plate_z",
):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
v = p[p.batter == V].copy()
ev = v.events.fillna("").astype(str)
v["pa"] = (ev != "") & ~ev.str.startswith(
    ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "other_advance")
)
v["bip"] = v.type == "X"
v["cls"] = np.select(
    [v.pitch_type.isin(["FF", "SI", "FC"]), v.pitch_type.isin(["SL", "ST", "CU", "KC", "SV", "CS"])],
    ["fastball", "breaking"],
    "offspeed",
)


def summ(d):
    pa, bip = d[d.pa], d[d.bip]
    wd = pa.woba_denom.fillna(0)
    sw = d[d.swing.astype(bool)]
    comp = sw[sw.bat_speed >= 50]
    hits = pa.events.isin(["single", "double", "triple", "home_run"])
    babip_den = (bip.events != "home_run").sum()
    return pd.Series(
        {
            "PA": len(pa),
            "wOBA": 1000 * pa.woba_value.sum() / wd.sum(),
            "xwOBA": 1000
            * np.nansum(
                np.where(
                    pa.estimated_woba_using_speedangle.notna(),
                    pa.estimated_woba_using_speedangle,
                    pa.woba_value * (wd > 0),
                )
            )
            / wd.sum(),
            "HR": pa.events.eq("home_run").sum(),
            "K%": 100 * pa.events.str.startswith("strikeout").mean(),
            "BB%": 100 * pa.events.isin(["walk"]).mean(),
            "BABIP": 1000 * (bip.events.isin(["single", "double", "triple"])).sum() / babip_den,
            "chase%": 100 * d[d.out_zone].swing.mean(),
            "zone swing%": 100 * d[d.in_zone].swing.mean(),
            "whiff%": 100 * sw.whiff.mean(),
            "exit velo": bip.launch_speed.mean(),
            "hard-hit%": 100 * (bip.launch_speed >= 95).mean(),
            "sweet-spot%": 100 * bip.launch_angle.between(8, 32).mean(),
            "launch angle": bip.launch_angle.mean(),
            "ground-ball%": 100 * (bip.bb_type == "ground_ball").mean(),
            "fly+line%": 100 * bip.bb_type.isin(["fly_ball", "line_drive"]).mean(),
            "bat speed": comp.bat_speed.mean(),
            "swing length": comp.swing_length.mean(),
            "attack angle": comp.attack_angle.mean(),
            "attack direction": comp.attack_direction.mean(),
            "contact point (in, + = out front)": comp.intercept_ball_minus_batter_pos_y_inches.mean(),
            "xwOBA on contact": 1000 * bip.estimated_woba_using_speedangle.mean(),
        }
    )


T = v.groupby("year").apply(summ).T
pd.set_option("display.width", 200)
print("=== Vladimir Guerrero Jr., by season")
print(T.round(1).to_string())
# league context by year (regulars) for the same metrics
lg = p.groupby("year").apply(
    lambda d: pd.Series(
        {
            "chase%": 100 * d[d.out_zone].swing.mean(),
            "whiff%": 100 * d[d.swing.astype(bool)].whiff.mean(),
            "bat speed": d[d.swing.astype(bool) & (d.bat_speed >= 50)].bat_speed.mean(),
            "exit velo": d[d.type == "X"].launch_speed.mean(),
        }
    )
)
print("\nleague by year:")
print(lg.round(1).T.to_string())
print("\n=== by pitch class: xwOBA on contact | whiff% | share of pitches seen")
for c in ("fastball", "breaking", "offspeed"):
    d = v[v.cls == c]
    row = d.groupby("year").apply(
        lambda z: f"{1000*z[z.bip].estimated_woba_using_speedangle.mean():4.0f} | {100*z[z.swing.astype(bool)].whiff.mean():4.1f}% | {100*len(z)/len(v[v.year==z.name]):4.1f}%"
    )
    print(f"   {c:9s} " + "   ".join(f"{y}: {s}" for y, s in row.items()))
# fastball velocity bands
fb = v[v.cls == "fastball"].copy()
fb["band"] = pd.cut(fb.release_speed, [0, 94, 97, 110], labels=["<94", "94-97", "97+"])
print("\n=== fastballs by velocity: xwOBA on contact (n BIP) / whiff%")
print(
    fb.groupby(["band", "year"])
    .apply(
        lambda z: f"{1000*z[z.bip].estimated_woba_using_speedangle.mean():.0f} ({z.bip.sum()}) / {100*z[z.swing.astype(bool)].whiff.mean():.0f}%"
    )
    .unstack()
    .to_string()
)
# where pitchers located to him: inside/away, up/down
sgn = 1.0  # RHB: + plate_x = away? For RHB, positive plate_x is AWAY (catcher's view)
v["away"] = v.plate_x > 0.236
v["inside"] = v.plate_x < -0.236
v["up"] = v.plate_z > 3.0
v["down"] = v.plate_z < 2.0
print("\n=== how pitchers attacked him (% of pitches) and his xwOBA on contact there")
for loc in ("inside", "away", "up", "down"):
    print(
        f"   {loc:7s} "
        + "   ".join(
            f"{y}: {100*g[loc].mean():4.1f}% seen, xwOBAcon {1000*g[g[loc] & g.bip].estimated_woba_using_speedangle.mean():.0f}"
            for y, g in v.groupby("year")
        )
    )
# stance by season
s = pd.read_parquet("stance_adj.parquet")
sv = s[s.batter == V]
H = pd.read_parquet("taxonomy/hitters_pca.parquet")
hv = H[H.batter == V]
print("\n=== stance by season")
print(sv.groupby("year")[["depth_adj", "off_plate_adj", "width", "angle"]].mean().round(1).to_string())
print(
    hv.set_index("year")[["stride_len_in", "width_land_in", "front_turn0", "front_turn2", "contact_vs_plate_in"]]
    .round(1)
    .to_string()
)
# 2026 month by month vs 2025
v["month"] = v.game_date.str[5:7]
M = (
    v[v.year.isin(["2025", "2026"])]
    .groupby(["year", "month"])
    .apply(
        lambda d: pd.Series(
            {
                "PA": d.pa.sum(),
                "wOBA": 1000 * d[d.pa].woba_value.sum() / d[d.pa].woba_denom.fillna(0).sum(),
                "xwOBA_con": 1000 * d[d.bip].estimated_woba_using_speedangle.mean(),
                "EV": d[d.bip].launch_speed.mean(),
                "LA": d[d.bip].launch_angle.mean(),
                "bat speed": d[d.swing.astype(bool) & (d.bat_speed >= 50)].bat_speed.mean(),
                "chase%": 100 * d[d.out_zone].swing.mean(),
            }
        )
    )
)
print("\n=== month by month")
print(M.round(1).to_string())
sd = sv[sv.year == "2026"].copy()
sd["month"] = sd.game_date.str[5:7]
print("\n2026 stance by month:")
print(sd.groupby("month")[["depth_adj", "off_plate_adj", "width", "angle"]].mean().round(1).to_string())
# ---- is the weak contact from swinging at worse pitches, or from worse contact on good pitches?
v["heart"] = (v.plate_x.abs() <= 0.56) & v.plate_z.between(2.0, 3.2)
print("\n=== balls in play: where they came from, and contact quality there")
for y, g in v.groupby("year"):
    b = g[g.bip]
    print(
        f"   {y}: BIP on pitches OUTSIDE the zone {100*b.out_zone.mean():4.1f}% (xwOBAcon {1000*b[b.out_zone].estimated_woba_using_speedangle.mean():.0f}) | "
        f"in zone {1000*b[b.in_zone].estimated_woba_using_speedangle.mean():.0f} | heart of the zone {1000*b[b.heart].estimated_woba_using_speedangle.mean():.0f} "
        f"(EV {b[b.heart].launch_speed.mean():.1f}, n {b.heart.sum()}) | heart swing% {100*g[g.heart].swing.mean():.0f}"
    )
