"""Shared loaders for the stance-angle (open/closed) study. Angle sign: NEGATIVE = OPEN, + = closed
(verified from Savant foot coordinates: Albies-L -35 deg has front foot 3.6 ft off the plate vs back 1.7)."""

import glob
import numpy as np, pandas as pd

WH = {"swinging_strike", "swinging_strike_blocked"}
SWING = WH | {"foul", "foul_tip", "hit_into_play"}


def stance(years=("2024", "2025", "2026")):
    out = []
    for y in years:
        for f in sorted(glob.glob(f"stance_daily/{y}-*.csv")):
            s = pd.read_csv(f, encoding="utf-8-sig")
            s["game_date"] = f[-14:-4]
            out.append(s)
    s = pd.concat(out, ignore_index=True).rename(
        columns={
            "id": "batter",
            "avg_batter_y_position": "depth",
            "avg_batter_x_position": "off_plate",
            "avg_foot_sep": "width",
            "avg_stance_angle": "angle",
        }
    )
    s["year"] = s.game_date.str[:4]
    return s


def pitches(years=("2024", "2025", "2026")):
    p = pd.concat(
        [pd.read_parquet(f) for y in years for f in sorted(glob.glob(f"raw/{y}*.parquet"))], ignore_index=True
    )
    p = p[~p.description.str.contains("bunt|pitchout", na=False)].copy()
    p["game_date"] = pd.to_datetime(p.game_date).dt.strftime("%Y-%m-%d")
    p["year"] = p.game_date.str[:4]
    p["side"] = p.stand
    p["swing"] = p.description.isin(SWING)
    p["whiff"] = p.description.isin(WH)
    # Statcast's per-pitch sz_top/sz_bot (and its `zone`, built from them) leak the swing (AUC .785 -> .991),
    # so the zone is rebuilt from the crossing point and the hitter-season MEDIAN zone (+ ball radius).
    p["zone_statcast"] = p.zone
    key = p.batter.astype(str) + p.side + p.year
    for c in ("plate_x", "plate_z", "sz_top", "sz_bot"):
        p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
    top = p.groupby(key).sz_top.transform("median")
    bot = p.groupby(key).sz_bot.transform("median")
    R = 1.45 / 12
    xin = p.plate_x.abs() <= 17 / 24 + R
    zin = p.plate_z.between(bot - R, top + R)
    col = np.clip(((p.plate_x + 17 / 24) / (17 / 36)).fillna(-1).astype(int), 0, 2)
    row = np.clip(((top - p.plate_z) / ((top - bot) / 3)).fillna(-1).astype(int), 0, 2)
    quad = np.where(
        p.plate_x < 0, np.where(p.plate_z >= (top + bot) / 2, 11, 13), np.where(p.plate_z >= (top + bot) / 2, 12, 14)
    )
    p["zone"] = np.where(xin & zin, 1 + 3 * row + col, quad).astype(float)
    p.loc[p.plate_x.isna() | p.plate_z.isna(), "zone"] = np.nan
    p["in_zone"] = p.zone.between(1, 9)
    p["out_zone"] = p.zone.between(11, 14)
    return p


def contact_table(p, keys):
    g = p.groupby(keys)
    t = pd.DataFrame(
        {
            "pitches": g.size(),
            "swings": g.swing.sum(),
            "whiffs": g.whiff.sum(),
            "z_swings": g.apply(lambda d: (d.swing & d.in_zone).sum()),
            "z_whiffs": g.apply(lambda d: (d.whiff & d.in_zone).sum()),
            "o_pitches": g.out_zone.sum(),
            "o_swings": g.apply(lambda d: (d.swing & d.out_zone).sum()),
        }
    )
    t["contact"] = 1 - t.whiffs / t.swings
    t["z_contact"] = 1 - t.z_whiffs / t.z_swings
    t["chase"] = t.o_swings / t.o_pitches
    return t
