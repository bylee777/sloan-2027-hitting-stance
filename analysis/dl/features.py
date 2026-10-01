"""Perception features for the hitter-decision model (v1).

Each pitch is seen from the hitter's side. Hitter geometry:
  contact plane y_c = front of plate - (depth - body_offset) where contact_vs_plate = intercept_vs_batter - depth
  (both in inches; body_offset = hitter-season mean intercept_y_vs_batter). Moving the hitter back by D moves
  the contact plane back by D (supported by movers: contact point moved 1:1 with depth).
  decision instant = TAU s before the ball reaches y_c.
Features (hitter-centric: x > 0 = AWAY from the hitter):
  at decision: x, z, vx, vz, vy, elapsed time since release, separation from the pitcher's primary-fastball
               mean trajectory at the same elapsed time (dx, dz, dy)
  at contact plane: x, z (z normalised to the hitter's zone: 0 = bottom, 1 = top)
Everything depends on depth only through physics -> counterfactual depth = recompute features.
"""

import glob, numpy as np, pandas as pd
import sys

sys.path.insert(0, "..")
TAU, G = 0.175, 32.174
Y_FRONT = 17 / 12
P9 = ["plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az"]


def t_at_y(vy0, ay, y):
    return (-vy0 - np.sqrt(vy0**2 - 2 * ay * (50.0 - y))) / ay


def pos_at_t(P, t):
    """P: dict of arrays (9P anchored at plate crossing). t: time after the y=50 reference."""
    tp = t_at_y(P["vy0"], P["ay"], Y_FRONT)
    x = P["plate_x"] + P["vx0"] * (t - tp) + 0.5 * P["ax"] * (t**2 - tp**2)
    z = P["plate_z"] + P["vz0"] * (t - tp) + 0.5 * P["az"] * (t**2 - tp**2)
    y = 50.0 + P["vy0"] * t + 0.5 * P["ay"] * t**2
    return x, y, z, P["vx0"] + P["ax"] * t, P["vy0"] + P["ay"] * t, P["vz0"] + P["az"] * t


def build(df, depth_in, fb):
    """df: pitches (with 9P, release_pos_y, stand, sz_top/bot, body_offset_in); depth_in: array of hitter depth
    (inches) to evaluate at; fb: DataFrame indexed by pitcher-year with the primary fastball's mean 9P + release_y."""
    P = {c: df[c].to_numpy(float) for c in P9}
    sgn = np.where(df.stand.to_numpy() == "R", 1.0, -1.0)
    contact_vs_plate_ft = (df.body_offset_in.to_numpy() - depth_in) / 12.0
    y_c = Y_FRONT + contact_vs_plate_ft
    t_c = t_at_y(P["vy0"], P["ay"], y_c)
    t_rel = t_at_y(P["vy0"], P["ay"], df.release_pos_y.to_numpy(float))
    t_d = t_c - TAU
    x_d, y_d, z_d, vx_d, vy_d, vz_d = pos_at_t(P, t_d)
    x_c, _, z_c, _, _, _ = pos_at_t(P, t_c)
    elapsed = t_d - t_rel
    F = fb.loc[list(zip(df.pitcher, df.game_year))]
    PF = {c: F[c].to_numpy(float) for c in P9}
    tf_rel = t_at_y(PF["vy0"], PF["ay"], F.release_pos_y.to_numpy(float))
    xf, yf, zf, _, _, _ = pos_at_t(PF, tf_rel + elapsed)
    zb, zt = df.sz_bot_m.to_numpy(float), df.sz_top_m.to_numpy(
        float
    )  # hitter-season MEDIAN zone (per-pitch sz leaks the swing)
    return pd.DataFrame(
        dict(
            d_x=sgn * x_d,
            d_z=z_d,
            d_vx=sgn * vx_d,
            d_vz=vz_d,
            d_vy=vy_d,
            d_elapsed=elapsed,
            sep_x=sgn * (x_d - xf),
            sep_z=z_d - zf,
            sep_y=y_d - yf,
            c_x=sgn * x_c,
            c_zn=(z_c - zb) / (zt - zb),
        ),
        index=df.index,
    )


def load():
    cols = [
        "game_pk",
        "game_date",
        "game_year",
        "pitcher",
        "batter",
        "stand",
        "p_throws",
        "pitch_type",
        "description",
        "zone",
        "balls",
        "strikes",
        "sz_top",
        "sz_bot",
        "release_pos_y",
        "release_speed",
        "release_spin_rate",
        "release_pos_x",
        "release_pos_z",
        "release_extension",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "delta_run_exp",
    ] + P9
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("../raw/*.parquet"))], ignore_index=True)
    df = df.dropna(subset=P9 + ["release_pos_y", "sz_top", "sz_bot", "pitch_type"])
    df = df[~df.description.str.contains("bunt|pitchout", na=False)]
    df["game_date"] = pd.to_datetime(df.game_date).dt.strftime("%Y-%m-%d")
    df["year"] = df.game_year.astype(str)
    df["bsy"] = df.batter.astype(str) + df.stand + df.year
    # hitter geometry: daily depth (park-adjusted) with season-mean fallback; body offset = season mean
    s = pd.read_parquet("../stance_adj.parquet")
    daily = s[["bsy", "game_date", "depth_adj"]]
    seas = (
        s.groupby("bsy")
        .agg(depth_season=("depth_adj", "mean"), body_offset_in=("avg_intercept_y_vs_batter", "mean"))
        .reset_index()
    )
    df = df.merge(seas, on="bsy").merge(daily, on=["bsy", "game_date"], how="left")
    df["depth"] = df.depth_adj.fillna(df.depth_season)
    df = df.dropna(subset=["body_offset_in"])
    # primary fastball (FF or SI, whichever more) mean 9P per pitcher-year
    fbp = df[df.pitch_type.isin(["FF", "SI"])]
    prim = fbp.groupby(["pitcher", "game_year"]).pitch_type.agg(lambda s: s.value_counts().index[0]).rename("fbt")
    fbp = fbp.merge(prim.reset_index(), on=["pitcher", "game_year"])
    fb = fbp[fbp.pitch_type == fbp.fbt].groupby(["pitcher", "game_year"])[P9 + ["release_pos_y"]].mean()
    df = df[df.set_index(["pitcher", "game_year"]).index.isin(fb.index)].reset_index(drop=True)
    WH = {"swinging_strike", "swinging_strike_blocked"}
    df["swing"] = df.description.isin(WH | {"foul", "foul_tip", "hit_into_play"}).astype(np.float32)
    df["whiff"] = df.description.isin(WH).astype(np.float32)
    # clean zone: crossing point vs hitter-season median zone + ball radius (Statcast zone/sz leak the swing)
    df["sz_top_m"] = df.groupby("bsy").sz_top.transform("median")
    df["sz_bot_m"] = df.groupby("bsy").sz_bot.transform("median")
    R = 1.45 / 12
    inz = (df.plate_x.abs() <= 17 / 24 + R) & df.plate_z.between(df.sz_bot_m - R, df.sz_top_m + R)
    df["out_zone"] = ~inz
    df["same"] = (df.stand == df.p_throws).astype(np.float32)
    return df, fb


if __name__ == "__main__":
    import time

    t0 = time.time()
    df, fb = load()
    feat = build(df, df.depth.to_numpy(float), fb)
    out = pd.concat([df, feat], axis=1)
    out.to_parquet("pitches_feat.parquet", index=False)
    fb.to_parquet("fb_ref.parquet")
    print(f"{len(out):,} pitches, {out.bsy.nunique()} hitter-seasons, {time.time()-t0:.0f}s")
    print(feat.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    # sanity: fastballs should have ~0 separation; sweepers large away separation for same-hand
    g = out.groupby([out.pitch_type.where(out.pitch_type.isin(["FF", "SI", "ST", "SL", "CH", "CU"])), "same"])[
        ["sep_x", "sep_z", "sep_y", "c_x"]
    ].mean()
    print(g.round(3).to_string())
