"""v2 perception features: depth AND lateral position enter through physics, and the plate-anchored zone is kept
separate from the body-anchored view. Hitter-centric coordinates: x > 0 = away from the hitter (toward the far
edge of the plate), plate center x = 0.
Hitter geometry (inches in, feet out): body x = -(8.5 + off_plate)/12; eye x = body x + 3/12 (head over the plate);
eye y = front of plate - depth/12; eye z = 0.81 * height (EYE=height, original) or, with EYE=posture, 0.81 * height
+ EYE_LEAN * (zone top - 0.569 * height): the hitter-season median zone top (Hawk-Eye, set from the body in 2024-25)
carries posture, and 0.569 is its mean share of height, so the average eye stays at 0.81 * height; EYE_LEAN = inches
the eye moves per inch of zone top (1 = knee bend, ~2 = trunk lean). contact plane y = eye y + body_offset/12;
decision = TAU before the ball reaches the contact plane.
ZONE (never moves with the hitter): plate_xh, plate_zn (crossing at front of plate vs median zone).
VIEW at decision: ball - eye (dx, dy, dz), angular velocity at the eye (deg/s), velocity (vx, vz),
  elapsed time, separation from the pitcher's primary-fastball path (x, z, y).
CONTACT: lateral distance of the ball from the body at the contact plane (reach), ball height at contact vs eye.
Counterfactual = rebuild with depth + dD, off_plate + dL and eye height + dE (inches)."""

import os, numpy as np, pandas as pd
from features import t_at_y, pos_at_t, P9, Y_FRONT, TAU

F2 = [
    "plate_xh",
    "plate_zn",
    "v_dx",
    "v_dy",
    "v_dz",
    "v_omega",
    "v_vx",
    "v_vz",
    "v_elapsed",
    "sep_x",
    "sep_z",
    "sep_y",
    "c_reach",
    "c_dz",
]
EYE_MODE = os.environ.get("EYE", "height")
EYE_LEAN = float(os.environ.get("EYE_LEAN", "1"))
TOP_RATIO = 0.569


def eye_z_ft(df, d_eye_in=0.0):
    z = 0.81 * df.height_in.to_numpy(float) / 12.0
    if EYE_MODE == "posture":
        z = z + EYE_LEAN * (df.sz_top_m.to_numpy(float) - TOP_RATIO * df.height_in.to_numpy(float) / 12.0)
    return z + d_eye_in / 12.0 if d_eye_in else z


def build2(df, depth_in, off_in, fb, d_eye_in=0.0):
    P = {c: df[c].to_numpy(float) for c in P9}
    sgn = np.where(df.stand.to_numpy() == "R", 1.0, -1.0)  # RHB stands at -x in Statcast coords -> away = +x
    body_x = -(8.5 + off_in) / 12.0
    eye_x = body_x + 3 / 12
    eye_y = Y_FRONT - depth_in / 12.0
    eye_z = eye_z_ft(df, d_eye_in)
    y_c = eye_y + df.body_offset_in.to_numpy(float) / 12.0
    t_c = t_at_y(P["vy0"], P["ay"], y_c)
    t_d = t_c - TAU
    t_rel = t_at_y(P["vy0"], P["ay"], df.release_pos_y.to_numpy(float))
    xd, yd, zd, vxd, vyd, vzd = pos_at_t(P, t_d)
    xc, _, zc, _, _, _ = pos_at_t(P, t_c)
    xd_h, vxd_h, xc_h = sgn * xd, sgn * vxd, sgn * xc
    r = np.stack([xd_h - eye_x, yd - eye_y, zd - eye_z])
    v = np.stack([vxd_h, vyd, vzd])
    omega = np.degrees(np.linalg.norm(np.cross(r.T, v.T), axis=1) / np.linalg.norm(r, axis=0) ** 2)
    elapsed = t_d - t_rel
    Fb = fb.loc[list(zip(df.pitcher, df.game_year))]
    PF = {c: Fb[c].to_numpy(float) for c in P9}
    xf, yf, zf, _, _, _ = pos_at_t(PF, t_at_y(PF["vy0"], PF["ay"], Fb.release_pos_y.to_numpy(float)) + elapsed)
    zb, zt = df.sz_bot_m.to_numpy(float), df.sz_top_m.to_numpy(float)
    return pd.DataFrame(
        dict(
            plate_xh=sgn * df.plate_x.to_numpy(float),
            plate_zn=(df.plate_z.to_numpy(float) - zb) / (zt - zb),
            v_dx=r[0],
            v_dy=r[1],
            v_dz=r[2],
            v_omega=omega,
            v_vx=vxd_h,
            v_vz=vzd,
            v_elapsed=elapsed,
            sep_x=sgn * (xd - xf),
            sep_z=zd - zf,
            sep_y=yd - yf,
            c_reach=xc_h - body_x,
            c_dz=zc - eye_z,
        ),
        index=df.index,
    )


def prepare():
    df = pd.read_parquet("pitches_feat.parquet")
    df = df[
        (df.d_x.abs() < 5)
        & (df.c_x.abs() < 5)
        & df.c_zn.between(-3, 5)
        & df.d_elapsed.between(0.1, 0.6)
        & (df.sep_y.abs() < 20)
    ].copy()
    s = pd.read_parquet("../stance_adj.parquet")
    off_d = s[["bsy", "game_date", "off_plate"]].rename(columns={"off_plate": "off_day"})
    off_s = s.groupby("bsy").off_plate.mean().rename("off_season").reset_index()
    df = df.merge(off_s, on="bsy").merge(off_d, on=["bsy", "game_date"], how="left")
    df["off_plate"] = df.off_day.fillna(df.off_season)
    df = df.merge(pd.read_csv("../bio_all.csv")[["batter", "height_in"]], on="batter")
    df["rv"] = df.delta_run_exp.astype(float)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    fb = pd.read_parquet("fb_ref.parquet")
    df = prepare()
    f = build2(df, df.depth.to_numpy(float), df.off_plate.to_numpy(float), fb)
    print(len(df), "pitches")
    print(f.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    g = (
        pd.concat([f, df[["pitch_type"]]], axis=1)
        .groupby("pitch_type")[["v_omega", "c_reach"]]
        .median()
        .loc[["FF", "SI", "SL", "ST", "CH", "CU"]]
    )
    print(g.round(2).to_string())
    out = pd.concat([df.drop(columns=[c for c in f.columns if c in df.columns]), f], axis=1)
    out.to_parquet("pitches_feat2.parquet", index=False)
