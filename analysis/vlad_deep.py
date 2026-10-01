"""What is Guerrero's problem? Swing path, timing, pitch plane, consistency, and a league contact-quality model.
Contact-quality model: gradient boosting trained on every other hitter's balls in play (2024-26) predicts exit velocity and
xwOBA on contact from the SWING (bat speed, swing length, attack angle, swing-path tilt, attack direction), the TIMING
(where the ball is met: in front of / beside the body) and the PITCH (location, speed, movement, descent angle, type).
Vlad actual vs predicted by year: if prediction falls with him, his measured inputs explain the drop; the rest is
how squarely he meets the ball given those inputs. Mean-shift counterfactuals: move one input group back to 2024."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from stance_lib import pitches
from physics import state_at_y, Y_FRONT

V = 665489
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
]
for c in num:
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
b = p[(p.type == "X") & (p.bat_speed >= 50)].dropna(subset=num).copy()
st = state_at_y(b, Y_FRONT)
b["vaa"] = np.degrees(np.arctan2(st["vz"], -st["vy"]))  # negative = descending
sg = np.where(b.stand == "R", 1.0, -1.0)
b["loc_x"] = b.plate_x * sg  # + = away from the hitter
b["mov_x"] = b.pfx_x * 12 * sg
b["mov_z"] = b.pfx_z * 12
b["contact_y"] = b.intercept_ball_minus_batter_pos_y_inches  # + = met farther out in front
b["contact_x"] = b.intercept_ball_minus_batter_pos_x_inches
b["mismatch"] = b.attack_angle + b.vaa  # 0 = bat rising at the angle the ball falls
b["cls"] = np.select(
    [b.pitch_type.isin(["FF", "SI", "FC"]), b.pitch_type.isin(["SL", "ST", "CU", "KC", "SV", "CS"])], [0, 1], 2
)
b["xwc"] = b.estimated_woba_using_speedangle * 1000
# ---- attack direction sign check (league): does it go negative when contact is farther out front (= pull)?
for s in ("R", "L"):
    z = b[b.stand == s]
    print(f"attack direction vs contact point out front ({s}HB): corr {z.attack_direction.corr(z.contact_y):+.2f}")
v = b[b.batter == V]
lg = b[b.batter != V]
print("\n=== Guerrero by season (balls in play with bat tracking); league 2026 in last column")


def row(d):
    return pd.Series(
        {
            "n": len(d),
            "bat speed": d.bat_speed.mean(),
            "bat speed SD": d.bat_speed.std(),
            "attack angle": d.attack_angle.mean(),
            "attack angle SD": d.attack_angle.std(),
            "swing-path tilt": d.swing_path_tilt.mean(),
            "attack direction": d.attack_direction.mean(),
            "contact point out front (in)": d.contact_y.mean(),
            "contact point SD (timing spread)": d.contact_y.std(),
            "plane mismatch (deg)": d.mismatch.mean(),
            "|plane mismatch|": d.mismatch.abs().mean(),
            "EV": d.launch_speed.mean(),
            "launch angle": d.launch_angle.mean(),
            "LA SD": d.launch_angle.std(),
            "topped (LA<-5)%": 100 * (d.launch_angle < -5).mean(),
            "under (LA>45)%": 100 * (d.launch_angle > 45).mean(),
        }
    )


T = v.groupby("year").apply(row).T
T["league 2026"] = row(lg[lg.year == "2026"])
pd.set_option("display.width", 200)
print(T.round(2).to_string())
# timing bins (league quintiles of contact point)
q = lg.contact_y.quantile([0.2, 0.8]).to_numpy()
v["timing"] = pd.cut(v.contact_y, [-99, q[0], q[1], 99], labels=["late (deep)", "on time", "early (out front)"])
print(
    f"\n=== timing (league 20th/80th pct of contact point: {q[0]:.1f} / {q[1]:.1f} in): share of balls in play | xwOBA on contact"
)
print(
    v.groupby(["timing", "year"])
    .apply(lambda z: f"{100*len(z)/len(v[v.year==z.name[1]]):4.1f}% | {z.xwc.mean():4.0f}")
    .unstack()
    .to_string()
)
# ---- league contact-quality model
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
    "cls",
]
GROUPS = {
    "swing path (attack angle, tilt, direction)": ["attack_angle", "swing_path_tilt", "attack_direction"],
    "timing (where he meets the ball)": ["contact_y", "contact_x"],
    "bat speed / length": ["bat_speed", "swing_length"],
    "pitch location & type": ["loc_x", "plate_z", "release_speed", "mov_x", "mov_z", "vaa"],
}
rng = np.random.default_rng(0)
msk = rng.random(len(lg)) < 0.8
print("\n=== league contact-quality model (trained on every other hitter)")
for y, lab in (("launch_speed", "exit velocity"), ("xwc", "xwOBA on contact")):
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=80, random_state=0
    )
    m.fit(lg.loc[msk, F], lg.loc[msk, y])
    te = lg[~msk]
    r2 = 1 - ((te[y] - m.predict(te[F])) ** 2).mean() / te[y].var()
    v[f"pred_{y}"] = m.predict(v[F])
    print(f"   {lab}: held-out R^2 {r2:.2f}")
    t = v.groupby("year").agg(actual=(y, "mean"), predicted=(f"pred_{y}", "mean"))
    t["actual minus predicted"] = t.actual - t.predicted
    print(t.round(1).to_string())
    base = v[v.year == "2026"][F].copy()
    ref = v[v.year == "2024"][F].mean()
    cur = base.mean()
    p0 = m.predict(base).mean()
    for gname, cols in GROUPS.items():
        x = base.copy()
        for c in cols:
            x[c] = x[c] + (ref[c] - cur[c])
        print(f"      2026 with {gname} moved back to his 2024 average: predicted {m.predict(x).mean() - p0:+.1f}")
v.to_parquet("vlad_bip.parquet")
