"""Do hitters already stand at different depths vs same- vs opposite-hand pitchers, and does it pay?

diff = depth vs SAME-hand pitchers - depth vs OPPOSITE-hand (inches; + = deeper vs same-hand).
Reliability: correlation of a hitter's diff between consecutive seasons (a real habit repeats; noise doesn't).
Payoff: hitter's platoon gap (same-hand minus opposite-hand) in chase, whiff, xwOBA-on-contact, run value
vs diff, controlling for overall depth and batter side (cross-section), and within hitter across seasons.
Hitters with >= 60 swings vs each hand."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

rng = np.random.default_rng(3)
st = []
for y in ("2024", "2025", "2026"):
    for h in ("L", "R"):
        s = pd.read_csv(f"hand_{y}_{h}.csv", encoding="utf-8-sig")
        s["year"] = y
        s["p_throws"] = h
        st.append(s)
st = pd.concat(st).rename(columns={"id": "batter", "avg_batter_y_position": "depth"})
st["same"] = np.where(st.side == st.p_throws, "same", "opp")
w = st.pivot_table(index=["batter", "name", "side", "year"], columns="same", values="depth").dropna().reset_index()
p = pitches()
p["same"] = np.where(p.stand == p.p_throws, "same", "opp")
p["side"] = p.stand
g = p.groupby(["batter", "side", "year", "same"])
o = pd.DataFrame(
    {
        "swings": g.swing.sum(),
        "whiffs": g.whiff.sum(),
        "oz": g.out_zone.sum(),
        "oz_sw": (p.swing & p.out_zone).groupby([p.batter, p.side, p.year, p.same]).sum(),
        "rv": g.delta_run_exp.sum(),
        "n": g.size(),
        "xw": p[(p.type == "X")].groupby(["batter", "side", "year", "same"]).estimated_woba_using_speedangle.mean(),
    }
).reset_index()
o["chase"] = 100 * o.oz_sw / o.oz
o["whiff"] = 100 * o.whiffs / o.swings
o["rv100"] = 100 * o.rv / o.n
o["xwcon"] = 1000 * o.xw
ow = o.pivot_table(
    index=["batter", "side", "year"], columns="same", values=["swings", "chase", "whiff", "rv100", "xwcon"]
).reset_index()
ow.columns = ["_".join(c).strip("_") for c in ow.columns]
d = w.merge(ow, on=["batter", "side", "year"])
d = d[(d.swings_same >= 60) & (d.swings_opp >= 60)].copy()
d["diff"] = d["same"] - d["opp"]
d["depth_all"] = (d["same"] + d["opp"]) / 2
for k in ("chase", "whiff", "rv100", "xwcon"):
    d[f"gap_{k}"] = d[f"{k}_same"] - d[f"{k}_opp"]
print(f"hitter-seasons with >= 60 swings vs each hand: {len(d)}")
print(
    f"diff (deeper vs same-hand, in): mean {d['diff'].mean():+.2f}, SD {d['diff'].std():.2f}, "
    f"|diff| >= 2 in: {(d['diff'].abs() >= 2).mean():.0%}, >= 3 in: {(d['diff'].abs() >= 3).mean():.0%}"
)
print("  by side:", d.groupby("side")["diff"].agg(["mean", "std", "size"]).round(2).to_dict("index"))
nx = d.assign(year=(d.year.astype(int) - 1).astype(str))
pair = d.merge(
    nx[["batter", "side", "year", "diff", "gap_chase", "gap_whiff", "gap_rv100", "gap_xwcon"]],
    on=["batter", "side", "year"],
    suffixes=("", "_next"),
)
print(
    f"\nreliability: season-to-season correlation of a hitter's diff = {pair['diff'].corr(pair.diff_next):+.2f} (n {len(pair)})"
    f"   [overall depth for comparison: r = 0.88]"
)


def ols(y, X, data):
    Xm = np.column_stack([np.ones(len(data))] + [data[c] for c in X])
    yy = data[y].to_numpy()
    b = np.linalg.lstsq(Xm, yy, rcond=None)[0]
    u = yy - Xm @ b
    XtXi = np.linalg.inv(Xm.T @ Xm)
    V = XtXi @ (Xm.T * u**2) @ Xm @ XtXi * len(yy) / (len(yy) - Xm.shape[1])
    return b[1], np.sqrt(V[1, 1])


d["is_R"] = (d.side == "R").astype(float)
d["depth_ft"] = d.depth_all / 12
d["diff_ft"] = d["diff"] / 12
print("\npayoff, cross-section: platoon gap (same - opp) per ft deeper vs SAME-hand than vs opposite-hand")
print("   (chase/whiff: - = helps vs same-hand; xwOBA-con / run value: + = helps)")
for k, u in (("chase", "pp"), ("whiff", "pp"), ("xwcon", "pts"), ("rv100", "runs/100")):
    b, se = ols(f"gap_{k}", ["diff_ft", "depth_ft", "is_R"], d)
    print(f"   {k:6s} {b:+7.2f} {u} (SE {se:.2f}, z {b/se:+.2f})")
pair["d_diff_ft"] = (pair.diff_next - pair["diff"]) / 12
print("\npayoff, within hitter: change in platoon gap vs change in diff (consecutive seasons)")
for k, u in (("chase", "pp"), ("whiff", "pp"), ("xwcon", "pts"), ("rv100", "runs/100")):
    pair["dg"] = pair[f"gap_{k}_next"] - pair[f"gap_{k}"]
    b, se = ols("dg", ["d_diff_ft"], pair)
    print(f"   {k:6s} {b:+7.2f} {u} (SE {se:.2f}, z {b/se:+.2f})")
print("\nlargest habitual splitters (|diff| >= 3 in both of two consecutive seasons):")
both = pair[
    (pair["diff"].abs() >= 3) & (pair.diff_next.abs() >= 3) & (np.sign(pair["diff"]) == np.sign(pair.diff_next))
]
print(both[["name", "side", "year", "diff", "diff_next"]].round(1).to_string(index=False) if len(both) else "  none")
d.to_csv("hand_depth.csv", index=False)
