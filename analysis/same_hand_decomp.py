"""Where does the same-hand penalty come from? 2024-26, all pitches. By batter side: pitch mix, run value by pitch
class, and a decomposition of the same-minus-opposite run-value gap into mix vs per-pitch results, plus chase /
whiff / contact quality by class."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

p = pitches()
p["same"] = np.where(p.stand == p.p_throws, "same", "opp")
p["cls"] = np.select(
    [
        p.pitch_type.isin(["FF", "SI", "FC"]),
        p.pitch_type.isin(["SL", "ST", "CU", "KC", "SV", "CS"]),
        p.pitch_type.isin(["CH", "FS", "FO", "SC"]),
    ],
    ["fastball", "breaking", "offspeed"],
    "other",
)
p = p[p.cls != "other"].copy()
p["rv"] = pd.to_numeric(p.delta_run_exp, errors="coerce").astype(float) * 100
p["xw"] = pd.to_numeric(p.estimated_woba_using_speedangle, errors="coerce").astype(float) * 1000
p["sw"], p["wh"] = p.swing.astype(bool), p.whiff.astype(bool)
for side in ("L", "R"):
    d = p[p.stand == side]
    print(f"\n=== {side}HB ({len(d):,} pitches)")
    tot = d.groupby("same").rv.mean()
    print(
        f"run value per 100 pitches: vs opposite {tot['opp']:+.2f}, vs same {tot['same']:+.2f}, gap {tot['same'] - tot['opp']:+.2f}"
    )
    sh = d.groupby(["same", "cls"]).size().unstack()
    sh = sh.div(sh.sum(1), axis=0)
    rv = d.groupby(["same", "cls"]).rv.mean().unstack()
    mix = ((sh.loc["same"] - sh.loc["opp"]) * rv.loc["opp"]).sum()
    res = sh.loc["same"] * (rv.loc["same"] - rv.loc["opp"])
    print(
        f"   gap from pitch MIX (pitchers throw different pitches): {mix:+.2f}; from RESULTS per pitch: {res.sum():+.2f} = "
        + ", ".join(f"{c} {res[c]:+.2f}" for c in res.index)
    )
    for c in ("fastball", "breaking", "offspeed"):
        z = d[d.cls == c]
        g = lambda s: pd.Series(
            {
                "share": 0,
                "chase": 100 * z[(z.same == s) & z.out_zone].sw.mean(),
                "whiff": 100 * z[(z.same == s) & z.sw].wh.mean(),
                "xwcon": z[(z.same == s) & (z.type == "X")].xw.mean(),
                "rv100": z[z.same == s].rv.mean(),
            }
        )
        a, b = g("same"), g("opp")
        print(
            f"   {c:9s} share opp {100*sh.loc['opp', c]:4.1f}% same {100*sh.loc['same', c]:4.1f}% | chase {b.chase:4.1f} -> {a.chase:4.1f} | "
            f"whiff {b.whiff:4.1f} -> {a.whiff:4.1f} | xwOBA on contact {b.xwcon:3.0f} -> {a.xwcon:3.0f} | rv/100 {b.rv100:+.2f} -> {a.rv100:+.2f}"
        )
