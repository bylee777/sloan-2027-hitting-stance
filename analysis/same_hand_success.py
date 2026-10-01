"""Is there any hitter who sets up differently vs SAME-hand pitchers and makes it work?

Setup difference (same-hand minus opposite-hand) per hitter-season (60+ swings vs each hand, 2024-26) on five levers:
depth, distance off the plate, width (in), stance angle, front-foot turn (deg).
ADJUSTER: a lever differs by >= 2 in (or >= 5 deg) in the same direction in at least two seasons (a habit, not noise).
WORKS: the hitter's same-hand penalty (same minus opposite xwOBA per PA, K%) is smaller than the league's for his
batting side that season. Because one season's split is very noisy, report the multi-season excess with its SE and
check (1) the base rate among non-adjusters, (2) the same hitter in seasons with vs without the adjustment."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
X = pd.read_csv("foot_adjusters.csv")
X["year"] = X.year.astype(str)
# distance off the plate by pitcher hand
st = []
for y in ("2024", "2025", "2026"):
    for h in ("L", "R"):
        s = pd.read_csv(f"hand_{y}_{h}.csv", encoding="utf-8-sig")
        s["year"] = y
        s["p_throws"] = h
        st.append(s)
st = pd.concat(st).rename(columns={"id": "batter", "avg_batter_x_position": "offp"})
st["m"] = np.where(st.side == st.p_throws, "same", "opp")
op = (
    st.pivot_table(index=["batter", "side", "year"], columns="m", values="offp")
    .reset_index()
    .rename(columns={"same": "offp_same", "opp": "offp_opp"})
)
X = X.merge(op, on=["batter", "side", "year"], how="left")
LEV = {
    "depth": ("depth_in", 2.0, "in", ("deeper", "closer to the pitcher")),
    "offp": ("offp", 2.0, "in", ("farther off the plate", "closer to the plate")),
    "width": ("width_in", 2.0, "in", ("wider", "narrower")),
    "angle": ("stance_angle", 5.0, "deg", ("more closed", "more open")),
    "foot": ("front_turn0", 5.0, "deg", ("front foot more open", "front foot more closed")),
}
for k, (c, thr, u, lab) in LEV.items():
    X[f"d_{k}"] = X[f"{c}_same"] - X[f"{c}_opp"]
    X[f"a_{k}"] = np.sign(X[f"d_{k}"]) * (X[f"d_{k}"].abs() >= thr)  # +1 / -1 / 0 this season
# league platoon penalty by side-year; excess = hitter's penalty minus league's (+ = better than typical vs same-hand)
for g in ("xw", "K"):
    X[f"gap_{g}"] = X[f"{g}_same"] - X[f"{g}_opp"]
    X[f"lg_{g}"] = X.groupby(["side", "year"])[f"gap_{g}"].transform("mean")
X["ex_xw"] = X.gap_xw - X.lg_xw  # + = smaller same-hand xwOBA penalty than typical
X["ex_K"] = -(X.gap_K - X.lg_K)  # + = smaller same-hand K penalty than typical
SD_XW, SD_K = 366.0, 41.5  # per-PA SD of xwOBA (x1000) and of K (x100)
X["se_xw"] = SD_XW * np.sqrt(1 / X.PA_same + 1 / X.PA_opp)
X["se_K"] = SD_K * np.sqrt(1 / X.PA_same + 1 / X.PA_opp)
print(
    f"{len(X)} hitter-seasons, {X.batter.nunique()} hitters. League same-hand penalty (xwOBA per PA): "
    + ", ".join(f"{s}HB {X[X.side == s].lg_xw.mean():+.0f} pts" for s in ("L", "R"))
    + f"; one season's split SE ~{X.se_xw.median():.0f} pts"
)
nx = X[["batter", "side", "year", "ex_xw"]].assign(year=lambda z: (z.year.astype(int) - 1).astype(str))
r = X.merge(nx, on=["batter", "side", "year"], suffixes=("", "_n"))
print(
    f"how repeatable is a hitter's excess same-hand penalty season to season? r = {r.ex_xw.corr(r.ex_xw_n):+.2f} (n {len(r)})"
)

rows = []
for k, (c, thr, u, lab) in LEV.items():
    for (b, s), g in X.groupby(["batter", "side"]):
        for sgn in (1, -1):
            on = g[g[f"a_{k}"] == sgn]
            if len(on) >= 2:
                off = g[g[f"a_{k}"] != sgn]
                w = 1 / on.se_xw**2
                wk = 1 / on.se_K**2
                rows.append(
                    dict(
                        name=g.name.iloc[0],
                        side=s,
                        lever=k,
                        move=f"{abs(on[f'd_{k}'].mean()):.1f} {u} {lab[0] if sgn > 0 else lab[1]}",
                        seasons=", ".join(on.year),
                        ex_xw=(on.ex_xw * w).sum() / w.sum(),
                        se=1 / np.sqrt(w.sum()),
                        ex_K=(on.ex_K * wk).sum() / wk.sum(),
                        se_K=1 / np.sqrt(wk.sum()),
                        every=bool((on.ex_xw > 0).all()),
                        n_off=len(off),
                        off_ex_xw=off.ex_xw.mean() if len(off) else np.nan,
                        batter=b,
                    )
                )
A = pd.DataFrame(rows)
print(
    f"\nhabitual adjusters (same direction, >= 2 seasons): {A.batter.nunique()} hitters, {len(A)} hitter-lever habits"
)
print("   by lever: " + ", ".join(f"{k} {n}" for k, n in A.lever.value_counts().items()))
# base rate: among ALL hitters with >= 2 seasons, how often is the multi-season excess > 0 / > 1.64 SE?
base = []
for (b, s), g in X.groupby(["batter", "side"]):
    if len(g) >= 2:
        w = 1 / g.se_xw**2
        e = (g.ex_xw * w).sum() / w.sum()
        se = 1 / np.sqrt(w.sum())
        base.append(dict(adj=b in set(A.batter), ex=e, z=e / se, every=bool((g.ex_xw > 0).all())))
B = pd.DataFrame(base)
for a, g in B.groupby("adj"):
    print(
        f"   {'adjusters    ' if a else 'non-adjusters'} n {len(g):3d}: mean excess {g.ex.mean():+5.1f} pts, better than league every season "
        f"{g.every.mean():.0%}, clearly better (z > 1.64) {(g.z > 1.64).mean():.0%}"
    )
print(f"   (by chance alone ~5% would be 'clearly better')")

print("\nhabitual adjusters, best same-hand results first (excess = smaller penalty than typical; + is good):")
A = A.sort_values("ex_xw", ascending=False)
A["verdict"] = np.select(
    [(A.ex_xw / A.se > 1.64) & A.every, (A.ex_xw > 0) & A.every],
    ["WORKS (clear)", "better every season, not clear"],
    "no",
)
show = A[["name", "side", "lever", "move", "seasons", "ex_xw", "se", "ex_K", "se_K", "off_ex_xw", "verdict"]].copy()
print(show.round(1).to_string(index=False))
# within hitter: seasons with the habit vs seasons without
w = A[A.n_off > 0].dropna(subset=["off_ex_xw"])
diff = w.ex_xw - w.off_ex_xw
print(
    f"\nsame hitter, seasons WITH the adjustment minus WITHOUT: {diff.mean():+.1f} pts xwOBA (SE {diff.std(ddof=1)/np.sqrt(len(diff)):.1f}, n {len(diff)})"
)
A.to_csv("same_hand_success.csv", index=False)

print("\nseason detail for the best cases (same = vs same-hand pitchers):")
for nm in ("Schwarber", "Vientos", "Young, Cole", "Mullins"):
    d = X[X.name.str.contains(nm)]
    for _, r in d.iterrows():
        print(
            f"   {r['name']:18s} {r.year}: off plate same {r.offp_same:5.1f} / opp {r.offp_opp:5.1f} in, angle same {r.stance_angle_same:6.1f} / opp {r.stance_angle_opp:6.1f} | "
            f"xwOBA same {r.xw_same:4.0f} / opp {r.xw_opp:4.0f} (PA {r.PA_same:.0f}/{r.PA_opp:.0f}) | K% same {r.K_same:4.1f} / opp {r.K_opp:4.1f} | excess {r.ex_xw:+4.0f} (SE {r.se_xw:.0f})"
        )
