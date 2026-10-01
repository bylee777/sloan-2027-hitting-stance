"""Turn the run-value accounting into an improvement plan for every 2026 regular (300+ PA).
For each part (decisions, whiffs, swing inputs, squaring up) the reliability-shrunk value below league average is a
GAP; the plan lists the two biggest gaps, a sub-diagnosis from the underlying measures, the matching lever and how
strong the evidence for that lever is. 'Potential' = runs per 600 PA if half the gap closes (an illustrative target,
not a forecast). Luck is reported separately (expect it to reverse). Also: decline vs the hitter's own 2024-25."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
H = pd.read_csv("hitter_diagnosis_intended.csv", dtype={"year": str})
P = pd.read_parquet("hitter_diagnosis_pitches.parquet", columns=["batter", "side", "year", "azone", "sw"])
Z = (
    P.groupby(["batter", "side", "year"])
    .apply(
        lambda d: pd.Series(
            {
                "heart_sw": 100 * d[d.azone == "heart"].sw.mean(),
                "chase_sw": 100 * d[d.azone.isin(["chase", "waste"])].sw.mean(),
            }
        )
    )
    .reset_index()
)
X = pd.read_parquet("taxonomy/hitters_pca.parquet")[
    [
        "batter",
        "side",
        "year",
        "bat_speed",
        "attack_angle",
        "whiff",
        "chase",
        "depth_in",
        "width_in",
        "height_in",
        "contact_vs_plate_in",
        "sweet",
        "team",
    ]
]
X["year"] = X.year.astype(str)
D = H.merge(Z, on=["batter", "side", "year"], how="left").merge(X, on=["batter", "side", "year"], how="left")
cur = D[(D.year == "2026") & (D.PA >= 250)].copy()
Lg = D[(D.year == "2026") & (D.PA >= 250)]
pct = lambda c, v: (Lg[c] < v).mean() * 100
bw = np.polyfit(
    Lg.dropna(subset=["height_in", "width_in"]).height_in, Lg.dropna(subset=["height_in", "width_in"]).width_in, 1
)
PARTS = ["DECISIONS", "WHIFFS", "INPUTS", "SQUARING"]
prior = (
    D[D.year.isin(["2024", "2025"]) & (D.PA >= 300)]
    .groupby(["batter", "side"])[[k + "_s" for k in PARTS + ["LUCK"]] + ["TOTAL"]]
    .mean()
)


def lever(r, part):
    if part == "DECISIONS":
        if r.chase_sw >= Lg.chase_sw.quantile(0.67):
            s = f"chases {r.chase_sw:.0f}% ({pct('chase_sw', r.chase_sw):.0f}th pct): approach / pitch recognition"
            if r.depth_in < Lg.depth_in.median():
                s += f"; stance: ~3 in deeper (stands {r.depth_in:.0f} in; model chase -0.5 pp per 3 in, validated on 2026)"
                return s, "coaching + moderate (stance)"
            return s, "coaching"
        if r.heart_sw <= Lg.heart_sw.quantile(0.33):
            return (
                f"too passive on pitches down the middle (swings at {r.heart_sw:.0f}%, {pct('heart_sw', r.heart_sw):.0f}th pct): attack hittable pitches",
                "coaching",
            )
        return "swing/take choices in specific counts and locations", "coaching"
    if part == "WHIFFS":
        s = f"whiffs {100*r.whiff:.0f}% ({pct('whiff', r.whiff):.0f}th pct): contact work"
        chase_prone = r.chase_sw >= Lg.chase_sw.quantile(0.67)
        if (
            pd.notna(r.width_in)
            and pd.notna(r.height_in)
            and r.width_in < np.polyval(bw, r.height_in)
            and not chase_prone
        ):
            s += "; stance: wider (narrower than typical for his height; model whiffs -0.3 pp per 3 in)"
            return s, "coaching + moderate (stance)"
        return s, "coaching"
    if part == "INPUTS":
        bits = []
        if r.bat_speed <= Lg.bat_speed.quantile(0.33):
            bits.append(
                f"bat speed {r.bat_speed:.1f} mph ({pct('bat_speed', r.bat_speed):.0f}th pct): strength / intent"
            )
        if r.attack_angle <= Lg.attack_angle.quantile(0.2):
            bits.append(f"flat swing path (attack angle {r.attack_angle:.0f}°, league {Lg.attack_angle.median():.0f}°)")
        if r.contact_vs_plate_in >= Lg.contact_vs_plate_in.quantile(0.8) and r.sweet <= Lg.sweet.quantile(0.4):
            bits.append(
                f"meets the ball far out front ({r.contact_vs_plate_in:.0f} in) with low sweet-spot: stance ~3 in deeper (sweet-spot +1.6 pp per 3 in)"
            )
        return ("; ".join(bits) if bits else "power inputs below average on the pitches he gets (swing + pitch mix)"), (
            "moderate (stance)" if "deeper" in " ".join(bits) else "coaching"
        )
    if part == "SQUARING":
        return (
            "not squaring the ball up given his swing: timing consistency / contact precision (check health if new)",
            "coaching",
        )


rows = []
for _, r in cur.iterrows():
    gaps = sorted([(r[k + "_s"], k) for k in PARTS], key=lambda t: t[0])
    plan = []
    for v, k in gaps[:2]:
        if v < -1.0:
            txt, ev = lever(r, k)
            plan.append((k, v, 0.5 * -v, txt, ev))
    pri = prior.loc[(r.batter, r.side)] if (r.batter, r.side) in prior.index else None
    decline = {k: r[k + "_s"] - pri[k + "_s"] for k in PARTS} if pri is not None else {}
    rows.append(
        dict(
            batter=r.batter,
            name=r["name"],
            side=r.side,
            team=r.team,
            PA=r.PA,
            total=r.TOTAL,
            **{k.lower(): r[k + "_s"] for k in PARTS + ["LUCK"]},
            gap1=plan[0][0] if plan else "",
            gap1_runs=plan[0][1] if plan else 0.0,
            gain1=plan[0][2] if plan else 0.0,
            lever1=plan[0][3] if plan else "no part clearly below average",
            ev1=plan[0][4] if plan else "",
            gap2=plan[1][0] if len(plan) > 1 else "",
            gain2=plan[1][2] if len(plan) > 1 else 0.0,
            lever2=plan[1][3] if len(plan) > 1 else "",
            ev2=plan[1][4] if len(plan) > 1 else "",
            biggest_decline=min(decline, key=decline.get) if decline else "",
            decline_runs=min(decline.values()) if decline else np.nan,
            regain_half=0.5 * -min(decline.values()) if decline and min(decline.values()) < -5 else 0.0,
        )
    )
R = pd.DataFrame(rows).sort_values("total")
R.to_csv("improvement_plans_2026_intended.csv", index=False)
print(
    f"2026 regulars with a plan (250+ PA): {len(R)}; with at least one clear gap (> 1 run below average): {(R.gap1 != '').sum()}"
)
print("main gap:", R[R.gap1 != ""].gap1.value_counts().to_dict())
print("share whose main gap has a STANCE lever with evidence:", f"{R.ev1.str.contains('stance').mean():.0%}")
S = R[R.ev1.str.contains("stance")]
wide = S.lever1.str.contains("wider")
print(
    f"   of which deeper: {(~wide).sum()} ({(~wide).sum() / len(R):.0%}; chasing {(~wide & (S.gap1 == 'DECISIONS')).sum()}, "
    f"contact far out front {(~wide & (S.gap1 == 'INPUTS')).sum()}); wider (trades power for contact): {wide.sum()}"
)
G = R[[p.lower() for p in PARTS]].min(axis=1)
print(f"hitters with a skill part 5+ runs below average: {(G <= -5).sum()}; 10+ runs: {(G <= -10).sum()}")
print("median potential if half the main gap closes:", f"{R[R.gap1 != ''].gain1.median():.1f} runs per 600 PA")
pd.set_option("display.width", 250, "display.max_colwidth", 120)
for nm in (
    "Guerrero Jr., Vladimir",
    "Clement, Ernie",
    "Okamoto, Kazuma",
    "Kirk, Alejandro",
    "Crow-Armstrong, Pete",
    "Giménez, Andrés",
):
    z = R[R.name == nm]
    if len(z):
        r = z.iloc[0]
        print(
            f"\n{nm}: total {r.total:+.1f} | decisions {r.decisions:+.1f}, whiffs {r.whiffs:+.1f}, inputs {r.inputs:+.1f}, squaring {r.squaring:+.1f}, luck {r.luck:+.1f}"
        )
        print(f"   1) {r.gap1} ({r.gain1:+.1f} if half closes): {r.lever1} [{r.ev1}]")
        if r.gap2:
            print(f"   2) {r.gap2} ({r.gain2:+.1f}): {r.lever2} [{r.ev2}]")
        if r.biggest_decline:
            print(
                f"   biggest decline vs his 2024-25: {r.biggest_decline} {r.decline_runs:+.1f}"
                + (f" -> regaining half = {r.regain_half:+.1f}" if r.regain_half else "")
            )
J = R[R.team == "Blue Jays"]
print("\n=== Blue Jays (runs per 600 PA vs league average, shrunk)")
print(
    J[
        [
            "name",
            "PA",
            "total",
            "decisions",
            "whiffs",
            "inputs",
            "squaring",
            "luck",
            "gap1",
            "gain1",
            "biggest_decline",
            "decline_runs",
        ]
    ]
    .round(1)
    .to_string(index=False)
)
