"""Per-player setup reports, 2026. Only evidence-backed levers, each with its evidence strength:
 A  DEEPER vs SAME-HAND pitchers  — chase-prone (2025 & 2026 > top-third), same-hand chase >= opp + 3 pp, room to move
    (depth < league 75th pct). Evidence: MODERATE (between hitters, same size 2025/2026; movers back chase -3.2 pp).
 B  WIDER, full-time              — whiff >= league 67th pct, NOT chase-prone, width below league median.
    Evidence: STRONG (201 width changes; non-chasers whiff -3.0 pp; -1.9 pp every season).
 C  CLOSER to the plate           — outside-minus-inside whiff gap >= league + 5 pp, off-plate above league median.
    Evidence: MODERATE (reach/jam trade-off, every season, between hitters; costs inside).
 D  DEEPER, full-time             — sweet-spot below league 40th pct, contact >= 6 in out front, depth below median.
    Evidence: MODERATE (movers back: sweet-spot +5.5 pp, positive every season; small n).
Sizes stay inside the observed range (<= 6 in depth, <= 6 in width, <= 3 in plate distance)."""

import sys, numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from stance_lib import pitches

H = pd.read_parquet("taxonomy/hitters_pca.parquet").merge(
    pd.read_parquet("taxonomy/hitters_k4.parquet")[["batter", "side", "year", "arch", "conf"]],
    on=["batter", "side", "year"],
)
NAMES = {1: "Balanced", 2: "Open-and-stride", 3: "Square", 4: "Wide-and-quiet"}
p = pitches()
p = p[p.year.isin(["2025", "2026"])]
p["same"] = p.stand == p.p_throws
sgn = np.where(p.stand == "R", 1.0, -1.0)
xa = p.plate_x.astype(float) * sgn
p["band"] = np.select([xa > 0.236, xa < -0.236], ["away", "in"], "mid")
g = p.groupby(["batter", "side", "year"])
d = pd.DataFrame(
    {
        "chase_same": g.apply(lambda z: z[z.out_zone & z.same].swing.mean()),
        "chase_opp": g.apply(lambda z: z[z.out_zone & ~z.same].swing.mean()),
        "whiff_away": g.apply(lambda z: z[z.swing & (z.band == "away")].whiff.mean()),
        "whiff_in": g.apply(lambda z: z[z.swing & (z.band == "in")].whiff.mean()),
        "same_share": g.same.mean(),
    }
).reset_index()
H = H.merge(d, on=["batter", "side", "year"], how="left")
import numpy as np

HH = H.dropna(subset=["height_in"])
bw = np.polyfit(HH.height_in, HH.width_in, 1)
bo = np.polyfit(HH.height_in, HH.off_plate_in, 1)
H["width_exp"] = np.polyval(bw, H.height_in.fillna(HH.height_in.median()))
H["off_exp"] = np.polyval(bo, H.height_in.fillna(HH.height_in.median()))
H["width_rel"] = H.width_in - H.width_exp
H["off_rel"] = H.off_plate_in - H.off_exp  # + = wider / farther off than typical for his height
cur = H[H.year == "2026"].copy()
prev = H[H.year == "2025"][["batter", "side", "chase"]].rename(columns={"chase": "chase_2025"})
cur = cur.merge(prev, on=["batter", "side"], how="left")
L = H[H.year == "2026"]
cut_chase26, cut_chase25 = L.chase.quantile(2 / 3), H[H.year == "2025"].chase.quantile(2 / 3)
q = lambda c, x: (L[c] < x).mean() * 100
cur["gap_away_in"] = cur.whiff_away - cur.whiff_in
lg_gap = (L.whiff_away - L.whiff_in).mean()
recs = []
for _, r in cur.iterrows():
    out = []
    chase_prone = r.chase > cut_chase26 and (pd.isna(r.chase_2025) or r.chase_2025 > cut_chase25)
    if (
        chase_prone
        and r.chase_same - r.chase_opp >= 0.03
        and r.depth_in < L.depth_in.quantile(0.75)
        and r.same_share >= 0.3
    ):
        out.append(
            (
                "A",
                f"stand ~{min(6, L.depth_in.quantile(.75) - r.depth_in):.0f} in DEEPER vs same-hand pitchers (chases {100*r.chase_same:.0f}% vs same-hand, {100*r.chase_opp:.0f}% vs opposite)",
                "moderate",
            )
        )
    if (not chase_prone) and r.whiff >= L.whiff.quantile(2 / 3) and r.width_rel < 0:
        out.append(
            (
                "B",
                f"go ~{min(6, 3 - r.width_rel):.0f} in WIDER, full-time (whiff {100*r.whiff:.0f}%, {q('whiff', r.whiff):.0f}th pct; width {r.width_in:.0f} in = {-r.width_rel:.0f} in narrower than typical for his {r.height_in:.0f}-in height)",
                "strong",
            )
        )
    if pd.notna(r.gap_away_in) and r.gap_away_in >= lg_gap + 0.05 and r.off_rel > 0:
        out.append(
            (
                "C",
                f"move ~2-3 in CLOSER to the plate (whiffs {100*r.whiff_away:.0f}% outside vs {100*r.whiff_in:.0f}% inside; {r.off_rel:.0f} in farther off than typical for his height; watch inside pitches)",
                "moderate",
            )
        )
    if r.sweet < L.sweet.quantile(0.4) and r.contact_vs_plate_in >= 6 and r.depth_in < L.depth_in.median():
        out.append(
            (
                "D",
                f"stand ~3-4 in DEEPER, full-time, for deeper contact (sweet-spot {100*r.sweet:.0f}%, contact {r.contact_vs_plate_in:.0f} in out front)",
                "moderate",
            )
        )
    recs.append(out)
cur["recs"] = recs
cur["n_recs"] = cur.recs.apply(len)
cur["archetype"] = cur.arch.map(NAMES)
print(
    f"2026 hitters with a report: {len(cur)}; with at least one evidence-backed change: {(cur.n_recs > 0).sum()} ({(cur.n_recs > 0).mean():.0%})"
)
for k, lab in (("A", "deeper vs same-hand"), ("B", "wider"), ("C", "closer to plate"), ("D", "deeper full-time")):
    print(f"   {k} {lab:22s}: {cur.recs.apply(lambda rs: any(x[0] == k for x in rs)).sum()} hitters")
print("\nby archetype, share of hitters with each change:")
for k in "ABCD":
    cur[k] = cur.recs.apply(lambda rs: any(x[0] == k for x in rs))
print(cur.groupby("archetype")[list("ABCD")].mean().round(2).to_string())


def card(r):
    pct = lambda c: (L[c] < r[c]).mean() * 100
    lines = [
        f"### {r['name']} ({r.team}, bats {r.side})",
        f"Stance fingerprint (league percentile): setup-vs-stride {pct('PC1'):.0f} | stride direction {pct('PC2'):.0f} | reach {pct('PC3'):.0f} | landing {pct('PC4'):.0f}",
        f"Stance: starts {r.depth_in:.0f} in deep, {r.off_plate_in:.0f} in off the plate, {r.width_in:.0f} in wide, angle {r.stance_angle:+.0f}° "
        f"(- = open); strides {r.stride_len_in:.0f} in, lands {r.width_land_in:.0f} in wide ({100*r.width_land_in/r.height_in:.0f}% of his height); contact {r.contact_vs_plate_in:+.0f} in vs front of plate. Height {r.height_in:.0f} in.",
        f"Profile (2026): K {100*r.K:.0f}% | chase {100*r.chase:.0f}% ({q('chase', r.chase):.0f}th pct) | whiff {100*r.whiff:.0f}% ({q('whiff', r.whiff):.0f}th) | "
        f"sweet-spot {100*r.sweet:.0f}% ({q('sweet', r.sweet):.0f}th) | xwOBA {r.xwoba:.3f} | bat speed {r.bat_speed:.1f}",
    ]
    if r.recs:
        for k, txt, ev in r.recs:
            lines.append(f"- **{k}** {txt} — evidence: {ev}")
    else:
        lines.append("- No evidence-backed change: current setup is not tied to a problem the data can fix.")
    return "\n".join(lines)


pick = [
    "Crow-Armstrong",
    "Jung, Josh",
    "Altuve",
    "Burger",
    "Muncy, Max",
    "Story",
    "Judge",
    "Soto",
    "Witt",
    "Walker, Jordan",
    "Devers",
    "Kwan",
]
cards = [card(r) for nm in pick for _, r in cur[cur.name.str.contains(nm)].iterrows()]
open("taxonomy/player_cards.md", "w").write("# Player setup reports (2026)\n\n" + "\n\n".join(cards) + "\n")
cur.drop(columns=["recs"]).assign(
    recs=cur.recs.apply(lambda rs: " | ".join(f"{k}: {t} [{e}]" for k, t, e in rs))
).to_csv("taxonomy/player_reports_2026.csv", index=False)
print("\n" + "\n\n".join(cards))
