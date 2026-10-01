"""(1) Blue Jays run-value accounting heatmap (2026, runs per 600 PA vs league average; raw parts add up to the total).
(2) League: what separates hitters (share of between-hitter variance) and how much of each part is skill (year-to-year r).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the analysis/ folder
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

B = ROOT
INK, INK2, GRID, BLUE, ORANGE, MID = "#0b0b0b", "#52514e", "#d9d8d3", "#2a78d6", "#eb6834", "#f2f1ee"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
H = pd.read_csv(f"{B}/hitter_diagnosis_intended.csv", dtype={"year": str})
rep = pd.read_csv(f"{B}/taxonomy/player_reports_2026.csv")
jays = set(rep[rep.team == "Blue Jays"].batter)
J = H[(H.year == "2026") & H.batter.isin(jays)].sort_values("TOTAL", ascending=False)
COLS = [
    ("DECISIONS", "Swing\ndecisions"),
    ("WHIFFS", "Whiffs"),
    ("INPUTS", "Swing\ninputs"),
    ("SQUARING", "Squaring\nup"),
    ("LUCK", "Luck"),
    ("OTHER", "Other"),
    ("TOTAL", "TOTAL"),
]
M = J[[c for c, _ in COLS]].to_numpy()
cmap = LinearSegmentedColormap.from_list("div", [ORANGE, MID, BLUE])
fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(J) + 1.1))
ax.imshow(np.clip(M, -30, 30), cmap=cmap, vmin=-30, vmax=30, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(
            j,
            i,
            f"{M[i, j]:+.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            fontweight="bold" if j == M.shape[1] - 1 else "normal",
        )
names = [" ".join(n.split(", ")[::-1]) + (f"  ({int(pa)} PA)" if pa < 250 else "") for n, pa in zip(J.name, J.PA)]
ax.set_yticks(range(len(J)))
ax.set_yticklabels(names, fontsize=8, color=INK)
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([l for _, l in COLS], fontsize=7.5, color=INK)
ax.xaxis.tick_top()
ax.axvline(len(COLS) - 1.5, color=INK2, lw=1)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.text(
    0.01,
    0.01,
    "Runs per 600 PA vs league average, 2026. Blue = gains, orange = losses. Parts add up exactly to the total.",
    fontsize=7,
    color=INK2,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{B}/jays/jays_accounting.png", dpi=220)
plt.close(fig)
# ---- league: what separates hitters
a = H[H.PA >= 300]
REL = pd.read_pickle(f"{B}/hitter_diagnosis_intended_rel.pkl")
P = ["SQUARING", "INPUTS", "LUCK", "DECISIONS", "OTHER", "WHIFFS"]
LAB = {
    "SQUARING": "Squaring up",
    "INPUTS": "Swing inputs",
    "LUCK": "Luck",
    "DECISIONS": "Swing decisions",
    "OTHER": "Other",
    "WHIFFS": "Whiffs",
}
share = [100 * np.cov(a[k], a.TOTAL)[0, 1] / a.TOTAL.var() for k in P]
rel = [REL[k] for k in P]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
y = np.arange(len(P))[::-1]
axs[0].barh(y, share, color=BLUE, height=0.55)
for yy, v in zip(y, share):
    axs[0].text(max(v, 0) + 1, yy, f"{v:.0f}%", va="center", fontsize=7.5, color=INK)
axs[0].axvline(0, color=INK2, lw=0.6)
axs[0].set_xlim(-5, 45)
axs[0].set_title(
    "What separates good and bad hitters\n(share of the spread in total runs)", fontsize=8, loc="left", color=INK
)
axs[1].barh(y, rel, color=BLUE, height=0.55)
for yy, v in zip(y, rel):
    axs[1].text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
axs[1].set_xlim(0, 1.05)
axs[1].set_title("How much is skill\n(same hitter, next season: r)", fontsize=8, loc="left", color=INK)
axs[0].set_yticks(y)
axs[0].set_yticklabels([LAB[k] for k in P], fontsize=8, color=INK)
for ax in axs:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{B}/jays/league_accounting.png", dpi=220)
plt.close(fig)
print("figures written;", len(J), "Jays rows")
