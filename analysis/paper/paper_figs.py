"""Figures for the paper draft: F1 stride equalizer (between hitters + same hitter), F3 slump decomposition."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the analysis/ folder
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = ROOT
INK, INK2, GRID, BLUE, ORANGE, GRAY = "#0b0b0b", "#52514e", "#d9d8d3", "#2a78d6", "#eb6834", "#b8b6b0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


H = pd.read_parquet(f"{B}/taxonomy/hitters_pca.parquet").dropna(subset=["width_in", "stride_len_in", "width_land_in"])
E = pd.read_csv(f"{B}/equalizer_events.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axs[0]
ax.scatter(H.width_in, H.stride_len_in, s=4, color=BLUE, alpha=0.35, lw=0)
ax.set_xlabel("Stance width at setup (in)", color=INK2)
ax.set_ylabel("Stride length (in)", color=INK2)
ax.set_title(
    f"(a) Wider setup, shorter stride\nr = {H.width_in.corr(H.stride_len_in):+.2f}", fontsize=8, loc="left", color=INK
)
style(ax)
ax = axs[1]
ax.scatter(H.width_in, H.width_land_in, s=4, color=BLUE, alpha=0.35, lw=0)
ax.axhline(H.width_land_in.median(), color=INK2, lw=0.8, ls="--")
ax.set_xlabel("Stance width at setup (in)", color=INK2)
ax.set_ylabel("Width at stride landing (in)", color=INK2)
ax.set_title(
    f"(b) Everyone lands near {H.width_land_in.median():.0f} in\nr = {H.width_in.corr(H.width_land_in):+.2f}",
    fontsize=8,
    loc="left",
    color=INK,
)
style(ax)
ax = axs[2]
ax.scatter(E.d_setup, E.d_land, s=9, color=ORANGE, alpha=0.7, lw=0)
lim = [-12, 12]
ax.plot(lim, lim, color=INK2, lw=0.8, ls=":")
ax.text(-11.3, 8.6, "dotted: landing moves\n1-for-1 with setup", fontsize=6.3, color=INK2)
b = np.polyfit(E.d_setup, E.d_land, 1)
xx = np.array(lim)
ax.plot(xx, np.polyval(b, xx), color=ORANGE, lw=1.6)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Change in setup width (in)", color=INK2)
ax.set_ylabel("Change in landing width (in)", color=INK2)
ax.set_title(f"(c) Same hitter, {len(E)} changes\nslope {b[0]:+.2f}", fontsize=8, loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(f"{B}/paper/fig1_equalizer.png", dpi=220)
plt.close(fig)
# ---- F3 slump decomposition (slump_test.txt, matched to 20 slump-alike non-changers; xwOBA points, after vs 30-day run-up;
#      n = changes with an xwOBA outcome, from dr_counterfactual.txt's match20 rows)
# side-by-side bars (no stacking): grey = what matched slumping non-changers regained anyway; blue = changers beyond that, 90% interval
cats = ["Wider", "Narrower", "Deeper", "Shallower"]
bounce = [13.91, 3.24, -0.11, -4.77]
real = [16.12, 16.50, 29.75, 7.26]
se = [4.97, 7.10, 13.57, 13.17]
n = [47, 31, 11, 12]
fig, ax = plt.subplots(figsize=(4.6, 2.7))
x = np.arange(len(cats))
w = 0.36
ax.bar(x - w / 2, bounce, width=w, color=GRAY, label="bounce-back: matched non-changers")
ax.bar(x + w / 2, real, width=w, color=BLUE, label="beyond bounce-back (90% interval)")
ax.errorbar(x + w / 2, real, yerr=1.645 * np.array(se), fmt="none", ecolor=INK, lw=0.8, capsize=2)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\nn = {k}" for c, k in zip(cats, n)], fontsize=7.5, color=INK)
ax.set_ylabel("xwOBA points, 60 days after\nvs the 30 days before", color=INK2)
ax.set_ylim(-30, 62)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
style(ax)
ax.grid(False, axis="x")
fig.tight_layout()
fig.savefig(f"{B}/paper/fig3_slump.png", dpi=220)
plt.close(fig)
print("ok")
