"""Guerrero diagnosis figure: (1) exit velocity above what the league model expects from his swing + pitch,
(2) attack angle vs league, (3) share of balls topped (LA < -5) and popped up (LA > 45)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the analysis/ folder
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#d9d8d3", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
yrs = ["2024", "2025", "2026"]
ev_above = [3.3, 2.4, 1.2]
aa = [2.7, -0.7, 0.0]
lg_aa = 8.7
topped = [28.2, 29.3, 30.9]
under = [7.4, 9.7, 11.9]
fig, axs = plt.subplots(3, 1, figsize=(2.5, 5.6))


def bars(ax, vals, title, fmt, ylim):
    bs = ax.bar(yrs, vals, color=BLUE, width=0.55)
    for b_, v in zip(bs, vals):
        ax.text(
            b_.get_x() + b_.get_width() / 2,
            v + (ylim[1] - ylim[0]) * (0.03 if v >= 0 else -0.1),
            (fmt.format(v) if abs(v) >= 0.05 else "0.0"),
            ha="center",
            fontsize=7.5,
            color=INK,
        )
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=INK2, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


bars(axs[0], ev_above, "Squaring up: exit velo above\nwhat his swing + pitch predict (mph)", "{:+.1f}", (0, 4))
bars(axs[1], aa, "Attack angle (deg)", "{:+.1f}", (-2, 10))
axs[1].axhline(lg_aa, color=ORANGE, lw=1.2, ls="--")
axs[1].text(2.35, lg_aa + 0.3, "league 8.7", color=ORANGE, fontsize=7, ha="right")
ax = axs[2]
ax.plot(yrs, topped, color=BLUE, marker="o", lw=1.8, ms=5)
ax.plot(yrs, under, color=ORANGE, marker="s", lw=1.8, ms=5)
ax.text(2.14, topped[-1], "topped", color=INK, fontsize=7, va="center")
ax.text(2.14, under[-1], "popped up", color=INK, fontsize=7, va="center")
ax.set_xlim(-0.3, 2.9)
ax.set_ylim(0, 36)
ax.set_title("Balls in play (%)", fontsize=8, loc="left", color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK2, labelsize=7)
ax.yaxis.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(h_pad=1.2)
fig.savefig(ROOT + "/jays/vlad_diag.png", dpi=220)
