"""Visual guide figures. Palette: validated reference slots (blue setup, orange landing/change, aqua 3rd)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the analysis/ folder
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch, Rectangle

BASE = ROOT
INK, INK2, GRID, SURF, TINT = "#0b0b0b", "#52514e", "#d9d8d3", "#ffffff", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": SURF, "axes.facecolor": SURF})
med = pd.read_csv(f"{BASE}/visual/typical_feet.csv", index_col=0).iloc[:, 0]
F = pd.read_parquet(f"{BASE}/visual/feet_2026.parquet")


def field(ax, title=None, ylim=(-40, 45), xlim=(-80, 20)):
    ax.add_patch(Polygon([(0, 0), (-8.5, 8.5), (-8.5, 17), (8.5, 17), (8.5, 8.5)], closed=True, fc=TINT, ec=INK2, lw=1))
    ax.plot([xlim[0] + 4, 8.5], [17, 17], color=GRID, lw=0.9, ls=":")
    ax.text(8.5, 19, "front of plate", ha="right", color=INK2, fontsize=7)
    ax.text(0, 8, "plate", ha="center", va="center", color=INK2, fontsize=7.5)
    ax.annotate(
        "", xy=(14, ylim[1] - 2), xytext=(14, ylim[1] - 18), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1)
    )
    ax.text(15.5, ylim[1] - 10, "to pitcher", rotation=90, va="center", color=INK2, fontsize=7.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


from matplotlib.patches import Ellipse


def foot(ax, row, lab, ph, fill, color, alpha=1.0):
    heel = np.array([-12 * row[f"{lab}_heel_x{ph}"], 12 * row[f"{lab}_heel_y{ph}"]])
    toe = np.array(
        [
            -12 * (row[f"{lab}_bigtoe_x{ph}"] + row[f"{lab}_smalltoe_x{ph}"]) / 2,
            12 * (row[f"{lab}_bigtoe_y{ph}"] + row[f"{lab}_smalltoe_y{ph}"]) / 2,
        ]
    )
    v = toe - heel
    L = max(np.linalg.norm(v), 8.0)
    ang = np.degrees(np.arctan2(v[1], v[0]))
    c = (heel + toe) / 2
    ax.add_patch(Ellipse(c, L * 1.25, L * 0.42, angle=ang, fc=color if fill else "none", ec=color, lw=1.6, alpha=alpha))
    return c


def dim(ax, a, b, text, off=(0, 0), color=INK, ha="center", fs=8):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1, shrinkA=0, shrinkB=0))
    ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha=ha, va="center", color=color, fontsize=fs)


# ---- V1: the stance language on a typical hitter
fig, ax = plt.subplots(figsize=(6.4, 5.6))
field(ax, xlim=(-85, 22), ylim=(-60, 48))
fs0 = foot(ax, med, "front", 0, True, BLUE)
bs0 = foot(ax, med, "back", 0, True, BLUE)
fs2 = foot(ax, med, "front", 2, False, ORANGE)
cx, cy = (fs0[0] + bs0[0]) / 2, (fs0[1] + bs0[1]) / 2
ax.plot([cx], [cy], marker="+", color=INK, ms=10, mew=1.6)
ax.text(cx + 2, cy - 4, "center of mass", fontsize=7, color=INK2)
xd = -17
ax.annotate("", xy=(xd, cy), xytext=(xd, 17), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1))
ax.text(xd + 2.5, (cy + 17) / 2, f"DEPTH  {med.depth_in:.1f} in", rotation=90, va="center", fontsize=8, color=INK)
ax.plot([cx, xd], [cy, cy], color=GRID, lw=0.8, ls=":")
yo = bs0[1] - 10
ax.plot([cx, cx], [cy - 2, yo], color=GRID, lw=0.8, ls=":")
ax.plot([-8.5, -8.5], [8.5, yo], color=GRID, lw=0.8, ls=":")
ax.annotate("", xy=(-8.5, yo), xytext=(cx, yo), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1))
ax.text((cx - 8.5) / 2, yo - 4.5, f"OFF THE PLATE  {med.off_plate_in:.1f} in", ha="center", fontsize=8, color=INK)
xw = min(fs0[0], bs0[0]) - 12
ax.annotate("", xy=(xw, fs0[1]), xytext=(xw, bs0[1]), arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=1))
ax.text(
    xw - 3,
    (fs0[1] + bs0[1]) / 2,
    f"WIDTH  {med.width_in:.1f} in",
    rotation=90,
    va="center",
    ha="right",
    fontsize=8,
    color=BLUE,
)
ax.annotate("", xy=fs2, xytext=fs0, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, shrinkA=6, shrinkB=6))
ax.text(
    fs2[0] - 4,
    fs2[1] + 8,
    f"STRIDE {med.stride_len_in:.1f} in → lands {med.width_land_in:.1f} in wide\n(about 57% of the hitter's height)",
    ha="right",
    color=ORANGE,
    fontsize=8,
)
ax.text(
    -84,
    -59,
    "Blue = feet at setup · orange outline = where the front foot lands. Typical 2026 regular (medians of 251),\ndrawn as a right-handed hitter; left-handers are the mirror image.",
    color=INK2,
    fontsize=7.2,
)
fig.tight_layout()
fig.savefig(f"{BASE}/visual/v1_language.png", dpi=220)
plt.close(fig)


# ---- V2: two opposite setups that land alike
def row(name):
    r = F[F.name.str.contains(name)].sort_values("PA", ascending=False).iloc[0]
    return r


a, b = row("Marte, Ketel"), row("Turner, Trea")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.9))
for ax, r in zip(axs, (a, b)):
    field(ax, xlim=(-80, 20), ylim=(-45, 45))
    f0 = foot(ax, r, "front", 0, True, BLUE)
    b0 = foot(ax, r, "back", 0, True, BLUE)
    f2 = foot(ax, r, "front", 2, False, ORANGE)
    ax.annotate("", xy=f2, xytext=f0, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    nm = " ".join(r["name"].split(", ")[::-1])
    ax.set_title(f"{nm}, {r.height_in:.0f} in tall", fontsize=9.5, loc="left", color=INK)
    ax.text(
        -79,
        -43,
        f"Setup {r.width_in:.0f} in wide  →  stride {r.stride_len_in:.0f} in  →  lands {r.width_land_in:.0f} in wide",
        fontsize=8,
        color=INK,
    )
fig.text(
    0.01,
    0.02,
    "Blue = setup, orange = front-foot landing. Marte's left-handed stance is mirrored so both face the same way.",
    fontsize=7.2,
    color=INK2,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{BASE}/visual/v2_marte_turner.png", dpi=220)
plt.close(fig)

# ---- V3: stance fingerprints of well-known hitters
H = pd.read_parquet(f"{BASE}/taxonomy/hitters_pca.parquet")
H26 = H[H.year == "2026"].copy()
for c in ("PC1", "PC2", "PC3", "PC4"):
    H26[c + "_pct"] = H26[c].rank(pct=True) * 100
names = [
    "Crow-Armstrong, Pete",
    "Judge, Aaron",
    "Soto, Juan",
    "Kwan, Steven",
    "Altuve, Jose",
    "Turner, Trea",
    "Burger, Jake",
    "Devers, Rafael",
]
dims = [
    ("PC1_pct", "Setup vs stride", "long\nstride", "wide,\nshort stride"),
    ("PC2_pct", "Stride direction", "steps\nin", "steps\nout"),
    ("PC3_pct", "Reach", "compact", "far off\nthe plate"),
    ("PC4_pct", "Landing", "back foot\ntravels", "front side\nopens"),
]
fig, axs = plt.subplots(len(names) + 1, 4, figsize=(7.2, 5.6), gridspec_kw={"height_ratios": [1.1] + [1] * len(names)})
for j, (col, title, lo, hi) in enumerate(dims):
    ax = axs[0, j]
    ax.axis("off")
    ax.set_xlim(-5, 105)
    ax.text(50, 0.85, title, ha="center", fontsize=9, color=INK, weight="bold", transform=ax.transData)
    ax.text(0, 0.1, lo, ha="left", fontsize=6.6, color=INK2)
    ax.text(100, 0.1, hi, ha="right", fontsize=6.6, color=INK2)
    ax.set_ylim(0, 1.2)
for i, nm in enumerate(names, start=1):
    r = H26[H26.name == nm].sort_values("PA", ascending=False).iloc[0]
    for j, (col, *_rest) in enumerate(dims):
        ax = axs[i, j]
        ax.set_xlim(-5, 105)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.plot([0, 100], [0, 0], color=GRID, lw=3, solid_capstyle="round")
        ax.plot([50, 50], [-0.5, 0.5], color=GRID, lw=0.8)
        ax.plot([r[col]], [0], "o", color=BLUE, ms=7, mec=SURF, mew=1)
        if j == 0:
            ax.text(-6, 0, " ".join(nm.split(", ")[::-1]), ha="right", va="center", fontsize=8, color=INK)
fig.text(
    0.99,
    0.01,
    "Dot = where the hitter ranks among 2026 regulars (0–100; middle tick = league median).",
    ha="right",
    fontsize=7,
    color=INK2,
)
fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.05, wspace=0.18, hspace=0.25)
fig.savefig(f"{BASE}/visual/v3_fingerprints.png", dpi=220)
plt.close(fig)

# ---- V4: which change fixes which problem
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)


def box(x, y, w, h, text, fc=TINT, ec=INK2, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        weight="bold" if bold else "normal",
    )


box(28, 51, 44, 7, "What is the hitter's main problem?", bold=True)
rows4 = [
    ("Chasing\n(chase-prone)", "Stand DEEPER — only vs\nsame-hand pitchers", "moderate", BLUE),
    ("Whiffing,\nbut not chasing", "Stand WIDER — always\n(same vs both hands)", "strong", AQUA),
    ("Whiffing on\noutside pitches", "Stand CLOSER to the plate\n(watch inside pitches)", "moderate", BLUE),
    ("Weak launch angles,\ncontact far out front", "Stand DEEPER —\nfull-time", "moderate", BLUE),
    ("None of these", "Keep the setup\n(no stance is 'best')", "", INK2),
]
for k, (prob, fix, ev, col) in enumerate(rows4):
    y = 40 - k * 9.6
    box(4, y, 26, 7, prob)
    box(46, y, 30, 7, fix, fc=SURF, ec=col)
    ax.annotate("", xy=(45.5, y + 3.5), xytext=(31, y + 3.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.plot([17, 17], [y + 7.4, 50.6] if k == 0 else [y + 7.4, y + 9.6], color=INK2, lw=0.8)
    if ev:
        ax.text(79, y + 3.5, f"evidence: {ev}", va="center", fontsize=7.8, color=INK2)
ax.plot([17, 50], [50.6, 50.6], color=INK2, lw=0.8)
fig.tight_layout()
fig.savefig(f"{BASE}/visual/v4_problem_change.png", dpi=220)
plt.close(fig)

# ---- V5: who should switch
c = pd.read_csv(f"{BASE}/candidates_2026_targeted.csv").head(10)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
field(ax1, xlim=(-80, 22), ylim=(-45, 45))
foot(ax1, med, "front", 0, True, BLUE)
foot(ax1, med, "back", 0, True, BLUE)
shifted = med.copy()
for k in shifted.index:
    if "_y" in k:
        shifted[k] = med[k] - 0.5  # 6 in deeper
foot(ax1, shifted, "front", 0, False, ORANGE)
foot(ax1, shifted, "back", 0, False, ORANGE)

ax1.text(
    -79,
    -43,
    "Filled = usual spot (vs opposite-hand pitchers)\nOrange = 6 in deeper vs same-hand pitchers",
    fontsize=7.4,
    color=INK,
)
ax1.set_title("A chase-prone right-handed hitter", fontsize=9.5, loc="left", color=INK)
y = np.arange(len(c))[::-1]
ax2.hlines(y, c.gain_lo, c.gain_hi, color=GRID, lw=3)
ax2.plot(c.gain_runs, y, "o", color=ORANGE, ms=7, mec=SURF, mew=1)
ax2.set_yticks(y)
ax2.set_yticklabels([" ".join(n.split(", ")[::-1]) + f" ({t})" for n, t in zip(c["name"], c.team)], fontsize=7.6)
ax2.set_xlabel("Est. runs per season (dot) with 95% range", fontsize=7.6, color=INK2)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.spines["bottom"].set_color(GRID)
ax2.tick_params(colors=INK2, labelsize=7.5)
ax2.grid(True, axis="x", color=GRID, lw=0.6)
ax2.set_axisbelow(True)
ax2.set_title("Top 10 candidates (2026)", fontsize=9.5, loc="left", color=INK)
fig.tight_layout()
fig.savefig(f"{BASE}/visual/v5_switch.png", dpi=220)
plt.close(fig)
print("ok", a["name"], a.side, b["name"], b.side)
