"""Follow-up to dr_counterfactual.py (2026-09-30): the LATE-SEASON asymmetry.
A control date needs a full 60-day 'after' window (r < last game - 59), so no control falls after ~late July; a changer only
needs 60 PA after the change, so 35 of the 135 analyzed changes (27 of 83 width) come later, with a truncated after window
at season's end, and are compared with earlier controls. Here: every counterfactual refit (a) on all changes and (b) without
the late changes (fewer than 59 days of season left after the change). Reads dr_counterfactual_events_{B,U}.parquet."""

import numpy as np, pandas as pd, warnings
from scipy.stats import norm
from scipy.linalg import solve_triangular

warnings.filterwarnings("ignore")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


T0 = pd.Timestamp("2024-01-01")
D = pd.read_parquet("hitter_day_outcomes.parquet")
D["t"] = (pd.to_datetime(D.game_date) - T0).dt.days
D["year"] = D.year.astype(str)
TMAX = D.groupby(["batter", "side", "year"]).t.max().rename("tmax").reset_index()
PRIOR_SD = {"whiff": 3.0, "chase": 3.0, "sweet": 5.0, "K": 3.0, "xwoba": 30.0, "bat_speed": 1.0}
NAMES = [
    "after-slump: width",
    "after-slump: depth",
    "after-slump: angle",
    "per 3 in WIDER",
    "wider x chase-prone",
    "per 3 in DEEPER",
    "per 10 deg CLOSED",
]


def design(d, cols=None):
    w, dp, an = (
        (d.lever == "width").astype(float),
        (d.lever == "depth").astype(float),
        (d.lever == "angle").astype(float),
    )
    X = np.column_stack(
        [w, dp, an, w * d["size"] / 3, w * d["size"] / 3 * d.chase_prone, dp * d["size"] / 3, an * d["size"] / 10]
    )
    return X if cols is None else X[:, cols]


def fit(d, ycol, vcol, m, rg, cols=None, draws=4000):
    ok = d[ycol].notna() & d[vcol].notna()
    d = d[ok]
    y, v = d[ycol].to_numpy(float), d[vcol].to_numpy(float)
    Xd = design(d, cols)
    k = Xd.shape[1]
    s = PRIOR_SD[m]
    grid = np.linspace(0, 3 * s, 301)
    lp = []
    base = Xd @ (np.eye(k) * s**2) @ Xd.T if k else np.zeros((len(y), len(y)))
    for tau in grid:
        L = np.linalg.cholesky(base + np.diag(v + tau**2))
        z = solve_triangular(L, y, lower=True)
        lp.append(-0.5 * (z @ z) - np.log(np.diag(L)).sum() + norm.logpdf(tau, 0, s))
    lp = np.array(lp)
    pt = np.exp(lp - lp.max())
    pt /= pt.sum()
    taus = rg.choice(grid, size=draws, p=pt)
    betas = np.zeros((draws, k))
    cache = {}
    for i, tau in enumerate(taus):
        if tau not in cache:
            Wt = 1 / (v + tau**2)
            V = np.linalg.inv(Xd.T @ (Xd * Wt[:, None]) + np.eye(k) / s**2)
            cache[tau] = (V @ (Xd.T @ (Wt * y)), V)
        mu, V = cache[tau]
        betas[i] = rg.multivariate_normal(mu, V)
    return dict(beta=betas, tau=taus)


B = pd.read_parquet("dr_counterfactual_events_B.parquet").merge(TMAX, on=["batter", "side", "year"], how="left")
B["late"] = (B.tmax - B.r) < 59
say("=" * 110)
say(
    f"B design: {B.y_nn_xwoba.notna().sum()} changes with outcomes, {int((B.late & B.y_nn_xwoba.notna()).sum())} late "
    f"(< 59 days of season left): "
    + ", ".join(
        f"{l} {int((B.late & B.y_nn_xwoba.notna() & (B.lever == l)).sum())}/{int((B.y_nn_xwoba.notna() & (B.lever == l)).sum())}"
        for l in ("width", "depth", "angle")
    )
)
say("=" * 110)
say("Mean effect per lever, late vs the rest (paper matching | gbm counterfactual), xwOBA points:")
for l in ("width", "depth", "angle"):
    for lab, sel in (("late", B.late), ("not late", ~B.late)):
        g = B[(B.lever == l) & sel & B.y_nn_xwoba.notna()]
        if len(g):
            say(
                f"  {l:6s} {lab:9s} n {len(g):3d} | match20 {g.y_nn_xwoba.mean():+6.1f} (SE {g.y_nn_xwoba.std() / np.sqrt(len(g)):4.1f}) | "
                f"gbm {g.y_gbm_xwoba.mean():+6.1f} (SE {g.y_gbm_xwoba.std() / np.sqrt(len(g)):4.1f}) | naive {g.d_xwoba.mean():+6.1f}"
            )
CELLS = [("sweet", 5), ("xwoba", 0), ("xwoba", 1), ("K", 0), ("chase", 2), ("whiff", 3)]
say("\nBayesian model, posterior mean [90% interval] (* excludes 0): all changes vs without the late changes")
say(
    f"  {'cell':30s} "
    + " ".join(f"{k:>23s}" for k in ("match20 all", "match20 no-late", "bias-corr no-late", "gbm all", "gbm no-late"))
)
for m, j in CELLS:
    cells = []
    for k, sub in (("nn", B), ("nn", B[~B.late]), ("bc", B[~B.late]), ("gbm", B), ("gbm", B[~B.late])):
        b = fit(sub, f"y_{k}_{m}", f"v_{m}", m, np.random.default_rng(7))["beta"][:, j]
        lo, hi = np.quantile(b, [0.05, 0.95])
        cells.append(f"{b.mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if lo > 0 or hi < 0 else ' '}")
    say(f"  {m + ': ' + NAMES[j]:30s} " + " ".join(f"{c:>23s}" for c in cells))

U = pd.read_parquet("dr_counterfactual_events_U.parquet").merge(TMAX, on=["batter", "side", "year"], how="left")
U["late"] = (U.tmax - U.r) < 59
U["kind"] = U.lever + ":" + U.dir.astype(str)
say("\n" + "=" * 110)
say("U design (Figure 3): effect beyond bounce-back, xwOBA points, all vs without late changes (match20 | gbm)")
say("=" * 110)
for kind in ("width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"):
    cells = []
    for lab, sel in (("all", U.kind == kind), ("no-late", (U.kind == kind) & ~U.late)):
        g = U[sel & U.y_nn_xwoba.notna()]
        cells.append(
            f"{lab} n {len(g):3d}: {g.y_nn_xwoba.mean():+6.1f} ({g.y_nn_xwoba.std() / np.sqrt(len(g)):4.1f}) | {g.y_gbm_xwoba.mean():+6.1f} ({g.y_gbm_xwoba.std() / np.sqrt(len(g)):4.1f})"
        )
    say(f"  {kind:15s} " + "   ".join(cells))
wd = U[U.y_nn_xwoba.notna()]
for lab, sel in (("all", wd.lever.notna()), ("no-late", ~wd.late)):
    g = wd[sel]
    say(
        f"  bounce-back share ({lab}, n {len(g)}): match20 {g.cf_nn_xwoba.mean() / g.d_xwoba.mean():.0%} | gbm {g.cf_gbm_xwoba.mean() / g.d_xwoba.mean():.0%}"
        f"  (raw gain {g.d_xwoba.mean():+.1f})"
    )
open("dr_followup.txt", "w").write("\n".join(LOG) + "\n")
