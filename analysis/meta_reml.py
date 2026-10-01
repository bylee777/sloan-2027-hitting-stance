"""Is the Bayesian stance-change model (stance_bayes_v2.py, BEFORE=B; paper Tables 3-4) driven by its priors or by being Bayesian?
Reads stance_bayes_v2_events_B.parquet (one row per change: y = change minus 20 matched non-changers, v = sampling variance).

A. PRIOR SENSITIVITY: the same exact posterior (tau grid + conjugate beta) with every prior scale s multiplied by
   0.5 / 1 (paper) / 2 / 10 (nearly flat). s sets both beta ~ Normal(0, s^2) and tau ~ Half-Normal(s).
B. FREQUENTIST: REML random-effects meta-regression (the metafor default): tau^2 by restricted maximum likelihood,
   beta by weighted least squares with weights 1 / (v + tau^2); 90% intervals normal-theory and Knapp-Hartung (t, k - p df).
C. VALIDATION under each version: train 2024-25, score the 2026 changes by total log predictive density; intercept-only and
   no-effect models refit as their own models (as in paper Table 4). REML uses the plug-in predictive
   y ~ Normal(x'b, v + tau^2 + x' Cov(b) x)."""

import numpy as np, pandas as pd, warnings
from scipy.stats import norm, t as tdist
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


ch = pd.read_parquet("stance_bayes_v2_events_B.parquet")
ch["year"] = ch.year.astype(str)
MET = ["whiff", "chase", "sweet", "K", "xwoba", "bat_speed"]
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
# the five cells whose 90% interval excludes zero in paper Table 3
BOLD = [("sweet", 5), ("xwoba", 0), ("xwoba", 1), ("K", 0), ("chase", 2)]


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


def data(d, m, cols=None):
    ok = d[f"y_{m}"].notna() & d[f"v_{m}"].notna()
    d = d[ok]
    return d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float), design(d, cols)


# ------------------------------------------------------------------ Bayesian (same model as the paper, prior scale x mult)
def fit_bayes(d, m, rng, mult=1.0, cols=None, draws=4000):
    y, v, Xd = data(d, m, cols)
    k = Xd.shape[1]
    s = PRIOR_SD[m] * mult
    grid = np.linspace(
        0, 3 * PRIOR_SD[m] * max(mult, 1.0), 601
    )  # the likelihood is negligible beyond 3 x the paper's s
    base = Xd @ (np.eye(k) * s**2) @ Xd.T if k else np.zeros((len(y), len(y)))
    lp = []
    for tau in grid:
        L = np.linalg.cholesky(base + np.diag(v + tau**2))
        z = solve_triangular(L, y, lower=True)
        lp.append(-0.5 * (z @ z) - np.log(np.diag(L)).sum() + norm.logpdf(tau, 0, s))
    lp = np.array(lp)
    pt = np.exp(lp - lp.max())
    pt /= pt.sum()
    taus = rng.choice(grid, size=draws, p=pt)
    betas = np.zeros((draws, k))
    cache = {}
    if k:
        for i, tau in enumerate(taus):
            if tau not in cache:
                Wt = 1 / (v + tau**2)
                V = np.linalg.inv(Xd.T @ (Xd * Wt[:, None]) + np.eye(k) / s**2)
                cache[tau] = (V @ (Xd.T @ (Wt * y)), V)
            mu, V = cache[tau]
            betas[i] = rng.multivariate_normal(mu, V)
    return dict(beta=betas, tau=taus, cols=cols)


def lpd_bayes(post, d, m):
    y, v, Xd = data(d, m, post["cols"])
    mu = post["beta"] @ Xd.T if Xd.shape[1] else np.zeros((len(post["tau"]), len(y)))
    sd = np.sqrt(v[None, :] + post["tau"][:, None] ** 2)
    return np.log(norm.pdf(y[None, :], mu, sd).mean(0)).sum()


# ------------------------------------------------------------------ REML random-effects meta-regression
def fit_reml(d, m, cols=None):
    y, v, X = data(d, m, cols)
    k, p = len(y), X.shape[1]

    def nll(tau):
        w = 1 / (v + tau**2)
        if p == 0:
            return 0.5 * (np.log(v + tau**2).sum() + (w * y * y).sum())  # no fixed effects: REML = ML
        XtWX = X.T @ (X * w[:, None])
        b = np.linalg.solve(XtWX, X.T @ (w * y))
        r = y - X @ b
        return 0.5 * (np.log(v + tau**2).sum() + np.linalg.slogdet(XtWX)[1] + (w * r * r).sum())

    hi = 3 * PRIOR_SD[m] * 3
    g = np.linspace(0, hi, 2001)
    vals = np.array([nll(t) for t in g])
    j = int(vals.argmin())
    lo_, hi_ = g[max(j - 1, 0)], g[min(j + 1, len(g) - 1)]
    tau = minimize_scalar(nll, bounds=(lo_, hi_), method="bounded").x if hi_ > lo_ else g[j]
    if nll(0.0) <= nll(tau):
        tau = 0.0
    w = 1 / (v + tau**2)
    if p == 0:
        return dict(b=np.zeros(0), cov=np.zeros((0, 0)), tau=tau, q=np.nan, df=k, cols=cols)
    XtWX = X.T @ (X * w[:, None])
    cov = np.linalg.inv(XtWX)
    b = cov @ (X.T @ (w * y))
    r = y - X @ b
    q = (w * r * r).sum() / (k - p)  # Knapp-Hartung scale
    return dict(b=b, cov=cov, tau=tau, q=q, df=k - p, cols=cols)


def lpd_reml(f, d, m):
    y, v, X = data(d, m, f["cols"])
    mu = X @ f["b"] if X.shape[1] else np.zeros(len(y))
    var = v + f["tau"] ** 2 + (np.einsum("ij,jk,ik->i", X, f["cov"], X) if X.shape[1] else 0.0)
    return norm.logpdf(y, mu, np.sqrt(var)).sum()


MULTS = [0.5, 1.0, 2.0, 10.0]
say("=" * 118)
say(
    "A + B. ESTIMATES (all 2024-26 changes): posterior mean [90% credible interval] by prior scale; REML estimate [90% CI]"
)
say("        (normal) and [90% CI] (Knapp-Hartung t). Paper Table 3 = prior x1. tau = spread of individual responses.")
say("=" * 118)
EST = {}
for m in MET:
    y, v, X = data(ch, m)
    say(f"\n--- {m}: n {len(y)} changes")
    res = {}
    for mult in MULTS:
        rng = np.random.default_rng(7)
        post = fit_bayes(ch, m, rng, mult=mult)
        b = post["beta"]
        lo, hi = np.quantile(b, [0.05, 0.95], axis=0)
        res[f"prior x{mult:g}"] = (b.mean(0), lo, hi, post["tau"].mean())
    f = fit_reml(ch, m)
    se = np.sqrt(np.diag(f["cov"]))
    z90 = norm.ppf(0.95)
    res["REML"] = (f["b"], f["b"] - z90 * se, f["b"] + z90 * se, f["tau"])
    seKH = np.sqrt(np.diag(f["cov"]) * f["q"])
    tcrit = tdist.ppf(0.95, f["df"])
    res["REML-KH"] = (f["b"], f["b"] - tcrit * seKH, f["b"] + tcrit * seKH, f["tau"])
    EST[m] = res
    say(f"{'term':22s} " + " ".join(f"{k:>25s}" for k in res))
    for j in range(len(NAMES)):
        say(f"{NAMES[j]:22s} " + " ".join(f"{r[0][j]:+7.2f} [{r[1][j]:+6.2f},{r[2][j]:+6.2f}]" for r in res.values()))
    say(
        f"{'tau':22s} "
        + " ".join(f"{r[3]:25.2f}" for r in res.values())
        + f"   (REML Knapp-Hartung scale q = {f['q']:.2f})"
    )

say("\n" + "=" * 118)
say("SUMMARY: the five Table 3 cells whose 90% interval excludes zero — does each still exclude zero?")
say("=" * 118)
say(f"{'cell':38s} " + " ".join(f"{k:>18s}" for k in EST["sweet"]))
for m, j in BOLD:
    cells = []
    for k, r in EST[m].items():
        mark = "*" if (r[1][j] > 0 or r[2][j] < 0) else " "
        cells.append(f"{r[0][j]:+7.2f} [{r[1][j]:+.1f},{r[2][j]:+.1f}]{mark}")
    say(f"{m + ': ' + NAMES[j]:38s} " + " ".join(f"{c:>18s}" for c in cells))
say("(* = 90% interval excludes zero)")

say("\n" + "=" * 118)
say(
    "C. VALIDATION: train 2024-25, score the 2026 changes (total log predictive density; intercept-only and no-effect REFIT)"
)
say("   columns per version: full minus no-effect | full minus intercept-only. Paper Table 4 = prior x1.")
say("=" * 118)
tr, te = ch[ch.year != "2026"], ch[ch.year == "2026"]
vers = [f"prior x{mult:g}" for mult in MULTS] + ["REML plug-in"]
say(f"{'outcome':10s} " + " ".join(f"{k:>22s}" for k in vers) + "   n test")
for m in MET:
    cells = []
    for mult in MULTS:
        rng = np.random.default_rng(11)
        full = lpd_bayes(fit_bayes(tr, m, rng, mult=mult), te, m)
        ic = lpd_bayes(fit_bayes(tr, m, rng, mult=mult, cols=[0, 1, 2]), te, m)
        ne = lpd_bayes(fit_bayes(tr, m, rng, mult=mult, cols=[]), te, m)
        cells.append(f"{full - ne:+7.2f} | {full - ic:+7.2f}")
    full = lpd_reml(fit_reml(tr, m), te, m)
    ic = lpd_reml(fit_reml(tr, m, [0, 1, 2]), te, m)
    ne = lpd_reml(fit_reml(tr, m, []), te, m)
    cells.append(f"{full - ne:+7.2f} | {full - ic:+7.2f}")
    say(f"{m:10s} " + " ".join(f"{c:>22s}" for c in cells) + f"   {te[f'y_{m}'].notna().sum()}")

say("\n" + "=" * 118)
say("D. HOW WELL IS tau (the spread of individual hitters' true responses) IDENTIFIED? (all 2024-26 changes)")
say("   Bayesian posterior (paper prior) vs REML estimate with its profile-likelihood 90% interval (2 x drop <= 2.71);")
say("   typical sampling SD of one change's y for scale.")
say("=" * 118)
say(
    f"{'outcome':10s} {'post. mean':>10s} {'median':>8s} {'90% interval':>18s} | {'REML':>6s} {'profile 90% CI':>18s} | typical sampling SD"
)
for m in MET:
    rng = np.random.default_rng(7)
    t_ = fit_bayes(ch, m, rng)["tau"]
    y, v, X = data(ch, m)
    f = fit_reml(ch, m)

    def nll(tau):
        w = 1 / (v + tau**2)
        XtWX = X.T @ (X * w[:, None])
        b = np.linalg.solve(XtWX, X.T @ (w * y))
        r = y - X @ b
        return 0.5 * (np.log(v + tau**2).sum() + np.linalg.slogdet(XtWX)[1] + (w * r * r).sum())

    g = np.linspace(0, 9 * PRIOR_SD[m], 3001)
    prof = np.array([nll(x) for x in g])
    ok = 2 * (prof - prof.min()) <= 2.706
    say(
        f"{m:10s} {t_.mean():10.2f} {np.median(t_):8.2f} [{np.quantile(t_, .05):6.2f}, {np.quantile(t_, .95):6.2f}] | "
        f"{f['tau']:6.2f} [{g[ok].min():6.2f}, {g[ok].max():6.2f}] | {np.sqrt(np.median(v)):.1f}"
    )
open("meta_reml.txt", "w").write("\n".join(LOG) + "\n")
