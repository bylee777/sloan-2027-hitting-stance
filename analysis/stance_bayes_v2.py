"""[v2: + STANCE-ANGLE changes; angle events within 30 days of a width/depth change dropped so each change is isolated]
Bayesian hierarchical model of how hitters RESPOND to a stance change (random-effects meta-regression).

Each sustained stance change i (width or depth event) gives an effect y_i for an outcome: the hitter's change from the
30 days before to the 60 days after, MINUS the same change for 20 matched hitters in a similar slump who did not change
(so bounce-back is removed). Model:
    y_i ~ Normal(theta_i, v_i)                 v_i = known sampling variance from the PA / swing counts
    theta_i ~ Normal(x_i' beta, tau^2)         tau = how much individual hitters' true responses differ
    beta ~ Normal(0, s^2)  (weakly informative), tau ~ Half-Normal(s)
x_i: lever intercepts (the 'change after a slump' effect, width / depth) + signed size of the change per 3 in
     (width: + = wider; depth: + = deeper) + wider x chase-prone (widening helps non-chasers only, pre-registered).
Posterior is exact: tau on a fine grid (marginal likelihood), beta | tau conjugate normal.
VALIDATION: fit on 2024-25 changes, predict the 2026 changes (log predictive density vs a no-effect model and an
intercept-only model; 80% interval coverage). Then fit all seasons and predict Blue Jays recommendations."""

import numpy as np, pandas as pd, warnings, json, os

BEFORE = os.environ.get("BEFORE", "B")  # B = days -90..-31 (primary), U = days -30..-1
warnings.filterwarnings("ignore")
from scipy.stats import norm, multivariate_normal

rng = np.random.default_rng(7)
D = pd.read_parquet("hitter_day_outcomes.parquet")
D["t"] = (pd.to_datetime(D.game_date) - pd.Timestamp("2024-01-01")).dt.days
COLS = ["PA", "xw_sum", "K", "sw", "wh", "oz", "oz_sw", "bip", "sweet", "bs_sum", "bs_n"]
MET = {
    "whiff": ("wh", "sw", 100),
    "chase": ("oz_sw", "oz", 100),
    "sweet": ("sweet", "bip", 100),
    "K": ("K", "PA", 100),
    "xwoba": ("xw_sum", "PA", 1000),
    "bat_speed": ("bs_sum", "bs_n", 1),
}
UNIT_VAR = {
    "whiff": 0.23 * 0.77 * 1e4,
    "chase": 0.30 * 0.70 * 1e4,
    "sweet": 0.35 * 0.65 * 1e4,
    "K": 0.22 * 0.78 * 1e4,
    "xwoba": 366.0**2,
    "bat_speed": 4.9**2,
}
PRIOR_SD = {"whiff": 3.0, "chase": 3.0, "sweet": 5.0, "K": 3.0, "xwoba": 30.0, "bat_speed": 1.0}
W = pd.read_csv("width_events.csv")
W["lever"] = "width"
W["size"] = W.persist  # + = wider
E = pd.read_csv("events_thr4.csv")
E["lever"] = "depth"
E["size"] = E.persist  # + = deeper (BACK)
A = pd.read_csv("angle_events.csv")
A["lever"] = "angle"
A["size"] = A.persist  # + = more CLOSED (deg)
wd = pd.concat([W, E])
wd["t0"] = pd.to_datetime(wd.event_date)
A["t0"] = pd.to_datetime(A.event_date)
ov = A.apply(lambda r: ((wd.bsy == r.bsy) & ((wd.t0 - r.t0).dt.days.abs() <= 30)).any(), axis=1)
print(f"angle events: {len(A)}; dropped {ov.sum()} within 30 days of a width/depth change")
A = A[~ov]
EV = pd.concat([W, E, A.drop(columns=["t0"])], ignore_index=True)
EV["year"] = EV.year.astype(str)
EV["t"] = (pd.to_datetime(EV.event_date) - pd.Timestamp("2024-01-01")).dt.days
# prior-season rates (baseline fallback for early-season changes)
S = D.groupby(["batter", "side", "year"])[COLS].sum().reset_index()
S["xwoba_prev"] = 1000 * S.xw_sum / S.PA
S["whiff_prev"] = 100 * S.wh / S.sw
S["chase_prev"] = 100 * S.oz_sw / S.oz
S["year"] = (S.year.astype(int) + 1).astype(str)
S = S[S.PA >= 200].set_index(["batter", "side", "year"])
chase_cut = (
    D.groupby(["batter", "side", "year"])[["oz_sw", "oz"]]
    .sum()
    .pipe(lambda z: z[z.oz >= 300])
    .pipe(lambda z: z.oz_sw / z.oz)
    .groupby(level="year")
    .quantile(2 / 3)
)
rows = []
for (b, s, y), g in D.groupby(["batter", "side", "year"]):
    g = g.sort_values("t")
    tt = g.t.to_numpy()
    cs = np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)])
    win = lambda lo, hi: cs[np.searchsorted(tt, hi, "right")] - cs[np.searchsorted(tt, lo, "left")]
    evs = EV[(EV.batter == b) & (EV.side == s) & (EV.year == y)]
    refs = [(int(e.t), e.lever, e["size"], e["name"]) for _, e in evs.iterrows()]
    evt = evs.t.to_numpy()
    for r in range(int(tt.min()) + 30, int(tt.max()) - 59, 3):
        if len(evt) == 0 or np.abs(evt - r).min() > 90:
            refs.append((r, "control", 0.0, None))
    prev = S.loc[(b, s, y)] if (b, s, y) in S.index else None
    for r, lever, size, nm in refs:
        B, U, A = win(r - 90, r - 31), win(r - 30, r - 1), win(r, r + 59)
        if U[0] < 40 or A[0] < 60:
            continue
        row = dict(batter=b, side=s, year=y, r=r, lever=lever, size=size, name=nm)
        for m, (n, d, sc) in MET.items():
            i, j = COLS.index(n), COLS.index(d)
            for lab, V_ in (("U", U), ("A", A), ("B", B)):
                row[f"{m}_{lab}"] = sc * V_[i] / V_[j] if V_[j] > 0 else np.nan
                row[f"n_{m}_{lab}"] = V_[j]
        if B[0] >= 40:
            row["xw_base"], row["wh_base"] = 1000 * B[1] / B[0], 100 * B[4] / B[3]
            row["chase_base"] = 100 * B[6] / B[5]
        elif prev is not None and BEFORE == "U":
            row["xw_base"], row["wh_base"], row["chase_base"] = prev.xwoba_prev, prev.whiff_prev, prev.chase_prev
        else:
            continue
        rows.append(row)
X = pd.DataFrame(rows)
X["slump_xw"] = X.xwoba_U - X.xw_base
X["slump_wh"] = X.whiff_U - X.wh_base
X["chase_prone"] = (X.chase_base / 100 > X.year.map(chase_cut)).astype(float)
ch, co = X[X.lever != "control"].copy(), X[X.lever == "control"]
Z = ["slump_xw", "slump_wh", "xw_base"]
sdz = co[Z].std()
for m in MET:
    ch[f"y_{m}"] = np.nan
    ch[f"v_{m}"] = np.nan
for i, e in ch.iterrows():
    pool = co[(co.year == e.year) & ((co.r - e.r).abs() <= 30) & (co.batter != e.batter)].dropna(subset=Z)
    dist = (((pool[Z].astype(float) - e[Z].astype(float)) / sdz) ** 2).sum(1).astype(float)
    mt = pool.loc[dist.nsmallest(20).index]
    for m in MET:
        ch.loc[i, f"y_{m}"] = (e[f"{m}_A"] - e[f"{m}_{BEFORE}"]) - (mt[f"{m}_A"] - mt[f"{m}_{BEFORE}"]).mean()
        nU, nA = e[f"n_{m}_{BEFORE}"], e[f"n_{m}_A"]
        ch.loc[i, f"v_{m}"] = UNIT_VAR[m] * (1 / max(nU, 1) + 1 / max(nA, 1)) * (1 + 1 / 20)
print(
    f"stance changes with usable windows: {len(ch)} of {len(EV)} "
    f"({(ch.lever == 'width').sum()} width, {(ch.lever == 'depth').sum()} depth, {(ch.lever == 'angle').sum()} angle; 2024-25 {(ch.year != '2026').sum()}, 2026 {(ch.year == '2026').sum()}); "
    f"control dates {len(co):,}"
)


def design(d):
    w, dp, an = (
        (d.lever == "width").astype(float),
        (d.lever == "depth").astype(float),
        (d.lever == "angle").astype(float),
    )
    return np.column_stack(
        [w, dp, an, w * d["size"] / 3, w * d["size"] / 3 * d.chase_prone, dp * d["size"] / 3, an * d["size"] / 10]
    )


NAMES = [
    "after-slump change: width",
    "after-slump change: depth",
    "after-slump change: angle",
    "per 3 in WIDER",
    "  extra if chase-prone (wider)",
    "per 3 in DEEPER",
    "per 10 deg more CLOSED",
]
TAU_GRID = lambda s: np.linspace(0, 3 * s, 301)


def fit(d, m, draws=4000):
    ok = d[f"y_{m}"].notna() & d[f"v_{m}"].notna()
    d = d[ok]
    y, v, Xd = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float), design(d)
    s = PRIOR_SD[m]
    P0 = np.eye(Xd.shape[1]) * s**2
    grid = TAU_GRID(s)
    lp = []
    for tau in grid:
        C = Xd @ P0 @ Xd.T + np.diag(v + tau**2)
        lp.append(multivariate_normal.logpdf(y, np.zeros(len(y)), C, allow_singular=True) + norm.logpdf(tau, 0, s))
    lp = np.array(lp)
    pt = np.exp(lp - lp.max())
    pt /= pt.sum()
    taus = rng.choice(grid, size=draws, p=pt)
    betas = np.empty((draws, Xd.shape[1]))
    cache = {}
    for k, tau in enumerate(taus):
        if tau not in cache:
            Wt = 1 / (v + tau**2)
            V = np.linalg.inv(Xd.T @ (Xd * Wt[:, None]) + np.eye(Xd.shape[1]) / s**2)
            cache[tau] = (V @ (Xd.T @ (Wt * y)), V)
        mu, V = cache[tau]
        betas[k] = rng.multivariate_normal(mu, V)
    return dict(beta=betas, tau=taus, n=len(y))


def lpd(post, d, m):
    """log predictive density of held-out changes, averaging over posterior draws."""
    ok = d[f"y_{m}"].notna()
    d = d[ok]
    y, v, Xd = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float), design(d)
    mu = post["beta"] @ Xd.T
    sd = np.sqrt(v[None, :] + post["tau"][:, None] ** 2)
    dens = norm.pdf(y[None, :], mu, sd).mean(0)
    lo, hi = np.quantile(mu + rng.standard_normal(mu.shape) * sd, [0.1, 0.9], axis=0)
    return np.log(dens).sum(), ((y >= lo) & (y <= hi)).mean(), np.corrcoef(mu.mean(0), y)[0, 1], len(y)


def restricted(post, keep):
    q = dict(post)
    q["beta"] = post["beta"] * np.isin(np.arange(post["beta"].shape[1]), keep)[None, :]
    return q


tr, te = ch[ch.year != "2026"], ch[ch.year == "2026"]
print("\n=== VALIDATION: fit on 2024-25 changes, predict the 2026 changes (higher log density = better)")
print(
    f"{'outcome':10s} {'full model':>11s} {'intercept only':>15s} {'no effect':>10s} | full minus no-effect | 80% coverage | corr(pred, obs) | n"
)
VAL = {}
for m in MET:
    post = fit(tr, m)
    full, cov, cor, n = lpd(post, te, m)
    ic, _, _, _ = lpd(restricted(post, [0, 1, 2]), te, m)
    ne, _, _, _ = lpd(restricted(post, []), te, m)
    VAL[m] = dict(full=full, intercept=ic, none=ne, coverage=cov, corr=cor, n=n)
    print(f"{m:10s} {full:11.1f} {ic:15.1f} {ne:10.1f} | {full - ne:+8.2f} | {cov:5.0%} | {cor:+.2f} | {n}")
print(
    "\n=== FULL POSTERIOR (all seasons): posterior mean [90% credible interval]; tau = spread of individual responses"
)
POST = {}
for m in MET:
    post = fit(ch, m)
    POST[m] = post
    b = post["beta"]
    lo, hi = np.quantile(b, [0.05, 0.95], axis=0)
    cells = [f"{NAMES[j].strip()}: {b[:, j].mean():+.2f} [{lo[j]:+.2f}, {hi[j]:+.2f}]" for j in range(b.shape[1])]
    print(f"{m:10s} tau {post['tau'].mean():.2f} | " + " | ".join(cells))
ch.to_parquet(f"stance_bayes_v2_events_{BEFORE}.parquet")
import pickle

pickle.dump(POST, open(f"stance_bayes_v2_post_{BEFORE}.pkl", "wb"))
pd.DataFrame(VAL).T.to_csv(f"stance_bayes_v2_validation_{BEFORE}.csv")
