"""Paper Table 6: predicted effect of each recommended Blue Jays stance change from the Bayesian stance-change model
(stance_bayes_v2.py, BEFORE=B; refit here from stance_bayes_v2_events_B.parquet). Directional part only: a planned offseason
change is not a slump response, so the after-slump intercepts are excluded. 'avg' = a typical hitter (beta only); P(better) =
posterior probability that the effect is in the helpful direction. Writes jays_predictions_B.csv."""

import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.linalg import solve_triangular

ch = pd.read_parquet("stance_bayes_v2_events_B.parquet")
MET = ["whiff", "chase", "sweet", "K", "xwoba", "bat_speed"]
PRIOR_SD = {"whiff": 3.0, "chase": 3.0, "sweet": 5.0, "K": 3.0, "xwoba": 30.0, "bat_speed": 1.0}
rng = np.random.default_rng(7)


def design(d):
    w, dp, an = (
        (d.lever == "width").astype(float),
        (d.lever == "depth").astype(float),
        (d.lever == "angle").astype(float),
    )
    return np.column_stack(
        [w, dp, an, w * d["size"] / 3, w * d["size"] / 3 * d.chase_prone, dp * d["size"] / 3, an * d["size"] / 10]
    )


def fit(d, m, draws=4000):
    ok = d[f"y_{m}"].notna() & d[f"v_{m}"].notna()
    d = d[ok]
    y, v, Xd = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float), design(d)
    k = Xd.shape[1]
    s = PRIOR_SD[m]
    grid = np.linspace(0, 3 * s, 301)
    lp = []
    base = Xd @ (np.eye(k) * s**2) @ Xd.T
    for tau in grid:
        L = np.linalg.cholesky(base + np.diag(v + tau**2))
        z = solve_triangular(L, y, lower=True)
        lp.append(-0.5 * (z @ z) - np.log(np.diag(L)).sum() + norm.logpdf(tau, 0, s))
    lp = np.array(lp)
    pt = np.exp(lp - lp.max())
    pt /= pt.sum()
    taus = rng.choice(grid, size=draws, p=pt)
    betas = np.empty((draws, k))
    cache = {}
    for i, tau in enumerate(taus):
        if tau not in cache:
            Wt = 1 / (v + tau**2)
            V = np.linalg.inv(Xd.T @ (Xd * Wt[:, None]) + np.eye(k) / s**2)
            cache[tau] = (V @ (Xd.T @ (Wt * y)), V)
        mu, V = cache[tau]
        betas[i] = rng.multivariate_normal(mu, V)
    return dict(beta=betas, tau=taus)


POST = {m: fit(ch, m) for m in MET}
# (name, lever, size in inches, chase-prone)
JAYS = [
    ("Okamoto, Kazuma", "width", +6.0, 0.0),
    ("Clement, Ernie", "depth", +3.5, 1.0),
    ("Kirk, Alejandro", "depth", +3.5, 0.0),
    ("Giménez, Andrés", "depth", +3.0, 1.0),
]
out = []
for nm, lever, size, cp in JAYS:
    x = np.zeros(7)
    if lever == "width":
        x[3] = size / 3
        x[4] = size / 3 * cp
    else:
        x[5] = size / 3
    for m in MET:
        mu = POST[m]["beta"] @ x
        ind = mu + rng.standard_normal(len(mu)) * POST[m]["tau"]
        out.append(
            dict(
                name=nm,
                lever=lever,
                size=size,
                outcome=m,
                mean=mu.mean(),
                lo=np.quantile(mu, 0.1),
                hi=np.quantile(mu, 0.9),
                ind_lo=np.quantile(ind, 0.1),
                ind_hi=np.quantile(ind, 0.9),
                p_better=(mu < 0).mean() if m in ("whiff", "chase", "K") else (mu > 0).mean(),
            )
        )
J = pd.DataFrame(out)
J.to_csv("jays_predictions_B.csv", index=False)
print("BLUE JAYS: predicted effect of the recommended change (directional part; 80% intervals; P(better))")
for nm, g in J.groupby("name", sort=False):
    r0 = g.iloc[0]
    print(f"  {nm}: {r0['size']:+.1f} in {'wider' if r0.lever == 'width' else 'deeper'}")
    for _, r in g.iterrows():
        print(f"     {r.outcome:9s} {r['mean']:+6.2f} [{r.lo:+6.2f}, {r.hi:+6.2f}]  P(better) {r.p_better:.0%}")
