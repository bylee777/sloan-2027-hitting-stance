# Read-only sensitivity: re-fit stance_bayes_v2's exact model on its saved event table, dropping depth events one at a time.
import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from scipy.stats import norm, multivariate_normal

rng = np.random.default_rng(7)
PRIOR_SD = {"whiff": 3.0, "chase": 3.0, "sweet": 5.0, "K": 3.0, "xwoba": 30.0, "bat_speed": 1.0}
ch = pd.read_parquet("stance_bayes_v2_events_B.parquet")


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
    s = PRIOR_SD[m]
    P0 = np.eye(Xd.shape[1]) * s**2
    grid = np.linspace(0, 3 * s, 301)
    lp = []
    for tau in grid:
        C = Xd @ P0 @ Xd.T + np.diag(v + tau**2)
        L = np.linalg.cholesky(C)
        z = np.linalg.solve(L, y)
        lp.append(-0.5 * (z @ z) - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * np.pi) + norm.logpdf(tau, 0, s))
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
    return betas


def summ(b, j=5):
    lo, hi = np.quantile(b[:, j], [0.05, 0.95])
    return f"{b[:, j].mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  P(>0)={np.mean(b[:, j] > 0):.2f}"


print("deeper per 3 in, sweet-spot (pp)")
print("  all 150 changes           :", summ(fit(ch, "sweet")))
pca = (ch.name == "Crow-Armstrong, Pete") & (ch.year == "2024")
print("  drop PCA 2024 (depth+width):", summ(fit(ch[~pca], "sweet")))
# merge the 4 same-window width/depth pairs: drop the width half of each pair
pairs = [
    ("Dubón, Mauricio", "2025"),
    ("Hernández, Heriberto", "2025"),
    ("Crow-Armstrong, Pete", "2024"),
    ("Lopez, Nicky", "2024"),
]
dup = ch.apply(lambda r: (r["name"], r.year) in pairs and r.lever == "width", axis=1)
print("  drop width half of 4 overlapping pairs:", summ(fit(ch[~dup], "sweet")))
d = ch[ch.lever == "depth"]
res = []
for i in d.index:
    b = fit(ch.drop(index=i), "sweet", draws=2000)
    lo = np.quantile(b[:, 5], 0.05)
    res.append((ch.loc[i, "name"], ch.loc[i, "year"], round(ch.loc[i, "size"], 1), b[:, 5].mean(), lo))
r = pd.DataFrame(res, columns=["name", "year", "size", "mean", "lo90"]).sort_values("mean")
print(
    "  leave-one-depth-event-out: mean range",
    f"{r['mean'].min():+.2f} .. {r['mean'].max():+.2f};",
    "lower 90% bound < 0 in",
    (r.lo90 < 0).sum(),
    "of",
    len(r),
)
print(r.head(4).round(2).to_string(index=False))
for m, lab in (("chase", "chase"), ("xwoba", "xwOBA")):
    print(f"deeper per 3 in, {lab}: all", summ(fit(ch, m)), "| drop PCA 2024", summ(fit(ch[~pca], m)))
