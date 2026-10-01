"""External check of the run-value accounting: do a hitter's parts in season N predict his NEXT season better than the
standard stats (wOBA, xwOBA) or his season-N total? Pairs 2024->25 and 2025->26, 300+ PA both seasons.
Targets (season N+1): wOBA (a neutral public stat) and our total run value. Predictors from season N:
 (a) wOBA  (b) xwOBA  (c) total run value  (d) the six parts with learned weights  (e) the four skill parts only
 (f) xwOBA + the parts. Out of sample two ways: 10-fold CV grouped by hitter, and train on 2024->25, test on 2025->26.
"""

import glob, numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

H = pd.read_csv("hitter_diagnosis.csv", dtype={"year": str})
# wOBA and xwOBA per hitter-season (standard definitions: per wOBA denominator)
ev = []
for f in glob.glob("raw/*.parquet"):
    d = pd.read_parquet(
        f,
        columns=[
            "batter",
            "stand",
            "game_date",
            "events",
            "woba_value",
            "woba_denom",
            "estimated_woba_using_speedangle",
        ],
    )
    ev.append(d[d.events.notna()])
e = pd.concat(ev)
e["year"] = pd.to_datetime(e.game_date).dt.year.astype(str)
e["side"] = e.stand
for c in ("woba_value", "woba_denom", "estimated_woba_using_speedangle"):
    e[c] = pd.to_numeric(e[c], errors="coerce").astype(float)
e = e[e.woba_denom == 1]
e["xw"] = np.where(e.estimated_woba_using_speedangle.notna(), e.estimated_woba_using_speedangle, e.woba_value)
W = e.groupby(["batter", "side", "year"]).agg(wOBA=("woba_value", "mean"), xwOBA=("xw", "mean")).reset_index()
W[["wOBA", "xwOBA"]] *= 1000
D = H.merge(W, on=["batter", "side", "year"])
PARTS = ["DECISIONS", "WHIFFS", "INPUTS", "SQUARING", "LUCK", "OTHER"]
a = D[D.PA >= 300]
nx = a.assign(year=(a.year.astype(int) - 1).astype(str))
pr = a.merge(nx, on=["batter", "side", "year"], suffixes=("", "_n"))
print(f"hitter pairs: {len(pr)} ({(pr.year == '2024').sum()} for 2024->25, {(pr.year == '2025').sum()} for 2025->26)")
SETS = {
    "(a) wOBA": ["wOBA"],
    "(b) xwOBA": ["xwOBA"],
    "(c) total run value": ["TOTAL"],
    "(d) six parts": PARTS,
    "(e) four skill parts": ["DECISIONS", "WHIFFS", "INPUTS", "SQUARING"],
    "(f) xwOBA + six parts": ["xwOBA"] + PARTS,
}


def oos(cols, y):
    pred = np.zeros(len(pr))
    for tr, te in GroupKFold(10).split(pr, groups=pr.batter):
        m = LinearRegression().fit(pr.iloc[tr][cols], pr.iloc[tr][y], sample_weight=pr.iloc[tr].PA_n)
        pred[te] = m.predict(pr.iloc[te][cols])
    w = pr.PA_n
    r2 = 1 - np.average((pr[y] - pred) ** 2, weights=w) / np.average(
        (pr[y] - np.average(pr[y], weights=w)) ** 2, weights=w
    )
    tr, te = pr.year == "2024", pr.year == "2025"
    m = LinearRegression().fit(pr.loc[tr, cols], pr.loc[tr, y], sample_weight=pr.loc[tr].PA_n)
    pt = m.predict(pr.loc[te, cols])
    wt = pr.loc[te].PA_n
    yt = pr.loc[te, y]
    r2t = 1 - np.average((yt - pt) ** 2, weights=wt) / np.average((yt - np.average(yt, weights=wt)) ** 2, weights=wt)
    return r2, r2t, m


for y, lab in (("wOBA_n", "NEXT-SEASON wOBA"), ("TOTAL_n", "NEXT-SEASON total run value")):
    print(f"\n=== predicting {lab}: out-of-sample R^2 (10-fold by hitter | train 2024->25, test 2025->26)")
    base = None
    for nm, cols in SETS.items():
        r2, r2t, m = oos(cols, y)
        if nm.startswith("(b)"):
            base = (r2, r2t)
        print(f"   {nm:24s} {r2:6.3f} | {r2t:6.3f}")
        if nm.startswith("(d)"):
            print(
                "      learned weights (per run of each part): "
                + ", ".join(f"{c.lower()} {w:+.2f}" for c, w in zip(cols, m.coef_))
            )
# bootstrap the gain of (d) over (b) for next-season wOBA (10-fold R^2), resampling hitters
rng = np.random.default_rng(0)
gains = []
ids = pr.batter.unique()
for _ in range(200):
    pick = rng.choice(ids, len(ids))
    idx = np.concatenate([np.where(pr.batter.to_numpy() == i)[0] for i in pick])
    q = pr.iloc[idx].reset_index(drop=True)

    def r2q(cols):
        pred = np.zeros(len(q))
        for tr, te in GroupKFold(10).split(q, groups=q.batter):
            m = LinearRegression().fit(q.iloc[tr][cols], q.iloc[tr].wOBA_n, sample_weight=q.iloc[tr].PA_n)
            pred[te] = m.predict(q.iloc[te][cols])
        w = q.PA_n
        return 1 - np.average((q.wOBA_n - pred) ** 2, weights=w) / np.average(
            (q.wOBA_n - np.average(q.wOBA_n, weights=w)) ** 2, weights=w
        )

    gains.append(r2q(PARTS) - r2q(["xwOBA"]))
g = np.array(gains)
print(
    f"\nsix parts minus xwOBA, next-season wOBA R^2: {g.mean():+.3f} (95% {np.quantile(g, .025):+.3f} to {np.quantile(g, .975):+.3f}); better in {100*(g > 0).mean():.0f}% of resamples"
)
