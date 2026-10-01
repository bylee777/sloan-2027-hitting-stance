"""Posture control for the stance-change models (2024-25 only: the 2026 zone is a fixed share of height and carries no posture).
Posture = mean Statcast zone top on TAKEN pitches in a window (inches). For each change, its posture shift is measured exactly like the
outcomes: the changer's (after - before) minus the mean over its 20 matched slump-alike non-changers.
A. Bayesian stance-change model (stance_bayes_v2.py procedure: BEFORE = baseline r-90..r-31, after r..r+59; width, depth and angle changes;
   same matching, priors and exact posterior), fit on 2024-25 changes with and without the posture shift as a covariate (per inch).
B. Slump-matched comparisons behind Figure 3 (slump_test.py procedure: after minus the 30-day run-up; width and depth changes), 2024-25:
   group effects unadjusted vs adjusted for the posture shift (OLS with group intercepts, HC1)."""

import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
from scipy.stats import norm

rng = np.random.default_rng(7)
# ---- day table: outcomes + posture sums on takes (2024-25)
D = pd.read_parquet("hitter_day_outcomes.parquet")
R = pd.concat(
    [
        pd.read_parquet(f, columns=["game_date", "game_type", "batter", "stand", "description", "sz_top"])
        for f in sorted(glob.glob("raw/2024*.parquet")) + sorted(glob.glob("raw/2025*.parquet"))
    ],
    ignore_index=True,
)
R = R[R.game_type == "R"].copy()
R["description"] = R.description.astype(object)
R = R[R.description.isin(["ball", "called_strike", "blocked_ball"])]
R["game_date"] = pd.to_datetime(R.game_date).dt.strftime("%Y-%m-%d")
R["top"] = 12 * pd.to_numeric(R.sz_top, errors="coerce").astype(float)
PT = (
    R.rename(columns={"stand": "side"})
    .groupby(["batter", "side", "game_date"])
    .agg(TOPS=("top", "sum"), TK=("top", "count"))
    .reset_index()
)
D = D.merge(PT, on=["batter", "side", "game_date"], how="left")
D[["TOPS", "TK"]] = D[["TOPS", "TK"]].fillna(0)
D = D[D.year.astype(str).isin(["2024", "2025"])].copy()
D["t"] = (pd.to_datetime(D.game_date) - pd.Timestamp("2024-01-01")).dt.days
COLS = ["PA", "xw_sum", "K", "sw", "wh", "oz", "oz_sw", "bip", "sweet", "bs_sum", "bs_n", "TOPS", "TK"]
MET = {
    "whiff": ("wh", "sw", 100),
    "chase": ("oz_sw", "oz", 100),
    "sweet": ("sweet", "bip", 100),
    "K": ("K", "PA", 100),
    "xwoba": ("xw_sum", "PA", 1000),
    "bat_speed": ("bs_sum", "bs_n", 1),
    "top": ("TOPS", "TK", 1),
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
W["size"] = W.persist
E = pd.read_csv("events_thr4.csv")
E["lever"] = "depth"
E["size"] = E.persist
A = pd.read_csv("angle_events.csv")
A["lever"] = "angle"
A["size"] = A.persist
wd = pd.concat([W, E])
wd["t0"] = pd.to_datetime(wd.event_date)
A["t0"] = pd.to_datetime(A.event_date)
A = A[~A.apply(lambda r: ((wd.bsy == r.bsy) & ((wd.t0 - r.t0).dt.days.abs() <= 30)).any(), axis=1)]


def windows(events, first_offset, excl_all, need_B):
    """events: frame with batter, side, year, event_date, lever, size (+ dir); returns changers + control reference dates with window rates."""
    ev = events.copy()
    ev["year"] = ev.year.astype(str)
    ev["t"] = (pd.to_datetime(ev.event_date) - pd.Timestamp("2024-01-01")).dt.days
    rows = []
    for (b, s, y), g in D.groupby(["batter", "side", "year"]):
        g = g.sort_values("t")
        tt = g.t.to_numpy()
        cs = np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)])
        win = lambda lo, hi: cs[np.searchsorted(tt, hi, "right")] - cs[np.searchsorted(tt, lo, "left")]
        evs = ev[(ev.batter == b) & (ev.side == s) & (ev.year == str(y))]
        allt = excl_all[(excl_all.batter == b) & (excl_all.side == s) & (excl_all.year == str(y))].t.to_numpy()
        refs = [(int(e.t), e.lever, e["size"], e.get("dir", None), e["name"]) for _, e in evs.iterrows()]
        for r in range(int(tt.min()) + first_offset, int(tt.max()) - 59, 3):
            if len(allt) == 0 or np.abs(allt - r).min() > 90:
                refs.append((r, "control", 0.0, None, None))
        for r, lever, size, dr, nm in refs:
            Bw, Uw, Aw = win(r - 90, r - 31), win(r - 30, r - 1), win(r, r + 59)
            if Uw[0] < 40 or Aw[0] < 60 or (need_B and Bw[0] < 40) or min(Bw[12], Uw[12], Aw[12]) < 20:
                continue
            row = dict(batter=b, side=s, year=str(y), r=r, lever=lever, size=size, dir=dr, name=nm)
            for m, (n, d, sc) in MET.items():
                i, j = COLS.index(n), COLS.index(d)
                for lab, V in (("U", Uw), ("A", Aw), ("B", Bw)):
                    row[f"{m}_{lab}"] = sc * V[i] / V[j] if V[j] > 0 else np.nan
                    row[f"n_{m}_{lab}"] = V[j]
            rows.append(row)
    X = pd.DataFrame(rows)
    X["xw_base"], X["wh_base"] = X.xwoba_B, X.whiff_B
    X["slump_xw"], X["slump_wh"] = X.xwoba_U - X.xwoba_B, X.whiff_U - X.whiff_B
    return X


def match(X, before):
    ch, co = X[X.lever != "control"].copy(), X[X.lever == "control"]
    Z = ["slump_xw", "slump_wh", "xw_base"]
    sdz = co[Z].std()
    for i, e in ch.iterrows():
        pool = co[(co.year == e.year) & ((co.r - e.r).abs() <= 30) & (co.batter != e.batter)].dropna(subset=Z)
        mt = pool.loc[((((pool[Z].astype(float) - e[Z].astype(float)) / sdz) ** 2).sum(1)).nsmallest(20).index]
        for m in MET:
            ch.loc[i, f"y_{m}"] = (e[f"{m}_A"] - e[f"{m}_{before}"]) - (mt[f"{m}_A"] - mt[f"{m}_{before}"]).mean()
            if m in UNIT_VAR:
                ch.loc[i, f"v_{m}"] = (
                    UNIT_VAR[m] * (1 / max(e[f"n_{m}_{before}"], 1) + 1 / max(e[f"n_{m}_A"], 1)) * (1 + 1 / 20)
                )
    return ch


# ---------------- A. Bayesian model (BEFORE = baseline), 2024-25
ALL = pd.concat([W, E, A], ignore_index=True)
ALL["year"] = ALL.year.astype(str)
ALL["t"] = (pd.to_datetime(ALL.event_date) - pd.Timestamp("2024-01-01")).dt.days
XB = windows(ALL, 30, ALL, need_B=True)
# chase-prone exactly as stance_bayes_v2.py: baseline-window chase above the year's 2/3 quantile of hitter-season chase (300+ out-of-zone pitches)
cz = D.groupby(["batter", "side", "year"])[["oz_sw", "oz"]].sum()
cz = cz[cz.oz >= 300]
chase_cut = (cz.oz_sw / cz.oz).groupby(level="year").quantile(2 / 3)
chase_cut.index = chase_cut.index.astype(str)
XB["chase_prone"] = (XB.chase_B / 100 > XB.year.map(chase_cut)).astype(float)
ch = match(XB, "B")


def design(d, posture):
    w, dp, an = (
        (d.lever == "width").astype(float),
        (d.lever == "depth").astype(float),
        (d.lever == "angle").astype(float),
    )
    cols = [w, dp, an, w * d["size"] / 3, w * d["size"] / 3 * d.chase_prone, dp * d["size"] / 3, an * d["size"] / 10]
    if posture:
        cols.append(d.y_top.to_numpy(float))
    return np.column_stack(cols)


NAMES = [
    "after-slump width",
    "after-slump depth",
    "after-slump angle",
    "per 3 in WIDER",
    "x chase-prone (wider)",
    "per 3 in DEEPER",
    "per 10 deg CLOSED",
    "posture: zone top +1 in",
]


def fit(d, m, posture, draws=4000):
    ok = d[f"y_{m}"].notna() & d[f"v_{m}"].notna() & d.y_top.notna()
    d = d[ok]
    y, v, Xd = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float), design(d, posture)
    s = PRIOR_SD[m]
    P0 = np.eye(Xd.shape[1]) * s**2
    grid = np.linspace(0, 3 * s, 301)
    lp = []
    for tau in grid:
        L = np.linalg.cholesky(Xd @ P0 @ Xd.T + np.diag(v + tau**2))
        z = np.linalg.solve(L, y)
        lp.append(-0.5 * z @ z - np.log(np.diag(L)).sum() + norm.logpdf(tau, 0, s))
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
    return betas, len(y)


cell = lambda b, j: f"{b[:, j].mean():+6.2f} [{np.quantile(b[:, j], .05):+6.2f}, {np.quantile(b[:, j], .95):+6.2f}]"
print(
    f"A. BAYESIAN STANCE-CHANGE MODEL, 2024-25 changes with posture measured: {len(ch)} "
    f"({(ch.lever == 'width').sum()} width, {(ch.lever == 'depth').sum()} depth, {(ch.lever == 'angle').sum()} angle); control dates {(XB.lever == 'control').sum():,}"
)
for lev in ("width", "depth", "angle"):
    e = ch[ch.lever == lev].assign(sz=lambda z: z["size"] / (10 if lev == "angle" else 3))
    r = smf.ols("y_top ~ sz", e).fit(cov_type="HC1")
    print(
        f"   posture shift beyond matched controls, {lev} changes: per {'10 deg closed' if lev == 'angle' else '3 in'} {r.params.iloc[1]:+.2f} in (z {r.tvalues.iloc[1]:+.1f})"
    )
out = []
for m in ["sweet", "whiff", "chase", "K", "xwoba", "bat_speed"]:
    b0, n0 = fit(ch, m, False)
    b1, n1 = fit(ch, m, True)
    print(f"\n   {m} (n {n0}): without posture -> with posture shift held fixed")
    for j in range(7):
        print(f"      {NAMES[j]:24s} {cell(b0, j)}  ->  {cell(b1, j)}")
    print(f"      {NAMES[7]:24s} {'':25s}      {cell(b1, 7)}")
    out.append(
        dict(
            outcome=m,
            **{f"{NAMES[j]}|without": b0[:, j].mean() for j in range(7)},
            **{f"{NAMES[j]}|with": b1[:, j].mean() for j in range(8)},
        )
    )
pd.DataFrame(out).to_csv("stance_posture_bayes.csv", index=False)
# ---------------- B. Figure 3 comparisons (after minus run-up), width and depth changes, 2024-25
WD = pd.concat([W.assign(lever="width"), E.assign(lever="depth")], ignore_index=True)
WD["year"] = WD.year.astype(str)
WD["t"] = (pd.to_datetime(WD.event_date) - pd.Timestamp("2024-01-01")).dt.days
XU = windows(WD, 90, WD, need_B=True)
cu = match(XU, "U")
cu["kind"] = cu.lever + ":" + cu["dir"].astype(str)
print(f"\nB. SLUMP-MATCHED COMPARISONS (Figure 3 design, after minus run-up), 2024-25: {len(cu)} changes")
for k in ["width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"]:
    print(f"   posture shift {k:15s} {cu.loc[cu.kind == k, 'y_top'].mean():+.2f} in (n {(cu.kind == k).sum()})")
print(
    f"   {'':18s} "
    + "  ".join(f"{k.split(':')[1]:>24s}" for k in ["width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"])
    + "   posture coef (per in)"
)
for m in ["xwoba", "whiff", "chase", "sweet", "K"]:
    d = cu.dropna(subset=[f"y_{m}", "y_top"])
    r0 = smf.ols(f"y_{m} ~ 0 + C(kind)", d).fit(cov_type="HC1")
    r1 = smf.ols(f"y_{m} ~ 0 + C(kind) + y_top", d).fit(cov_type="HC1")
    cells = [
        f"{r0.params[f'C(kind)[{k}]']:+6.2f} -> {r1.params[f'C(kind)[{k}]']:+6.2f}"
        for k in ["width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"]
    ]
    print(
        f"   {m:18s} "
        + "  ".join(f"{c:>24s}" for c in cells)
        + f"   {r1.params['y_top']:+.2f} (z {r1.tvalues['y_top']:+.1f})"
    )
cu.to_csv("stance_posture_events.csv", index=False)
