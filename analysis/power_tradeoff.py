"""Are we trading contact for home runs? Contact and power outcomes for each stance lever, in the paper's two designs.
A. Within-hitter monthly panel (hitter x side FE + year-month FE, clustered by hitter): outcome ~ width + depth + plate distance
   (park-adjusted game-level stance, monthly means), each per 3 in, holding the other two fixed.
B. Slump-matched stance-change events (slump_test.py procedure exactly: after 0-59 d minus run-up 30 d, 20 nearest slump-alike
   non-changers) with power outcomes added.
Power: HR, expected HR (EV x LA model) and barrels per 600 PA; barrel % and hard-hit % of tracked balls in play; EV; ISO.
Contact: whiff % per swing, strikeout % per PA. Net: xwOBA per PA (1 point over 600 PA ~ 0.5 runs)."""

import numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from fe import demean

D = pd.read_parquet("hitter_day_outcomes.parquet").merge(
    pd.read_parquet("hitter_day_power.parquet"), on=["batter", "side", "game_date"], how="left"
)
PWR = ["HR", "XHR", "BRL", "TRK", "HARD", "EV_SUM", "TBX", "AB", "PA_p"]
D[PWR] = D[PWR].fillna(0)
MET = {  # name: (numerator, denominator, scale, weight column in the monthly panel, label)
    "whiff": ("wh", "sw", 100, "sw", "whiff % per swing"),
    "K": ("K", "PA", 100, "PA", "strikeout % per PA"),
    "hr600": ("HR", "PA_p", 600, "PA_p", "HR per 600 PA"),
    "xhr600": ("XHR", "PA_p", 600, "PA_p", "expected HR per 600 PA"),
    "brl600": ("BRL", "PA_p", 600, "PA_p", "barrels per 600 PA"),
    "brl": ("BRL", "TRK", 100, "TRK", "barrel % of balls in play"),
    "hard": ("HARD", "TRK", 100, "TRK", "hard-hit % (95+ mph)"),
    "ev": ("EV_SUM", "TRK", 1, "TRK", "exit velocity (mph)"),
    "iso": ("TBX", "AB", 1000, "AB", "ISO (points)"),
    "sweet": ("sweet", "bip", 100, "bip", "sweet-spot % of balls in play"),
    "xwoba": ("xw_sum", "PA", 1000, "PA", "xwOBA per PA (points)"),
}
# ---------------- A. monthly panel
S = pd.read_parquet("stance_adj.parquet")
S["month"] = S.game_date.astype(str).str[:7]
# width_adj is empty in stance_adj.parquet -> raw game-level width (as in the paper's width panels); depth and plate distance park-adjusted
SM = (
    S.groupby(["batter", "side", "month"])
    .agg(
        width=("width", "mean"), depth=("depth_adj", "mean"), off=("off_plate_adj", "mean"), games=("depth_adj", "size")
    )
    .reset_index()
)
D["month"] = D.game_date.str[:7]
NEED = sorted({c for v in MET.values() for c in v[:2]} | {v[3] for v in MET.values()})
M = D.groupby(["batter", "side", "month"])[NEED].sum().reset_index()
X = SM.merge(M, on=["batter", "side", "month"])
X = X[(X.games >= 8) & (X.PA >= 50)].dropna(subset=["width", "depth", "off"]).copy()
X["bs"] = X.batter.astype(str) + X.side
for c in ("width", "depth", "off"):
    X[c + "3"] = X[c] / 3
for m, (n, d, sc, w, lab) in MET.items():
    X[m] = np.where(X[d] > 0, sc * X[n] / X[d].where(X[d] > 0), np.nan)


def fe_fit(d, y, xs, w):
    codes = [pd.factorize(d.bs)[0], pd.factorize(d.month)[0]]
    Mx = demean(
        np.column_stack([d[y].to_numpy(float)] + [d[x].to_numpy(float) for x in xs]), codes, iters=300, tol=1e-10
    )
    r = sm.WLS(Mx[:, 0], Mx[:, 1:], weights=d[w].to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.bs)[0]}
    )
    return r.params, r.bse


print(
    f"A. WITHIN-HITTER MONTHLY PANEL: {len(X):,} hitter-months, {X.bs.nunique()} hitters (50+ PA, 8+ stance games); "
    f"each lever per 3 in, the other two held fixed"
)
print(f"{'outcome':32s} {'3 in WIDER':>17s} {'3 in DEEPER':>17s} {'3 in FARTHER OFF':>17s}")
A = []
for m, (n, d, sc, w, lab) in MET.items():
    dd = X.dropna(subset=[m])
    dd = dd[dd[w] > 0]
    b, s = fe_fit(dd, m, ["width3", "depth3", "off3"], w)
    print(f"{lab:32s} " + " ".join(f"{b[i]:+7.2f} (z {b[i]/s[i]:+5.1f})" for i in range(3)))
    A.append(
        dict(
            outcome=m,
            **{f"{k}_b": b[i] for i, k in enumerate(("wider", "deeper", "off"))},
            **{f"{k}_z": b[i] / s[i] for i, k in enumerate(("wider", "deeper", "off"))},
        )
    )
pd.DataFrame(A).to_csv("power_tradeoff_monthly.csv", index=False)
# ---------------- B. slump-matched events (slump_test.py procedure)
D["t"] = (pd.to_datetime(D.game_date) - pd.Timestamp("2024-01-01")).dt.days
COLS = sorted({c for v in MET.values() for c in v[:2]} | {"xw_sum", "PA", "wh", "sw"})
Wv = pd.read_csv("width_events.csv")
Wv["lever"] = "width"
Ev = pd.read_csv("events_thr4.csv")
Ev["lever"] = "depth"
EV = pd.concat([Wv, Ev], ignore_index=True)
EV["year"] = EV.year.astype(str)
EV["t"] = (pd.to_datetime(EV.event_date) - pd.Timestamp("2024-01-01")).dt.days
rows = []
for (b, s, y), g in D.groupby(["batter", "side", "year"]):
    g = g.sort_values("t")
    tt = g.t.to_numpy()
    cs = np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)])

    def win(lo, hi):
        i, j = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "right")
        return cs[j] - cs[i]

    evs = EV[(EV.batter == b) & (EV.side == s) & (EV.year == y)]
    refs = [(int(e.t), f"{e.lever}:{e.dir}", e["name"]) for _, e in evs.iterrows()]
    evt = evs.t.to_numpy()
    for r in range(int(tt.min()) + 90, int(tt.max()) - 59, 3):
        if len(evt) == 0 or np.abs(evt - r).min() > 90:
            refs.append((r, "control", None))
    iPA = COLS.index("PA")
    for r, kind, nm in refs:
        B, U, A_ = win(r - 90, r - 31), win(r - 30, r - 1), win(r, r + 59)
        if min(B[iPA], U[iPA]) < 40 or A_[iPA] < 60:
            continue
        row = dict(batter=b, side=s, year=y, r=r, kind=kind, name=nm)
        for m, (n, d, sc, w, lab) in MET.items():
            i, j = COLS.index(n), COLS.index(d)
            for labw, v in (("B", B), ("U", U), ("A", A_)):
                row[f"{m}_{labw}"] = sc * v[i] / v[j] if v[j] > 0 else np.nan
        rows.append(row)
Xe = pd.DataFrame(rows)
Xe["slump_xw"] = Xe.xwoba_U - Xe.xwoba_B
Xe["slump_wh"] = Xe.whiff_U - Xe.whiff_B
ch, co = Xe[Xe.kind != "control"], Xe[Xe.kind == "control"]
Z = ["slump_xw", "slump_wh", "xwoba_B"]
sd = co[Z].std()
res = []
for _, e in ch.iterrows():
    pool = co[(co.year == e.year) & ((co.r - e.r).abs() <= 30) & (co.batter != e.batter)].dropna(subset=Z)
    dist = (((pool[Z].astype(float) - e[Z].astype(float)) / sd) ** 2).sum(1).astype(float)
    mt = pool.loc[dist.nsmallest(20).index]
    row = dict(kind=e.kind, name=e["name"])
    for k in MET:
        row[f"eff_{k}"] = (e[f"{k}_A"] - e[f"{k}_U"]) - (mt[f"{k}_A"] - mt[f"{k}_U"]).mean()
    res.append(row)
R = pd.DataFrame(res)
print(
    f"\nB. SLUMP-MATCHED STANCE CHANGES: effect beyond bounce-back (60 days after vs the 30 days before), {len(R)} changes"
)
kinds = [("width:WIDER", "Wider"), ("width:NARROWER", "Narrower"), ("depth:BACK", "Deeper"), ("depth:UP", "Shallower")]
print(f"{'outcome':32s} " + " ".join(f"{lab + ' n' + str((R.kind == k).sum()):>17s}" for k, lab in kinds))
for m, (n, d, sc, w, lab) in MET.items():
    cells = []
    for k, _ in kinds:
        e = R.loc[R.kind == k, f"eff_{m}"].dropna()
        se = e.std() / np.sqrt(len(e))
        cells.append(f"{e.mean():+7.2f} (z {e.mean()/se:+5.1f})")
    print(f"{lab:32s} " + " ".join(cells))
R.to_csv("power_tradeoff_events.csv", index=False)
