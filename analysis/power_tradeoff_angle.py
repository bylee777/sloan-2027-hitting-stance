"""Contact vs home runs for STANCE ANGLE (open/closed) and the front foot — companion to power_tradeoff.py.
A. Within-hitter monthly panel, same design as power_tradeoff.py A, with all four stance measurements entered jointly:
   width, depth, plate distance (per 3 in) and stance angle (per 10 deg more CLOSED; the data code open as negative).
A2. Front-foot check from the monthly foot-geometry file (vizData months): adds front-foot turn at setup (per 10 deg more open toe).
B. Slump-matched stance-angle changes (OPENED / CLOSED; angle changes within 30 days of a width or depth change dropped, as in the
   Bayesian model), same procedure as slump_test.py; controls exclude dates within 90 days of any width, depth or angle change.
"""

import numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from fe import demean

D = pd.read_parquet("hitter_day_outcomes.parquet").merge(
    pd.read_parquet("hitter_day_power.parquet"), on=["batter", "side", "game_date"], how="left"
)
PWR = ["HR", "XHR", "BRL", "TRK", "HARD", "EV_SUM", "TBX", "AB", "PA_p"]
D[PWR] = D[PWR].fillna(0)
MET = {
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
D["month"] = D.game_date.str[:7]
NEED = sorted({c for v in MET.values() for c in v[:2]} | {v[3] for v in MET.values()})
M = D.groupby(["batter", "side", "month"])[NEED].sum().reset_index()


def rates(X):
    for m, (n, d, sc, w, lab) in MET.items():
        X[m] = np.where(X[d] > 0, sc * X[n] / X[d].where(X[d] > 0), np.nan)
    return X


def fe_fit(d, y, xs, w):
    codes = [pd.factorize(d.bs)[0], pd.factorize(d.month)[0]]
    Mx = demean(
        np.column_stack([d[y].to_numpy(float)] + [d[x].to_numpy(float) for x in xs]), codes, iters=300, tol=1e-10
    )
    r = sm.WLS(Mx[:, 0], Mx[:, 1:], weights=d[w].to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.bs)[0]}
    )
    return r.params, r.bse


def panel(X, xs, heads, title):
    print(title)
    print(f"{'outcome':32s} " + " ".join(f"{h:>17s}" for h in heads))
    out = []
    for m, (n, d, sc, w, lab) in MET.items():
        dd = X.dropna(subset=[m])
        dd = dd[dd[w] > 0]
        b, s = fe_fit(dd, m, xs, w)
        print(f"{lab:32s} " + " ".join(f"{b[i]:+7.2f} (z {b[i]/s[i]:+5.1f})" for i in range(len(xs))))
        out.append(
            dict(
                outcome=m,
                **{f"{x}_b": b[i] for i, x in enumerate(xs)},
                **{f"{x}_z": b[i] / s[i] for i, x in enumerate(xs)},
            )
        )
    return pd.DataFrame(out)


# ---------------- A. four stance measurements jointly (game-level stance, monthly means)
S = pd.read_parquet("stance_adj.parquet")
S["month"] = S.game_date.astype(str).str[:7]
SM = (
    S.groupby(["batter", "side", "month"])
    .agg(
        width=("width", "mean"),
        depth=("depth_adj", "mean"),
        off=("off_plate_adj", "mean"),
        angle=("angle", "mean"),
        games=("depth_adj", "size"),
    )
    .reset_index()
)
X = rates(SM.merge(M, on=["batter", "side", "month"]))
X = X[(X.games >= 8) & (X.PA >= 50)].dropna(subset=["width", "depth", "off", "angle"]).copy()
X["bs"] = X.batter.astype(str) + X.side
for c in ("width", "depth", "off"):
    X[c + "3"] = X[c] / 3
X["closed10"] = X.angle / 10
A = panel(
    X,
    ["width3", "depth3", "off3", "closed10"],
    ["3 in WIDER", "3 in DEEPER", "3 in FARTHER OFF", "10 deg CLOSED"],
    f"A. WITHIN-HITTER MONTHLY PANEL, all four stance measurements jointly: {len(X):,} hitter-months, {X.bs.nunique()} hitters",
)
A.to_csv("power_tradeoff_angle_monthly.csv", index=False)
# ---------------- A2. front foot (monthly foot geometry)
G = pd.read_parquet("front_foot_monthly_geometry.parquet").dropna(
    subset=["width_in", "depth_in", "off_plate_in", "stance_angle", "front_turn0"]
)
XG = rates(G.merge(M, on=["batter", "side", "month"]))
XG = XG[XG.PA >= 50].copy()
XG["bs"] = XG.batter.astype(str) + XG.side
for c in ("width_in", "depth_in", "off_plate_in"):
    XG[c + "3"] = XG[c] / 3
XG["closed10"] = XG.stance_angle / 10
XG["toe_open10"] = XG.front_turn0 / 10
A2 = panel(
    XG,
    ["width_in3", "depth_in3", "off_plate_in3", "closed10", "toe_open10"],
    ["3 in WIDER", "3 in DEEPER", "3 in FARTHER OFF", "10 deg CLOSED", "front toe +10 open"],
    f"\nA2. FRONT-FOOT CHECK (monthly foot geometry, not park-adjusted): {len(XG):,} hitter-months, {XG.bs.nunique()} hitters",
)
A2.to_csv("power_tradeoff_frontfoot_monthly.csv", index=False)
# ---------------- B. slump-matched stance-angle changes
D["t"] = (pd.to_datetime(D.game_date) - pd.Timestamp("2024-01-01")).dt.days
COLS = sorted({c for v in MET.values() for c in v[:2]} | {"xw_sum", "PA", "wh", "sw"})
Wv = pd.read_csv("width_events.csv")
Wv["lever"] = "width"
Ev = pd.read_csv("events_thr4.csv")
Ev["lever"] = "depth"
Av = pd.read_csv("angle_events.csv")
Av["lever"] = "angle"
wd = pd.concat([Wv, Ev])
wd["t0"] = pd.to_datetime(wd.event_date)
Av["t0"] = pd.to_datetime(Av.event_date)
ov = Av.apply(lambda r: ((wd.bsy == r.bsy) & ((wd.t0 - r.t0).dt.days.abs() <= 30)).any(), axis=1)
ALL = pd.concat([Wv, Ev, Av], ignore_index=True)
ALL["year"] = ALL.year.astype(str)
ALL["t"] = (pd.to_datetime(ALL.event_date) - pd.Timestamp("2024-01-01")).dt.days
ANG = Av[~ov].copy()
ANG["year"] = ANG.year.astype(str)
ANG["t"] = (pd.to_datetime(ANG.event_date) - pd.Timestamp("2024-01-01")).dt.days
rows = []
iPA = COLS.index("PA")
for (b, s, y), g in D.groupby(["batter", "side", "year"]):
    g = g.sort_values("t")
    tt = g.t.to_numpy()
    cs = np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)])

    def win(lo, hi):
        i, j = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "right")
        return cs[j] - cs[i]

    allt = ALL[(ALL.batter == b) & (ALL.side == s) & (ALL.year == y)].t.to_numpy()
    evs = ANG[(ANG.batter == b) & (ANG.side == s) & (ANG.year == y)]
    refs = [(int(e.t), f"angle:{e.dir}", e["name"]) for _, e in evs.iterrows()]
    for r in range(int(tt.min()) + 90, int(tt.max()) - 59, 3):
        if len(allt) == 0 or np.abs(allt - r).min() > 90:
            refs.append((r, "control", None))
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
kinds = [("angle:CLOSED", "Closed"), ("angle:OPENED", "Opened")]
print(
    f"\nB. SLUMP-MATCHED STANCE-ANGLE CHANGES: effect beyond bounce-back ({len(R)} of {len(ANG)} angle changes with full windows)"
)
print(f"{'outcome':32s} " + " ".join(f"{lab + ' n' + str((R.kind == k).sum()):>17s}" for k, lab in kinds))
for m, (n, d, sc, w, lab) in MET.items():
    cells = []
    for k, _ in kinds:
        e = R.loc[R.kind == k, f"eff_{m}"].dropna()
        se = e.std() / np.sqrt(len(e))
        cells.append(f"{e.mean():+7.2f} (z {e.mean()/se:+5.1f})")
    print(f"{lab:32s} " + " ".join(cells))
R.to_csv("power_tradeoff_angle_events.csv", index=False)
