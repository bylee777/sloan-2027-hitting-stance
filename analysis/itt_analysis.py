"""ITT re-analysis of the stance-change designs (Sloan paper, 2026-09-28). Reads itt_events.csv (itt_events.py).

Designs re-run (each first reproduces the original script's numbers exactly):
  SLUMP  = slump_test.py: effect = changer's (after [0,59] - run-up [-30,-1]) minus 20 matched slump-alike controls'.
  BAYES  = stance_bayes_v2.py (BEFORE=B): y = changer's (after - baseline [-90,-31]) minus matched controls'; random-
           effects meta-regression, exact tau-grid posterior (Cholesky likelihood instead of scipy logpdf).
Event / control sets:
  (a) original events (persisted only)      + original controls (no original event within +-90 days, past OR future)
  (b) ITT events (persisted + reverted)     + original controls
  (c) ITT events                            + ITT controls: no ITT event START in [r-90, r+3] (future changes allowed;
                                              dates within +-3 days of the hitter's own change excluded)
Bayesian ITT dose = 'move' (15-game change actually attempted); original dose = 'persist' (30-game median change),
which is itself post-treatment, so a bridge run (original events, dose = move) separates the dose from the event set.
Validation (train 2024-25 -> test 2026) with the intercept-only and no-effect models REFIT as their own models
(own tau posterior), next to the original zeroed-coefficient version."""

import numpy as np, pandas as pd, warnings
from scipy.stats import norm
from scipy.linalg import solve_triangular

warnings.filterwarnings("ignore")

LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


T0 = pd.Timestamp("2024-01-01")
D = pd.read_parquet("hitter_day_outcomes.parquet")
D["t"] = (pd.to_datetime(D.game_date) - T0).dt.days
D["year"] = D.year.astype(str)
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
chase_cut = (
    D.groupby(["batter", "side", "year"])[["oz_sw", "oz"]]
    .sum()
    .pipe(lambda z: z[z.oz >= 300])
    .pipe(lambda z: z.oz_sw / z.oz)
    .groupby(level="year")
    .quantile(2 / 3)
)


# ------------------------------------------------------------------ events
def prep(ev, lever=None, dose="persist"):
    ev = ev.copy()
    if lever:
        ev["lever"] = lever
    ev["year"] = ev.year.astype(str)
    ev["t"] = (pd.to_datetime(ev.event_date) - T0).dt.days
    ev["size"] = ev[dose]
    return ev


OW, OE, OA = pd.read_csv("width_events.csv"), pd.read_csv("events_thr4.csv"), pd.read_csv("angle_events.csv")
ITT = pd.read_csv("itt_events.csv")
IW, IE, IA = (ITT[ITT.lever == l].drop(columns="lever") for l in ("width", "depth", "angle"))


def v2_events(W, E, A, dose):
    """stance_bayes_v2 event table: width, depth, angle (angle dropped if within 30 days of a width/depth change)."""
    W, E, A = prep(W, "width", dose), prep(E, "depth", dose), prep(A, "angle", dose)
    wd = pd.concat([W, E])
    wd["t0"] = pd.to_datetime(wd.event_date)
    A = A.copy()
    A["t0"] = pd.to_datetime(A.event_date)
    ov = A.apply(lambda r: ((wd.bsy == r.bsy) & ((wd.t0 - r.t0).dt.days.abs() <= 30)).any(), axis=1)
    EV = pd.concat([W, E, A[~ov].drop(columns=["t0"])], ignore_index=True)
    if EV["size"].isna().any():  # reverted angle changes whose 30-game median hits a missing-angle game
        say(
            f"  ({dose} dose undefined for {EV['size'].isna().sum()} {sorted(EV[EV['size'].isna()].lever.unique())} events -> dropped in this run)"
        )
        EV = EV[EV["size"].notna()].reset_index(drop=True)
    return EV, int(ov.sum()), len(A)


# ------------------------------------------------------------------ windows for every (hitter-season, date) needed
G = {}
for key, g in D.groupby(["batter", "side", "year"]):
    g = g.sort_values("t")
    tt = g.t.to_numpy()
    G[key] = (tt, np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)]))
ALL_EV = pd.concat(
    [prep(x, l) for x, l in ((OW, "width"), (OE, "depth"), (OA, "angle"), (IW, "width"), (IE, "depth"), (IA, "angle"))]
)
ev_days = ALL_EV.groupby(["batter", "side", "year"]).t.apply(lambda z: sorted(set(z))).to_dict()


def rates(V, lab):
    out = {}
    for m, (n, d, sc) in MET.items():
        i, j = COLS.index(n), COLS.index(d)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{m}_{lab}"] = np.where(V[:, j] > 0, sc * V[:, i] / np.where(V[:, j] > 0, V[:, j], 1), np.nan)
        out[f"n_{m}_{lab}"] = V[:, j]
    return out


rows = []
for key, (tt, cs) in G.items():
    grid = np.arange(int(tt.min()) + 30, int(tt.max()) - 59, 3)
    r = np.unique(np.concatenate([grid, np.array(ev_days.get(key, []), dtype=int)])).astype(int)
    if len(r) == 0:
        continue
    win = lambda lo, hi: cs[np.searchsorted(tt, hi, "right")] - cs[np.searchsorted(tt, lo, "left")]
    B, U, A = win(r - 90, r - 31), win(r - 30, r - 1), win(r, r + 59)
    i0 = np.searchsorted(tt, r, "left")
    E15 = cs[np.minimum(i0 + 15, len(tt))] - cs[i0]  # first 15 games from r
    d = dict(
        batter=key[0],
        side=key[1],
        year=key[2],
        r=r,
        tmin=int(tt.min()),
        tmax=int(tt.max()),
        PA_B=B[:, 0],
        PA_U=U[:, 0],
        PA_A=A[:, 0],
        PA_E=E15[:, 0],
    )
    for V, lab in ((B, "B"), (U, "U"), (A, "A"), (E15, "E")):
        d.update(rates(V, lab))
    rows.append(pd.DataFrame(d))
WT = pd.concat(rows, ignore_index=True)
WT["full"] = (WT.PA_B >= 40) & (WT.PA_U >= 40) & (WT.PA_A >= 60)
off = WT.r - (WT.tmin + 30)
WT["grid_v2"] = (off >= 0) & (off % 3 == 0) & (WT.r < WT.tmax - 59)
WT["grid_slump"] = WT.grid_v2 & (WT.r >= WT.tmin + 90)
KEY = ["batter", "side", "year", "r"]


def controls(grid_col, ev, rule):
    """rule 'orig': no event of `ev` within +-90 days; rule 'itt': no event start in [r-90, r+3]."""
    days = ev.groupby(["batter", "side", "year"]).t.apply(np.array).to_dict()
    c = WT[WT[grid_col] & WT.full].copy()
    keep = np.ones(len(c), bool)
    for idx, (b, s, y, r) in enumerate(zip(c.batter, c.side, c.year, c.r)):
        e = days.get((b, s, y))
        if e is None or len(e) == 0:
            continue
        keep[idx] = (np.abs(e - r).min() > 90) if rule == "orig" else not ((e >= r - 90) & (e <= r + 3)).any()
    return c[keep]


def attach(ev, cols):
    x = ev[cols].rename(columns={"t": "r"}).merge(WT, on=KEY, how="left")
    return x[x.full.fillna(False).astype(bool)]


# ------------------------------------------------------------------ 1. mechanism: early outcomes of reverters vs persisters
say("=" * 110)
say("1. ITT EVENTS and the MECHANISM (do abandoned changes start badly?)")
say("=" * 110)
say(
    ITT.groupby(["lever", "dir", "persisted"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={False: "reverted", True: "persisted"})
    .to_string()
)
M = prep(ITT).rename(columns={"t": "r"}).merge(WT, on=KEY, how="left")
M = M[(M.PA_U >= 40) & (M.PA_E >= 40)]
say(
    f"\nfirst 15 games after the change minus the 30-day run-up (events with >= 40 PA in both): {len(M)} "
    f"({(~M.persisted).sum()} reverted, {M.persisted.sum()} persisted)"
)
say(f"{'metric':10s} {'group':12s} {'reverted':>9s} {'persisted':>10s} | reverted - persisted (SE, z)")
for grp, sel in (
    ("all levers", M.lever.notna()),
    ("width+depth", M.lever.isin(["width", "depth"])),
    ("angle", M.lever == "angle"),
):
    mm = M[sel]
    for m in ("xwoba", "whiff", "chase", "sweet", "K"):
        dd = mm[f"{m}_E"] - mm[f"{m}_U"]
        a, b = dd[~mm.persisted].dropna(), dd[mm.persisted].dropna()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        say(
            f"{m:10s} {grp:12s} {a.mean():+9.2f} {b.mean():+10.2f} | {a.mean() - b.mean():+7.2f} (SE {se:.2f}, z {(a.mean() - b.mean()) / se:+.1f})  n {len(a)}/{len(b)}"
        )
    s_r = (mm.xwoba_U - mm.xwoba_B)[~mm.persisted & (mm.PA_B >= 40)]
    s_p = (mm.xwoba_U - mm.xwoba_B)[mm.persisted & (mm.PA_B >= 40)]
    say(
        f"{'':10s} {grp:12s} run-up slump xwOBA (U - B): reverted {s_r.mean():+.1f} (n {len(s_r)}) vs persisted {s_p.mean():+.1f} (n {len(s_p)})"
    )


# ------------------------------------------------------------------ 2. slump-matched comparison (slump_test.py)
def slump(ev, co, label):
    ch = attach(
        ev, ["batter", "side", "year", "t", "lever", "dir", "name"] + (["persisted"] if "persisted" in ev else [])
    )
    ch["kind"] = ch.lever + ":" + ch.dir
    for X in (ch, co):
        X["slump_xw"] = X.xwoba_U - X.xwoba_B
        X["slump_wh"] = X.whiff_U - X.whiff_B
    Z = ["slump_xw", "slump_wh", "xwoba_B"]
    sd = co[Z].std()
    res = []
    for _, e in ch.iterrows():
        pool = co[(co.year == e.year) & ((co.r - e.r).abs() <= 30) & (co.batter != e.batter)].dropna(subset=Z)
        dist = (((pool[Z].astype(float) - e[Z].astype(float)) / sd) ** 2).sum(1).astype(float)
        mt = pool.loc[dist.nsmallest(20).index]
        row = dict(kind=e.kind, persisted=e.get("persisted", True))
        for k in MET:
            row[f"naive_{k}"] = e[f"{k}_A"] - e[f"{k}_U"]
            row[f"ctrl_{k}"] = (mt[f"{k}_A"] - mt[f"{k}_U"]).mean()
        res.append(row)
    R = pd.DataFrame(res)
    say(f"\n--- {label}: changers with full windows {len(ch)} of {len(ev)}; control dates {len(co):,}")
    for kind in ("width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"):
        r = R[R.kind == kind]
        cells = []
        for m in ("xwoba", "whiff", "chase", "sweet", "K", "bat_speed"):
            eff = r[f"naive_{m}"] - r[f"ctrl_{m}"]
            se = eff.std() / np.sqrt(eff.notna().sum())
            cells.append(f"{m} {eff.mean():+.2f} ({se:.2f}, z {eff.mean() / se:+.1f})")
        say(f"  {kind:15s} n {len(r):3d} (reverted {int((~r.persisted.astype(bool)).sum())}) | " + " | ".join(cells))
    return R


say("\n" + "=" * 110)
say(
    "2. SLUMP-MATCHED 'BEYOND BOUNCE-BACK' (slump_test.py: after [0,59] minus run-up [-30,-1], minus 20 matched controls)"
)
say(
    "   cells: EFFECT (SE, z). Target for (a): WIDER xwoba +16.12 whiff -1.66 | NARROWER xwoba +16.50 | BACK chase -2.44 xwoba +29.75 | UP xwoba +7.26"
)
say("=" * 110)
oev = pd.concat([prep(OW, "width"), prep(OE, "depth")], ignore_index=True)
iev = pd.concat([prep(IW, "width"), prep(IE, "depth")], ignore_index=True)
co_orig = controls("grid_slump", oev, "orig")
co_itt = controls("grid_slump", iev, "itt")
Ra = slump(oev, co_orig.copy(), "(a) original events + original controls")
Rb = slump(iev, co_orig.copy(), "(b) ITT events + original controls")
Rc = slump(iev, co_itt.copy(), "(c) ITT events + ITT controls")
say("\n  reverted changes only, (c) controls:")
Rr = Rc[~Rc.persisted.astype(bool)]
for kind in ("width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"):
    r = Rr[Rr.kind == kind]
    if len(r) == 0:
        say(f"  {kind:15s} n 0")
        continue
    eff = r.naive_xwoba - r.ctrl_xwoba
    say(f"  {kind:15s} n {len(r):3d} | xwoba {eff.mean():+.2f} (SE {eff.std() / np.sqrt(len(r)):.2f})")
r = Rr
eff = r.naive_xwoba - r.ctrl_xwoba
say(
    f"  ALL reverted width/depth n {len(r)} | xwoba {eff.mean():+.2f} (SE {eff.std() / np.sqrt(len(r)):.2f}) vs persisted in (c): "
    f"{(Rc[Rc.persisted.astype(bool)].naive_xwoba - Rc[Rc.persisted.astype(bool)].ctrl_xwoba).mean():+.2f}"
)


# ------------------------------------------------------------------ 3. Bayesian hierarchical model (stance_bayes_v2.py)
def build_bayes(ev, co):
    ch = attach(
        ev, ["batter", "side", "year", "t", "lever", "size", "name"] + (["persisted"] if "persisted" in ev else [])
    )
    ch = ch.copy()
    co = co.copy()
    for X in (ch, co):
        X["xw_base"], X["wh_base"], X["chase_base"] = X.xwoba_B, X.whiff_B, X.chase_B
        X["slump_xw"] = X.xwoba_U - X.xw_base
        X["slump_wh"] = X.whiff_U - X.wh_base
        X["chase_prone"] = (X.chase_base / 100 > X.year.map(chase_cut)).astype(float)
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
            ch.loc[i, f"y_{m}"] = (e[f"{m}_A"] - e[f"{m}_B"]) - (mt[f"{m}_A"] - mt[f"{m}_B"]).mean()
            ch.loc[i, f"v_{m}"] = UNIT_VAR[m] * (1 / max(e[f"n_{m}_B"], 1) + 1 / max(e[f"n_{m}_A"], 1)) * (1 + 1 / 20)
    return ch


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


def fit(d, m, rng, cols=None, draws=4000):
    ok = d[f"y_{m}"].notna() & d[f"v_{m}"].notna()
    d = d[ok]
    y, v = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float)
    Xd = design(d, cols)
    k = Xd.shape[1]
    s = PRIOR_SD[m]
    grid = np.linspace(0, 3 * s, 301)
    lp = []
    base = Xd @ (np.eye(k) * s**2) @ Xd.T if k else np.zeros((len(y), len(y)))
    for tau in grid:
        L = np.linalg.cholesky(base + np.diag(v + tau**2))
        z = solve_triangular(L, y, lower=True)
        lp.append(-0.5 * (z @ z) - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * np.pi) + norm.logpdf(tau, 0, s))
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
    return dict(beta=betas, tau=taus, n=len(y), cols=cols)


def lpd(post, d, m, rng, zero_keep=None):
    """zero_keep=None: use the posterior as fitted; else zero every coefficient not in zero_keep (v2's shortcut)."""
    ok = d[f"y_{m}"].notna()
    d = d[ok]
    y, v = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float)
    Xd = design(d, post["cols"])
    b = post["beta"]
    if zero_keep is not None:
        b = b * np.isin(np.arange(b.shape[1]), zero_keep)[None, :]
    mu = b @ Xd.T if Xd.shape[1] else np.zeros((len(post["tau"]), len(y)))
    sd = np.sqrt(v[None, :] + post["tau"][:, None] ** 2)
    dens = norm.pdf(y[None, :], mu, sd).mean(0)
    lo, hi = np.quantile(mu + rng.standard_normal(mu.shape) * sd, [0.1, 0.9], axis=0)
    cor = np.corrcoef(mu.mean(0), y)[0, 1] if mu.std() > 0 else np.nan
    return np.log(dens).sum(), ((y >= lo) & (y <= hi)).mean(), cor, len(y)


def bayes(ch, label, reproduce=False):
    rng = np.random.default_rng(7)
    tr, te = ch[ch.year != "2026"], ch[ch.year == "2026"]
    say(
        f"\n--- {label}: {len(ch)} changes ({(ch.lever == 'width').sum()} width, {(ch.lever == 'depth').sum()} depth, "
        f"{(ch.lever == 'angle').sum()} angle; reverted {int((~ch.persisted.astype(bool)).sum()) if 'persisted' in ch else 0}; "
        f"2024-25 {len(tr)}, 2026 {len(te)})"
    )
    say(f"  VALIDATION 2024-25 -> 2026 (total log predictive density over the 2026 changes; higher = better)")
    say(
        f"  {'outcome':10s} | v2-style (coefficients zeroed): full  int-only  none  full-none | REFIT: int-only  none | "
        f"full-none  full-int  int-none | 80% cov  corr  n"
    )
    V = {}
    for m in MET:
        post = fit(tr, m, rng)
        full, cov, cor, n = lpd(post, te, m, rng)
        icz, _, _, _ = lpd(post, te, m, rng, zero_keep=[0, 1, 2])
        nez, _, _, _ = lpd(post, te, m, rng, zero_keep=[])
        V[m] = (full, icz, nez, cov, cor, n)
    rng2 = np.random.default_rng(11)
    for m in MET:
        full, icz, nez, cov, cor, n = V[m]
        ic = lpd(fit(tr, m, rng2, cols=[0, 1, 2]), te, m, rng2)[0]
        ne = lpd(fit(tr, m, rng2, cols=[]), te, m, rng2)[0]
        say(
            f"  {m:10s} | {full:8.1f} {icz:8.1f} {nez:6.1f} {full - nez:+7.2f} | {ic:8.1f} {ne:6.1f} | "
            f"{full - ne:+7.2f} {full - ic:+7.2f} {ic - ne:+7.2f} | {cov:5.0%} {cor:+.2f} {n}"
        )
    say(f"  FULL POSTERIOR (all seasons): mean [90% interval]")
    P = {}
    for m in MET:
        post = fit(ch, m, rng)
        b = post["beta"]
        lo, hi = np.quantile(b, [0.05, 0.95], axis=0)
        P[m] = post
        say(
            f"  {m:10s} tau {post['tau'].mean():5.2f} | "
            + " | ".join(f"{NAMES[j]} {b[:, j].mean():+.2f} [{lo[j]:+.2f},{hi[j]:+.2f}]" for j in range(7))
        )
    return P


say("\n" + "=" * 110)
say(
    "3. BAYESIAN HIERARCHICAL MODEL (stance_bayes_v2.py, BEFORE=B: after [0,59] minus baseline [-90,-31], minus matched)"
)
say(
    "   target for the original run: 150 changes (95/26/29), controls 14,064; sweet per 3 in DEEPER +1.55 [+0.41,+2.71];"
)
say("   xwoba after-slump width +13.88, depth +18.14; chase validation full-none +2.23; whiff -1.70; sweet -0.68")
say("=" * 110)
E_o, dropped_o, nA_o = v2_events(OW, OE, OA, "persist")
E_om, _, _ = v2_events(OW, OE, OA, "move")
E_i, dropped_i, nA_i = v2_events(IW, IE, IA, "move")
E_ip, _, _ = v2_events(IW, IE, IA, "persist")
say(
    f"angle events dropped for overlapping a width/depth change within 30 days: original {dropped_o} of {nA_o}; ITT {dropped_i} of {nA_i}"
)
cB_orig = controls("grid_v2", E_o, "orig")
cB_itt = controls("grid_v2", E_i, "itt")
say(f"control dates: original rule {len(cB_orig):,}; ITT rule {len(cB_itt):,}")
POST = {}
ch_o = build_bayes(E_o, cB_orig)
POST["orig"] = bayes(ch_o, "ORIGINAL (a): original events, dose = persist, original controls  [reproduction]")


def decompose(ch, label, metrics=("chase", "sweet")):
    """2026 log score of the full model vs every drop-one-term refit, plus refit intercept-only and no-effect models."""
    tr, te = ch[ch.year != "2026"], ch[ch.year == "2026"]
    say(
        f"\n{'=' * 110}\nCHASE VALIDATION DECOMPOSITION ({label}): train 2024-25, score the 2026 changes; every model REFIT "
        f"(own tau posterior); delta = model minus full (negative = that model predicts 2026 worse = the dropped term(s) helped)\n{'=' * 110}"
    )
    for m in metrics:
        rng = np.random.default_rng(21)
        full = lpd(fit(tr, m, rng), te, m, rng)
        say(f"  {m}: n test {full[3]}; FULL model log score {full[0]:.2f}")
        models = [(f"drop {NAMES[j]}", [c for c in range(7) if c != j]) for j in range(7)]
        models += [
            ("intercept-only (3 lever intercepts)", [0, 1, 2]),
            ("slopes only (no intercepts)", [3, 4, 5, 6]),
            ("no effect (tau only)", []),
        ]
        for lab, cols in models:
            sc = lpd(fit(tr, m, rng, cols=cols), te, m, rng)[0]
            say(f"    {lab:40s} {sc:9.2f}   delta vs full {sc - full[0]:+6.2f}")
        # which 2026 events carry the chase gain: per-event log-density difference full vs no-effect
        pf = fit(tr, m, rng)
        pn = fit(tr, m, rng, cols=[])
        d = te[te[f"y_{m}"].notna()]
        y, v = d[f"y_{m}"].to_numpy(float), d[f"v_{m}"].to_numpy(float)
        muf = pf["beta"] @ design(d).T
        sdf = np.sqrt(v[None, :] + pf["tau"][:, None] ** 2)
        sdn = np.sqrt(v[None, :] + pn["tau"][:, None] ** 2)
        gain = np.log(norm.pdf(y[None, :], muf, sdf).mean(0)) - np.log(norm.pdf(y[None, :], 0, sdn).mean(0))
        g = d.assign(gain=gain, pred=muf.mean(0))[["name", "lever", "size", f"y_{m}", "pred", "gain"]].sort_values(
            "gain"
        )
        say(
            f"    per-lever sum of (full minus no-effect) log density: "
            + ", ".join(
                f"{l} {g[g.lever == l].gain.sum():+.2f} (n {(g.lever == l).sum()})" for l in ("width", "depth", "angle")
            )
        )
        say(
            f"    largest single-event contributions: "
            + "; ".join(
                f"{r['name']} ({r.lever} {r['size']:+.1f}: y {r[f'y_{m}']:+.1f}, pred {r.pred:+.1f}, {r.gain:+.2f})"
                for _, r in pd.concat([g.head(2), g.tail(3)]).iterrows()
            )
        )


decompose(ch_o, "ORIGINAL events, dose = persist, original controls")
POST["bridge"] = bayes(
    build_bayes(E_om, cB_orig), "BRIDGE: original events, dose = move (15-game change), original controls"
)
POST["itt_b"] = bayes(build_bayes(E_i, cB_orig), "ITT (b): ITT events, dose = move, original controls")
POST["itt_c"] = bayes(build_bayes(E_i, cB_itt), "ITT (c): ITT events, dose = move, ITT controls")
POST["itt_c_persist"] = bayes(
    build_bayes(E_ip, cB_itt), "ITT (c'): ITT events, dose = persist (v2 definition), ITT controls"
)

import pickle

pickle.dump(
    {k: {m: dict(beta=p["beta"], tau=p["tau"]) for m, p in v.items()} for k, v in POST.items()},
    open("itt_posteriors.pkl", "wb"),
)
open("itt_run.txt", "w").write("\n".join(LOG) + "\n")
