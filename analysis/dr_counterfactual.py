"""Does the stance-change result depend on HOW bounce-back is estimated? (Sloan paper; 2026-09-30)

The paper's counterfactual for a changer = the average (after - before) of 20 non-changers matched on 3 numbers (run-up xwOBA
slump, run-up whiff slump, baseline xwOBA). Here the counterfactual is instead LEARNED from the control dates (~14,000 in the
Bayesian design, ~6,000 in the run-up design) with many pre-change covariates, and combined with a model of who changes:

  nn     paper: y = changer's (after - before) - mean over 20 matched controls                 [reproduced first]
  gbm    regression imputation: y = changer's (after - before) - mu(X); mu = gradient boosting fitted on control dates,
         cross-fitted by hitter (5 folds: no hitter's own dates predict him)
  ridge  the same with a linear (ridge) model, to see whether nonlinearity matters
  bc     bias-corrected matching (Abadie & Imbens, 2011): nn - [mu(X_changer) - mean mu(X_matched controls)]
  dr     doubly robust ATT (AIPW, normalized weights): mean over changers of (after - before - mu) minus the propensity-odds-
         weighted mean of the controls' residuals; propensity = L2 logistic model of changing, cross-fitted by hitter;
         SE = hitter-cluster bootstrap (nuisance models held fixed)
X, all measured BEFORE the change date: the six outcome rates (whiff, chase, sweet-spot, K, xwOBA, bat speed) in the baseline
(days -90..-31), run-up (-30..-1) and season-to-date before the baseline, their sample sizes and run-up slumps; prior-season
rates; day of season, season, batting side, age, height, weight; league drift (other hitters' mean after-minus-before within
+-15 days, own hitter excluded).

1. Score every counterfactual on the CONTROLS, where the truth is observed: which predicts a non-changer's after-minus-before
   best? (out-of-fold; hyperparameters picked on controls only, never on changers)
2. Balance: do the matched controls resemble the changers on what the matching ignores?
3. B design (Bayesian model; paper Tables 3-4): after [0,59] minus baseline [-90,-31]; same model refit on each y.
4. U design (Figure 3 / slump test): after [0,59] minus run-up [-30,-1] -> effect beyond bounce-back + bounce-back share.
"""

import numpy as np, pandas as pd, warnings
from scipy.stats import norm
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

LOG = []


def say(s=""):
    print(s, flush=True)
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
KEY = ["batter", "side", "year", "r"]
rng = np.random.default_rng(20260930)


# ------------------------------------------------------------------ events (as stance_bayes_v2.py / slump_test.py)
def prep(ev, lever, dose="persist"):
    ev = ev.copy()
    ev["lever"] = lever
    ev["year"] = ev.year.astype(str)
    ev["t"] = (pd.to_datetime(ev.event_date) - T0).dt.days
    ev["size"] = ev[dose]
    return ev


OW, OE, OA = pd.read_csv("width_events.csv"), pd.read_csv("events_thr4.csv"), pd.read_csv("angle_events.csv")


def v2_events():
    W, E, A = prep(OW, "width"), prep(OE, "depth"), prep(OA, "angle")
    wd = pd.concat([W, E])
    wd["t0"] = pd.to_datetime(wd.event_date)
    A["t0"] = pd.to_datetime(A.event_date)
    ov = A.apply(lambda r: ((wd.bsy == r.bsy) & ((wd.t0 - r.t0).dt.days.abs() <= 30)).any(), axis=1)
    EV = pd.concat([W, E, A[~ov].drop(columns=["t0"])], ignore_index=True)
    return EV[EV["size"].notna()].reset_index(drop=True)


EV_B = v2_events()  # width + depth + isolated angle (Bayesian design)
EV_U = pd.concat([prep(OW, "width"), prep(OE, "depth")], ignore_index=True)  # width + depth (slump design)

# ------------------------------------------------------------------ windows for every (hitter-season, grid or event date)
G = {}
for key, g in D.groupby(["batter", "side", "year"]):
    g = g.sort_values("t")
    G[key] = (g.t.to_numpy(), np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)]))
ev_days = pd.concat([EV_B, EV_U]).groupby(["batter", "side", "year"]).t.apply(lambda z: sorted(set(z))).to_dict()


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
    B, U, A, P = win(r - 90, r - 31), win(r - 30, r - 1), win(r, r + 59), win(np.full(len(r), -(10**6)), r - 91)
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
        PA_P=P[:, 0],
    )
    for V, lab in ((B, "B"), (U, "U"), (A, "A"), (P, "P")):
        d.update(rates(V, lab))
    rows.append(pd.DataFrame(d))
WT = pd.concat(rows, ignore_index=True)
WT["full"] = (WT.PA_B >= 40) & (WT.PA_U >= 40) & (WT.PA_A >= 60)
off = WT.r - (WT.tmin + 30)
WT["grid_v2"] = (off >= 0) & (off % 3 == 0) & (WT.r < WT.tmax - 59)
WT["grid_slump"] = WT.grid_v2 & (WT.r >= WT.tmin + 90)
for m in MET:
    WT.loc[WT.PA_P < 40, f"{m}_P"] = np.nan
    WT[f"slump_{m}"] = WT[f"{m}_U"] - WT[f"{m}_B"]
# prior season (>= 100 PA)
S = D.groupby(["batter", "side", "year"])[COLS].sum()
PV = pd.DataFrame(
    {f"{m}_prev": np.where(S[d] > 0, sc * S[n] / S[d].where(S[d] > 0, 1), np.nan) for m, (n, d, sc) in MET.items()},
    index=S.index,
)
PV["PA_prev"] = S.PA
PV = PV.reset_index()
PV["year"] = (PV.year.astype(int) + 1).astype(str)
PV.loc[PV.PA_prev < 100, [f"{m}_prev" for m in MET]] = np.nan
WT = WT.merge(PV, on=["batter", "side", "year"], how="left")
bio = pd.read_csv("bio_all.csv")
bio["birth"] = pd.to_datetime(bio.birth, errors="coerce")
WT = WT.merge(bio, on="batter", how="left")
WT["age"] = ((T0 + pd.to_timedelta(WT.r, unit="D")) - WT.birth).dt.days / 365.25
WT["doy"] = WT.r - WT.year.map({y: (pd.Timestamp(f"{y}-01-01") - T0).days for y in WT.year.unique()})
WT["yr"] = WT.year.astype(int)
WT["lhb"] = (WT.side == "L").astype(float)
WT["xw_base"], WT["wh_base"], WT["chase_base"] = WT.xwoba_B, WT.whiff_B, WT.chase_B
WT["slump_xw"], WT["slump_wh"] = WT.slump_xwoba, WT.slump_whiff
WT["chase_prone"] = (WT.chase_base / 100 > WT.year.map(chase_cut)).astype(float)


def controls(grid_col, ev):
    days = ev.groupby(["batter", "side", "year"]).t.apply(np.array).to_dict()
    c = WT[WT[grid_col] & WT.full].copy()
    keep = np.ones(len(c), bool)
    for idx, (b, s, y, r) in enumerate(zip(c.batter, c.side, c.year, c.r)):
        e = days.get((b, s, y))
        if e is not None and len(e):
            keep[idx] = np.abs(e - r).min() > 90
    return c[keep]


def attach(ev, cols):
    x = ev[cols].rename(columns={"t": "r"}).merge(WT, on=KEY, how="left")
    return x[x.full.fillna(False).astype(bool)].copy()


Z = ["slump_xw", "slump_wh", "xw_base"]


def match(ch, co):
    """the paper's matching: 20 nearest controls, same season, within 30 days, other hitters, on standardized Z."""
    sdz = co[Z].std()
    out = {}
    for i, e in ch.iterrows():
        pool = co[(co.year == e.year) & ((co.r - e.r).abs() <= 30) & (co.batter != e.batter)].dropna(subset=Z)
        dist = (((pool[Z].astype(float) - e[Z].astype(float)) / sdz) ** 2).sum(1).astype(float)
        out[i] = dist.nsmallest(20).index
    return out


def match_controls(co):
    """the same matching applied to every control date (its truth is known): list of 20 matched control positions."""
    sdz = co[Z].std().to_numpy()
    zz = co[Z].to_numpy(float)
    rr = co.r.to_numpy()
    bb = co.batter.to_numpy()
    yy = co.year.to_numpy()
    okz = ~np.isnan(zz).any(1)
    out = [None] * len(co)
    for y in np.unique(yy):
        idx = np.where((yy == y) & okz)[0]
        idx = idx[np.argsort(rr[idx], kind="stable")]
        rs = rr[idx]
        for j in np.where(yy == y)[0]:
            if not okz[j]:
                continue
            lo, hi = np.searchsorted(rs, rr[j] - 30, "left"), np.searchsorted(rs, rr[j] + 30, "right")
            cand = idx[lo:hi]
            cand = cand[bb[cand] != bb[j]]
            d = (((zz[cand] - zz[j]) / sdz) ** 2).sum(1)
            out[j] = cand[np.argsort(d, kind="stable")[:20]]
    return out


def drift(frame, co, dcols):
    """leave-own-hitter-out mean of the controls' after-minus-before within +-15 days, same season."""
    out = np.full((len(frame), len(dcols)), np.nan)
    fy, fr, fb = frame.year.to_numpy(), frame.r.to_numpy(), frame.batter.to_numpy()
    for y in np.unique(fy):
        c = co[co.year == y].sort_values("r", kind="stable")
        rr, bb = c.r.to_numpy(), c.batter.to_numpy()
        vals = c[dcols].to_numpy(float)
        ok = ~np.isnan(vals)
        v0 = np.where(ok, vals, 0.0)
        csum = np.vstack([np.zeros((1, len(dcols))), np.cumsum(v0, 0)])
        cnt = np.vstack([np.zeros((1, len(dcols))), np.cumsum(ok, 0)])
        for k in np.where(fy == y)[0]:
            lo, hi = np.searchsorted(rr, fr[k] - 15, "left"), np.searchsorted(rr, fr[k] + 15, "right")
            s, n = csum[hi] - csum[lo], cnt[hi] - cnt[lo]
            own = bb[lo:hi] == fb[k]
            if own.any():
                s = s - v0[lo:hi][own].sum(0)
                n = n - ok[lo:hi][own].sum(0)
            out[k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


BASEF = []
for m in MET:
    BASEF += [f"{m}_B", f"{m}_U", f"n_{m}_B", f"n_{m}_U", f"slump_{m}", f"{m}_P", f"{m}_prev"]
BASEF += ["PA_P", "PA_prev", "doy", "yr", "lhb", "age", "height_in", "weight_lb"]
PF = [
    "slump_xw",
    "slump_wh",
    "xwoba_B",
    "whiff_B",
    "chase_B",
    "sweet_B",
    "K_B",
    "bat_speed_B",
    "PA_U",
    "PA_B",
    "doy",
    "yr",
    "age",
]
CFG = [
    dict(max_leaf_nodes=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200),
    dict(max_leaf_nodes=8, max_iter=200, learning_rate=0.05, min_samples_leaf=200),
    dict(max_leaf_nodes=8, max_iter=500, learning_rate=0.03, min_samples_leaf=100),
    dict(max_leaf_nodes=16, max_iter=300, learning_rate=0.03, min_samples_leaf=300),
]
ALPHAS = [1.0, 10.0, 100.0, 1000.0]


class RidgeW:
    def __init__(self, alpha):
        self.p = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha)
        )

    def fit(self, X, y, sample_weight=None):
        self.p.fit(X, y, ridge__sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.p.predict(X)


def hgb(cfg):
    return HistGradientBoostingRegressor(l2_regularization=1.0, early_stopping=False, random_state=0, **cfg)


def crossfit(make, Xc, yc, wc, fc, Xt, ft):
    pc, pt = np.full(len(Xc), np.nan), np.full(len(Xt), np.nan)
    for k in range(5):
        tr = (fc != k) & ~np.isnan(yc)
        mdl = make().fit(Xc[tr], yc[tr], sample_weight=wc[tr])
        if (fc == k).any():
            pc[fc == k] = mdl.predict(Xc[fc == k])
        if (ft == k).any():
            pt[ft == k] = mdl.predict(Xt[ft == k])
    return pc, pt


def wmse(y, p, w):
    ok = ~np.isnan(y) & ~np.isnan(p)
    return (w[ok] * (y[ok] - p[ok]) ** 2).sum() / w[ok].sum()


def cluster_boot(groups_num, groups_den, B=2000):
    """bootstrap over hitters: each element is (per-hitter numerator sums, per-hitter denominator sums) for one ratio;
    returns draws of sum_k sign_k * ratio_k."""
    nb = len(groups_num[0][1])
    cnt = rng.multinomial(nb, np.full(nb, 1 / nb), size=B).astype(float)
    tot = np.zeros(B)
    for (sign, num), (_, den) in zip(groups_num, groups_den):
        tot += sign * (cnt @ num) / np.maximum(cnt @ den, 1e-12)
    return tot


# ------------------------------------------------------------------ Bayesian model (identical to stance_bayes_v2 / itt_analysis)
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
    if k:
        for i, tau in enumerate(taus):
            if tau not in cache:
                Wt = 1 / (v + tau**2)
                V = np.linalg.inv(Xd.T @ (Xd * Wt[:, None]) + np.eye(k) / s**2)
                cache[tau] = (V @ (Xd.T @ (Wt * y)), V)
            mu, V = cache[tau]
            betas[i] = rg.multivariate_normal(mu, V)
    return dict(beta=betas, tau=taus, cols=cols)


def lpd(post, d, ycol, vcol):
    ok = d[ycol].notna()
    d = d[ok]
    y, v = d[ycol].to_numpy(float), d[vcol].to_numpy(float)
    Xd = design(d, post["cols"])
    mu = post["beta"] @ Xd.T if Xd.shape[1] else np.zeros((len(post["tau"]), len(y)))
    sd = np.sqrt(v[None, :] + post["tau"][:, None] ** 2)
    return np.log(norm.pdf(y[None, :], mu, sd).mean(0)).sum()


# ================================================================== run both designs
bats = np.array(sorted(set(WT.batter)))
FOLD = dict(zip(bats, rng.permutation(len(bats)) % 5))
RESULTS = {}
for DES, grid_col, EVX, BEFORE in (("B", "grid_v2", EV_B, "B"), ("U", "grid_slump", EV_U, "U")):
    co = controls(grid_col, EVX)
    ch = attach(EVX, ["batter", "side", "year", "t", "lever", "size", "name", "dir"])
    say("\n" + "=" * 118)
    say(
        f"DESIGN {DES}: after [0,59] minus {'baseline [-90,-31]' if BEFORE == 'B' else 'run-up [-30,-1]'}; "
        f"{len(ch)} changes with full windows ({(ch.lever == 'width').sum()} width, {(ch.lever == 'depth').sum()} depth, "
        f"{(ch.lever == 'angle').sum()} angle); {len(co):,} control dates"
    )
    say("=" * 118)
    for X in (co, ch):
        for m in MET:
            X[f"d_{m}"] = X[f"{m}_A"] - X[f"{m}_{BEFORE}"]
    DC = [f"d_{m}" for m in MET]
    for X in (co, ch):
        dr_ = drift(X, co, DC)
        for j, m in enumerate(MET):
            X[f"drift_{m}"] = dr_[:, j]
    FEAT = BASEF + [f"drift_{m}" for m in MET]
    Xc, Xt = co[FEAT].to_numpy(float), ch[FEAT].to_numpy(float)
    fc, ft = co.batter.map(FOLD).to_numpy(), ch.batter.map(FOLD).to_numpy()
    MT = match(ch, co)  # paper matching for changers (pandas; identical to the paper's code)
    MC = match_controls(co)  # the same rule applied to each control date
    pos = pd.Series(np.arange(len(co)), index=co.index)

    # ---------------- 1. which counterfactual predicts non-changers best?
    say(
        f"\n1. ACCURACY ON CONTROL DATES (truth observed): weighted MSE of after-minus-before, out of fold; ratio vs the paper's"
    )
    say(
        f"   matching (< 1 = better than matching); 90% interval of the ratio from a hitter-cluster bootstrap. 'slumping' = controls"
    )
    say(f"   whose run-up xwOBA slump is in the worst quarter (the changers' region).")
    say(
        f"   {'outcome':10s} {'model':6s} {'zero':>7s} {'drift':>7s} {'match20':>8s} {'ridge':>7s} {'gbm':>7s} | gbm/match [90%]      | "
        f"slumping: gbm/match [90%]     | gbm config"
    )
    PRED = {}
    wq = co.slump_xw.quantile(0.25)
    for m in MET:
        yc = co[f"d_{m}"].to_numpy(float)
        nb_, na_ = co[f"n_{m}_{BEFORE}"].to_numpy(float), co[f"n_{m}_A"].to_numpy(float)
        wc = 1 / (1 / np.maximum(nb_, 1) + 1 / np.maximum(na_, 1))
        best = None
        for ci, cfg in enumerate(CFG):
            pc, pt = crossfit(lambda: hgb(cfg), Xc, yc, wc, fc, Xt, ft)
            sc = wmse(yc, pc, wc)
            if best is None or sc < best[0]:
                best = (sc, ci, pc, pt)
        _, ci, pc_g, pt_g = best
        bestr = None
        for a in ALPHAS:
            pc, pt = crossfit(lambda: RidgeW(a), Xc, yc, wc, fc, Xt, ft)
            sc = wmse(yc, pc, wc)
            if bestr is None or sc < bestr[0]:
                bestr = (sc, a, pc, pt)
        _, a_r, pc_r, pt_r = bestr
        pc_nn = np.array([np.nanmean(yc[ix]) if ix is not None and len(ix) else np.nan for ix in MC])
        pc_dr = co[f"drift_{m}"].to_numpy(float)
        ok = ~np.isnan(yc) & ~np.isnan(pc_nn) & ~np.isnan(pc_g) & ~np.isnan(pc_r) & ~np.isnan(pc_dr)
        e = {
            k: w_
            for k, w_ in (
                ("zero", np.zeros(len(yc))),
                ("drift", pc_dr),
                ("match20", pc_nn),
                ("ridge", pc_r),
                ("gbm", pc_g),
            )
        }
        M = {k: wmse(yc[ok], p[ok], wc[ok]) for k, p in e.items()}

        # bootstrap the ratio gbm / match20 by hitter (all controls and slumping controls)
        def ratio_ci(sel):
            b_ = co.batter.to_numpy()[sel]
            ub, inv = np.unique(b_, return_inverse=True)
            ng = np.bincount(inv, weights=(wc[sel] * (yc[sel] - pc_g[sel]) ** 2), minlength=len(ub))
            nn_ = np.bincount(inv, weights=(wc[sel] * (yc[sel] - pc_nn[sel]) ** 2), minlength=len(ub))
            cnt = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=2000).astype(float)
            rr_ = (cnt @ ng) / (cnt @ nn_)
            return ng.sum() / nn_.sum(), np.quantile(rr_, [0.05, 0.95])

        r_all, ci_all = ratio_ci(ok)
        r_sl, ci_sl = ratio_ci(ok & (co.slump_xw.to_numpy() <= wq))
        say(
            f"   {m:10s} {'':6s} {M['zero']:7.1f} {M['drift']:7.1f} {M['match20']:8.1f} {M['ridge']:7.1f} {M['gbm']:7.1f} | "
            f"{r_all:.3f} [{ci_all[0]:.3f}, {ci_all[1]:.3f}] | {r_sl:.3f} [{ci_sl[0]:.3f}, {ci_sl[1]:.3f}]        | #{ci} ridge a={a_r:g}"
        )
        pc_bcadj = np.array([np.nanmean(pc_g[ix]) if ix is not None and len(ix) else np.nan for ix in MC])
        PRED[m] = dict(pc_g=pc_g, pt_g=pt_g, pc_r=pc_r, pt_r=pt_r, wc=wc, pc_nn=pc_nn, pc_bcadj=pc_bcadj)

    # ---------------- changer effects under each counterfactual
    for m in MET:
        dcol = f"d_{m}"
        yc = co[dcol].to_numpy(float)
        pc_g = PRED[m]["pc_g"]
        ch[f"y_nn_{m}"] = [ch.at[i, dcol] - co.loc[MT[i], dcol].mean() for i in ch.index]
        ch[f"cf_nn_{m}"] = [co.loc[MT[i], dcol].mean() for i in ch.index]
        ch[f"y_gbm_{m}"] = ch[dcol].to_numpy(float) - PRED[m]["pt_g"]
        ch[f"y_ridge_{m}"] = ch[dcol].to_numpy(float) - PRED[m]["pt_r"]
        ch[f"y_bc_{m}"] = [
            ch.at[i, f"y_nn_{m}"] - (PRED[m]["pt_g"][k] - np.nanmean(pc_g[pos[MT[i]].to_numpy()]))
            for k, i in enumerate(ch.index)
        ]
        ch[f"cf_gbm_{m}"] = PRED[m]["pt_g"]
        ch[f"v_{m}"] = (
            UNIT_VAR[m] * (1 / np.maximum(ch[f"n_{m}_{BEFORE}"], 1) + 1 / np.maximum(ch[f"n_{m}_A"], 1)) * (1 + 1 / 20)
        )

    # ---------------- 2. balance on what the matching ignores
    say(
        f"\n2. BALANCE: standardized mean difference, changers minus comparison (control SD units); |SMD| > 0.1 = imbalance"
    )
    BAL = [
        "xwoba_B",
        "slump_xw",
        "whiff_B",
        "slump_wh",
        "chase_B",
        "slump_chase",
        "sweet_B",
        "slump_sweet",
        "K_B",
        "bat_speed_B",
        "xwoba_prev",
        "chase_prev",
        "sweet_prev",
        "PA_U",
        "age",
        "doy",
    ]
    say(
        f"   {'feature':14s} {'vs all controls':>16s} {'vs matched 20':>14s}   (matched on slump_xw, slump_wh, xwoba_B)"
    )
    for f in BAL:
        sd = co[f].std()
        a_ = ch[f].mean()
        mt_ = np.nanmean([co.loc[MT[i], f].mean() for i in ch.index])
        say(f"   {f:14s} {(a_ - co[f].mean()) / sd:+16.2f} {(a_ - mt_) / sd:+14.2f}")

    # ---------------- propensity (cross-fitted by hitter) for the doubly robust ATT
    grp_col = "lever" if DES == "B" else "kind"
    ch["kind"] = ch.lever + ":" + ch.dir.astype(str)
    groups = ["width", "depth", "angle"] if DES == "B" else ["width:WIDER", "width:NARROWER", "depth:BACK", "depth:UP"]
    OMEGA = {}
    for gname in groups:
        chg = ch[ch[grp_col] == gname]
        Xp = pd.concat([co[PF], chg[PF]]).to_numpy(float)
        T = np.r_[np.zeros(len(co)), np.ones(len(chg))]
        fp = np.r_[fc, chg.batter.map(FOLD).to_numpy()]
        e_c = np.full(len(co), np.nan)
        for k in range(5):
            mdl = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)
            )
            mdl.fit(Xp[fp != k], T[fp != k])
            e_c[fc == k] = mdl.predict_proba(co[PF].to_numpy(float)[fc == k])[:, 1]
        OMEGA[gname] = e_c / (1 - e_c)

    say(
        f"\n{'3' if DES == 'B' else '4'}. EFFECT BY {'LEVER (all its changes)' if DES == 'B' else 'KIND (Figure 3 cells)'}: mean effect (SE) under each counterfactual;"
        f" dr = doubly robust (cluster-bootstrap SE); ESS = effective number of weighted controls"
    )
    say(
        f"   {'group':15s} {'outcome':9s} {'n':>3s} | {'naive':>7s} {'match20 (paper)':>17s} {'bias-corr':>15s} {'ridge':>15s} {'gbm':>15s} {'dr':>15s} | "
        f"cf match / gbm | ESS"
    )
    for gname in groups:
        chg = ch[ch[grp_col] == gname]
        om = OMEGA[gname]
        for m in ("xwoba", "whiff", "chase", "sweet", "K", "bat_speed"):
            dcol = f"d_{m}"
            yc = co[dcol].to_numpy(float)
            pc_g = PRED[m]["pc_g"]
            sel = chg[f"y_nn_{m}"].notna() & chg[f"y_gbm_{m}"].notna()
            cells = []
            for k in ("nn", "bc", "ridge", "gbm"):
                v_ = chg.loc[sel, f"y_{k}_{m}"]
                cells.append(f"{v_.mean():+7.2f} ({v_.std() / np.sqrt(sel.sum()):5.2f})")
            # doubly robust
            resid_t = chg.loc[sel, f"y_gbm_{m}"].to_numpy(float)
            bt = chg.loc[sel, "batter"].to_numpy()
            okc = ~np.isnan(yc) & ~np.isnan(pc_g) & ~np.isnan(om)
            rc = (yc - pc_g)[okc]
            wv = om[okc]
            bc_ = co.batter.to_numpy()[okc]
            corr = (wv * rc).sum() / wv.sum()
            dr_est = resid_t.mean() - corr
            ub = np.unique(np.r_[bt, bc_])
            it = np.searchsorted(ub, bt)
            ic = np.searchsorted(ub, bc_)
            tn = np.bincount(it, weights=resid_t, minlength=len(ub))
            td = np.bincount(it, minlength=len(ub)).astype(float)
            cn = np.bincount(ic, weights=wv * rc, minlength=len(ub))
            cd = np.bincount(ic, weights=wv, minlength=len(ub))
            draws = cluster_boot([(+1, tn), (-1, cn)], [(+1, td), (-1, cd)])
            cells.append(f"{dr_est:+7.2f} ({draws.std():5.2f})")
            ess = wv.sum() ** 2 / (wv**2).sum()
            naive = chg.loc[sel, dcol].mean()
            say(
                f"   {gname:15s} {m:9s} {sel.sum():3d} | {naive:+7.2f} {cells[0]:>17s} "
                + " ".join(f"{c:>15s}" for c in cells[1:])
                + f" | {chg.loc[sel, f'cf_nn_{m}'].mean():+6.2f} / {chg.loc[sel, f'cf_gbm_{m}'].mean():+6.2f} | {ess:5.0f}"
            )
            RESULTS[(DES, gname, m)] = dict(
                n=int(sel.sum()),
                naive=naive,
                nn=chg.loc[sel, f"y_nn_{m}"].mean(),
                gbm=chg.loc[sel, f"y_gbm_{m}"].mean(),
                dr=dr_est,
                dr_se=draws.std(),
                cf_nn=chg.loc[sel, f"cf_nn_{m}"].mean(),
                cf_gbm=chg.loc[sel, f"cf_gbm_{m}"].mean(),
                corr=corr,
            )

    if DES == "U":
        say(f"\n   BOUNCE-BACK SHARE of the raw xwOBA gain, width + depth changes pooled (paper: about 30%):")
        wd = ch[ch.lever.isin(["width", "depth"]) & ch.y_nn_xwoba.notna() & ch.y_gbm_xwoba.notna()]
        naive = wd.d_xwoba.mean()
        corr_all = np.average(
            [RESULTS[("U", g, "xwoba")]["corr"] for g in groups],
            weights=[RESULTS[("U", g, "xwoba")]["n"] for g in groups],
        )
        say(
            f"   raw gain {naive:+.1f} | bounce-back: match20 {wd.cf_nn_xwoba.mean():+.1f} ({wd.cf_nn_xwoba.mean() / naive:.0%}) | "
            f"gbm {wd.cf_gbm_xwoba.mean():+.1f} ({wd.cf_gbm_xwoba.mean() / naive:.0%}) | dr {wd.cf_gbm_xwoba.mean() + corr_all:+.1f} "
            f"({(wd.cf_gbm_xwoba.mean() + corr_all) / naive:.0%})  [n {len(wd)}]"
        )

    if DES == "B":
        # reproduction check against the paper's stored y
        ref = pd.read_parquet("stance_bayes_v2_events_B.parquet")
        ref["year"] = ref.year.astype(str)
        cmp_ = ch.merge(ref[KEY + ["lever"] + [f"y_{m}" for m in MET]], on=KEY + ["lever"], how="inner")
        dif = max(np.nanmax(np.abs(cmp_[f"y_nn_{m}"] - cmp_[f"y_{m}"])) for m in MET)
        say(f"\n   reproduction: {len(cmp_)} of {len(ref)} paper changes matched; max |y_nn - paper y| = {dif:.2e}")
        # ---------------- the Bayesian model refit on each counterfactual
        say(
            f"\n3b. BAYESIAN MODEL (paper Table 3/4) REFIT ON EACH COUNTERFACTUAL: posterior mean [90% interval]; * = excludes 0"
        )
        CELLS = [
            ("sweet", 5),
            ("xwoba", 0),
            ("xwoba", 1),
            ("K", 0),
            ("chase", 2),
            ("chase", 5),
            ("whiff", 3),
            ("xwoba", 5),
        ]
        say(f"   {'cell':34s} " + " ".join(f"{k:>24s}" for k in ("match20 (paper)", "bias-corrected", "ridge", "gbm")))
        POSTS = {}
        for k in ("nn", "bc", "ridge", "gbm"):
            for m in MET:
                POSTS[(k, m)] = fit(ch, f"y_{k}_{m}", f"v_{m}", m, np.random.default_rng(7))
        for m, j in CELLS:
            cells = []
            for k in ("nn", "bc", "ridge", "gbm"):
                b = POSTS[(k, m)]["beta"][:, j]
                lo, hi = np.quantile(b, [0.05, 0.95])
                cells.append(f"{b.mean():+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if lo > 0 or hi < 0 else ' '}")
            say(f"   {m + ': ' + NAMES[j]:34s} " + " ".join(f"{c:>24s}" for c in cells))
        say(
            f"   {'tau (xwOBA / sweet / chase)':34s} "
            + " ".join(
                f"{POSTS[(k, 'xwoba')]['tau'].mean():>12.2f} /{POSTS[(k, 'sweet')]['tau'].mean():5.2f} /{POSTS[(k, 'chase')]['tau'].mean():5.2f}"
                for k in ("nn", "bc", "ridge", "gbm")
            )
        )
        say(f"\n   full 7-term posterior for each counterfactual (all outcomes):")
        for k in ("nn", "bc", "ridge", "gbm"):
            for m in MET:
                b = POSTS[(k, m)]["beta"]
                lo, hi = np.quantile(b, [0.05, 0.95], axis=0)
                say(
                    f"   {k:6s} {m:9s} tau {POSTS[(k, m)]['tau'].mean():5.2f} | "
                    + " | ".join(f"{NAMES[j]} {b[:, j].mean():+.2f} [{lo[j]:+.2f},{hi[j]:+.2f}]" for j in range(7))
                )
        say(
            f"\n3c. VALIDATION 2024-25 -> 2026 on each counterfactual (full minus refit no-effect | full minus refit intercept-only)"
        )
        tr, te = ch[ch.year != "2026"], ch[ch.year == "2026"]
        say(
            f"   {'outcome':10s} "
            + " ".join(f"{k:>18s}" for k in ("match20 (paper)", "bias-corrected", "ridge", "gbm"))
            + "   n test"
        )
        for m in MET:
            cells = []
            for k in ("nn", "bc", "ridge", "gbm"):
                rg = np.random.default_rng(11)
                yv = f"y_{k}_{m}"
                full = lpd(fit(tr, yv, f"v_{m}", m, rg), te, yv, f"v_{m}")
                ic = lpd(fit(tr, yv, f"v_{m}", m, rg, cols=[0, 1, 2]), te, yv, f"v_{m}")
                ne = lpd(fit(tr, yv, f"v_{m}", m, rg, cols=[]), te, yv, f"v_{m}")
                cells.append(f"{full - ne:+7.2f} | {full - ic:+6.2f}")
            say(f"   {m:10s} " + " ".join(f"{c:>18s}" for c in cells) + f"   {te[f'y_nn_{m}'].notna().sum()}")
    if DES == "B":
        say(
            f"\n3d. SAME, WITH EACH METHOD'S NOISE MEASURED ON THE CONTROLS: residual^2 = a*u/n_before + b*u/n_after + c fitted on"
        )
        say(
            f"    control dates (u = unit variance; paper assumes a = b = 1.05, c = 0); v_i from the fit. Coefficients (a, b, c):"
        )
        for m in MET:
            yc = co[f"d_{m}"].to_numpy(float)
            P_ = PRED[m]
            A_ = np.column_stack(
                [
                    UNIT_VAR[m] / np.maximum(co[f"n_{m}_{BEFORE}"].to_numpy(float), 1),
                    UNIT_VAR[m] / np.maximum(co[f"n_{m}_A"].to_numpy(float), 1),
                    np.ones(len(co)),
                ]
            )
            At = np.column_stack(
                [
                    UNIT_VAR[m] / np.maximum(ch[f"n_{m}_{BEFORE}"].to_numpy(float), 1),
                    UNIT_VAR[m] / np.maximum(ch[f"n_{m}_A"].to_numpy(float), 1),
                    np.ones(len(ch)),
                ]
            )
            res = {
                "nn": yc - P_["pc_nn"],
                "gbm": yc - P_["pc_g"],
                "bc": (yc - P_["pc_nn"]) - (P_["pc_g"] - P_["pc_bcadj"]),
            }
            cells = []
            for k, r_ in res.items():
                ok = ~np.isnan(r_)
                coef, _ = nnls(A_[ok], r_[ok] ** 2)
                ch[f"vc_{k}_{m}"] = At @ coef
                cells.append(f"{k} ({coef[0]:.2f}, {coef[1]:.2f}, {coef[2]:.2f})")
            say(f"    {m:10s} " + "   ".join(cells))
        CAL = {}
        for k in ("nn", "bc", "gbm"):
            for m in MET:
                CAL[(k, m)] = fit(ch, f"y_{k}_{m}", f"vc_{k}_{m}", m, np.random.default_rng(7))
        say(f"   {'cell':34s} " + " ".join(f"{k:>24s}" for k in ("match20 (paper)", "bias-corrected", "gbm")))
        for m, j in CELLS:
            cells = []
            for k in ("nn", "bc", "gbm"):
                b = CAL[(k, m)]["beta"][:, j]
                lo, hi = np.quantile(b, [0.05, 0.95])
                cells.append(f"{b.mean():+7.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if lo > 0 or hi < 0 else ' '}")
            say(f"   {m + ': ' + NAMES[j]:34s} " + " ".join(f"{c:>24s}" for c in cells))
        say(
            f"   {'tau (xwOBA / sweet / chase)':34s} "
            + " ".join(
                f"{CAL[(k, 'xwoba')]['tau'].mean():>12.2f} /{CAL[(k, 'sweet')]['tau'].mean():5.2f} /{CAL[(k, 'chase')]['tau'].mean():5.2f}"
                for k in ("nn", "bc", "gbm")
            )
        )
        say(f"\n   full 7-term posterior, calibrated variances:")
        for k in ("nn", "bc", "gbm"):
            for m in MET:
                b = CAL[(k, m)]["beta"]
                lo, hi = np.quantile(b, [0.05, 0.95], axis=0)
                say(
                    f"   {k:6s} {m:9s} tau {CAL[(k, m)]['tau'].mean():5.2f} | "
                    + " | ".join(f"{NAMES[j]} {b[:, j].mean():+.2f} [{lo[j]:+.2f},{hi[j]:+.2f}]" for j in range(7))
                )
        tr, te = ch[ch.year != "2026"], ch[ch.year == "2026"]  # re-slice: vc_ columns were added after 3c
        say(
            f"\n3e. VALIDATION with calibrated variances (full minus refit no-effect | full minus refit intercept-only)"
        )
        say(
            f"   {'outcome':10s} "
            + " ".join(f"{k:>18s}" for k in ("match20 (paper)", "bias-corrected", "gbm"))
            + "   n test"
        )
        for m in MET:
            cells = []
            for k in ("nn", "bc", "gbm"):
                rg = np.random.default_rng(11)
                yv, vv = f"y_{k}_{m}", f"vc_{k}_{m}"
                full = lpd(fit(tr, yv, vv, m, rg), te, yv, vv)
                ic = lpd(fit(tr, yv, vv, m, rg, cols=[0, 1, 2]), te, yv, vv)
                ne = lpd(fit(tr, yv, vv, m, rg, cols=[]), te, yv, vv)
                cells.append(f"{full - ne:+7.2f} | {full - ic:+6.2f}")
            say(f"   {m:10s} " + " ".join(f"{c:>18s}" for c in cells) + f"   {te[f'y_nn_{m}'].notna().sum()}")
    keep = (
        KEY
        + ["lever", "size", "name", "dir", "chase_prone"]
        + [c for c in ch.columns if c.startswith(("y_", "cf_", "v_", "d_"))]
    )
    ch[keep].to_parquet(f"dr_counterfactual_events_{DES}.parquet")
open("dr_counterfactual.txt", "w").write("\n".join(LOG) + "\n")
