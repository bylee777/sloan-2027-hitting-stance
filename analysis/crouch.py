"""Are we accounting for how upright hitters stand (crouch)? The Savant stance data record only the feet, so posture comes from Statcast's
per-pitch strike-zone top and bottom, which in 2024-25 Hawk-Eye set from each hitter's body as the pitch arrived (top ~ midpoint of
shoulders and belt, bottom ~ hollow below the knee). In 2026 the zone is a fixed share of height (no posture) -> 2024-25 only.
TAKES ONLY (ball / called strike / blocked ball): on swings the zone values move with the swing (the leak in paper §3.3).
Crouch = zone top lowered (per 1 inch, within hitter the hitter FE absorbs height); knee bend = zone bottom lowered.
A. How much crouch varies, and how it moves with width/depth (within hitter, month to month).
B. Monthly panel 2024-25 (hitter x side + year-month FE, clustered by hitter): the four stance levers with and without crouch, and crouch's
   own contact/power effects (same outcomes as power_tradeoff.py).
C. Between hitters (season level): posture as a share of height vs outcomes, with year FE and height."""

import glob, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
from fe import demean

R = pd.concat(
    [
        pd.read_parquet(f, columns=["game_date", "game_type", "batter", "stand", "description", "sz_top", "sz_bot"])
        for f in sorted(glob.glob("raw/*.parquet"))
    ],
    ignore_index=True,
)
R = R[(R.game_type == "R")].copy()
R["game_date"] = pd.to_datetime(R.game_date).dt.strftime("%Y-%m-%d")
R = R[R.game_date < "2026-01-01"]
R["description"] = R.description.astype(object)
R = R[R.description.isin(["ball", "called_strike", "blocked_ball"])].rename(columns={"stand": "side"})
R["top_in"] = 12 * pd.to_numeric(R.sz_top, errors="coerce").astype(float)
R["bot_in"] = 12 * pd.to_numeric(R.sz_bot, errors="coerce").astype(float)
R["month"] = R.game_date.str[:7]
R["year"] = R.game_date.str[:4]
POS = (
    R.groupby(["batter", "side", "month"])
    .agg(top=("top_in", "median"), bot=("bot_in", "median"), takes=("top_in", "size"))
    .reset_index()
)
POS = POS[POS.takes >= 30]
H = pd.read_csv("bio_all.csv")[["batter", "height_in"]]
# ---------------- outcomes + stance (as power_tradeoff.py)
D = pd.read_parquet("hitter_day_outcomes.parquet").merge(
    pd.read_parquet("hitter_day_power.parquet"), on=["batter", "side", "game_date"], how="left"
)
PWR = ["HR", "XHR", "BRL", "TRK", "HARD", "EV_SUM", "TBX", "AB", "PA_p"]
D[PWR] = D[PWR].fillna(0)
MET = {
    "whiff": ("wh", "sw", 100, "sw", "whiff % per swing"),
    "K": ("K", "PA", 100, "PA", "strikeout % per PA"),
    "chase": ("oz_sw", "oz", 100, "oz", "chase %"),
    "xhr600": ("XHR", "PA_p", 600, "PA_p", "expected HR per 600 PA"),
    "brl600": ("BRL", "PA_p", 600, "PA_p", "barrels per 600 PA"),
    "hard": ("HARD", "TRK", 100, "TRK", "hard-hit % (95+ mph)"),
    "ev": ("EV_SUM", "TRK", 1, "TRK", "exit velocity (mph)"),
    "sweet": ("sweet", "bip", 100, "bip", "sweet-spot %"),
    "xwoba": ("xw_sum", "PA", 1000, "PA", "xwOBA per PA (points)"),
}
D["month"] = D.game_date.str[:7]
NEED = sorted({c for v in MET.values() for c in v[:2]} | {v[3] for v in MET.values()})
M = D.groupby(["batter", "side", "month"])[NEED].sum().reset_index()
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
X = (
    SM.merge(M, on=["batter", "side", "month"])
    .merge(POS, on=["batter", "side", "month"])
    .merge(H, on="batter", how="left")
)
X = X[(X.games >= 8) & (X.PA >= 50)].dropna(subset=["width", "depth", "off", "angle", "top", "bot"]).copy()
X["bs"] = X.batter.astype(str) + X.side
for m, (n, d, sc, w, lab) in MET.items():
    X[m] = np.where(X[d] > 0, sc * X[n] / X[d].where(X[d] > 0), np.nan)
for c in ("width", "depth", "off"):
    X[c + "3"] = X[c] / 3
X["closed10"] = X.angle / 10
X["crouch"] = -X.top
X["kneebend"] = -X.bot  # +1 = zone top (bottom) 1 inch lower


def fe_resid(d, cols):
    codes = [pd.factorize(d.bs)[0], pd.factorize(d.month)[0]]
    return demean(np.column_stack([d[c].to_numpy(float) for c in cols]), codes, iters=300, tol=1e-10)


def fe_fit(d, y, xs, w):
    Mx = fe_resid(d, [y] + xs)
    r = sm.WLS(Mx[:, 0], Mx[:, 1:], weights=d[w].to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.bs)[0]}
    )
    return r.params, r.bse


# ---------------- A. how much posture varies
print(f"2024-25 hitter-months with posture + stance + outcomes: {len(X):,} ({X.bs.nunique()} hitters)")
Ctop = fe_resid(X, ["top", "bot", "width", "depth", "off", "angle"])
print(
    f"A. zone top: between-hitter SD {X.groupby('bs').top.mean().std():.2f} in; within-hitter month-to-month SD {Ctop[:, 0].std():.2f} in "
    f"(zone bottom {Ctop[:, 1].std():.2f} in); share of hitter-months >= 1 in off the hitter's usual: {(np.abs(Ctop[:, 0]) >= 1).mean():.1%}"
)
cc = pd.DataFrame(Ctop, columns=["top", "bot", "width", "depth", "off", "angle"]).corr().round(2)
print("   within-hitter correlations of posture with the stance levers:")
print("   " + cc.loc[["top", "bot"], ["width", "depth", "off", "angle"]].to_string().replace("\n", "\n   "))
b, s = fe_fit(X, "top", ["width3", "depth3", "off3", "closed10"], "PA")
print(
    "   zone top (in) per 3 in wider / deeper / farther off / 10 deg closed: "
    + ", ".join(f"{b[i]:+.2f} (z {b[i]/s[i]:+.1f})" for i in range(4))
)
Xs = X.assign(tp=X.top / X.height_in).dropna(subset=["tp"])
print(
    f"   between hitters, zone top as a share of height: median {Xs.groupby('bs').tp.mean().median():.3f}, 10th-90th pct "
    f"{Xs.groupby('bs').tp.mean().quantile(.1):.3f}-{Xs.groupby('bs').tp.mean().quantile(.9):.3f}"
)
# ---------------- B. monthly panel with and without posture
print("\nB. WITHIN-HITTER MONTHLY PANEL 2024-25: stance levers without -> with posture controls; posture's own effects")
hdr = f"{'outcome':26s} {'3 in wider':>21s} {'3 in deeper':>21s} {'crouch: top 1 in lower':>24s} {'knees: bottom 1 in lower':>26s}"
print(hdr)
out = []
for m, (n, d, sc, w, lab) in MET.items():
    dd = X.dropna(subset=[m])
    dd = dd[dd[w] > 0]
    b0, s0 = fe_fit(dd, m, ["width3", "depth3", "off3", "closed10"], w)
    b1, s1 = fe_fit(dd, m, ["width3", "depth3", "off3", "closed10", "crouch", "kneebend"], w)
    print(
        f"{lab:26s} {b0[0]:+6.2f} -> {b1[0]:+6.2f} (z {b1[0]/s1[0]:+4.1f}) {b0[1]:+6.2f} -> {b1[1]:+6.2f} (z {b1[1]/s1[1]:+4.1f}) "
        f"{b1[4]:+11.2f} (z {b1[4]/s1[4]:+4.1f}) {b1[5]:+13.2f} (z {b1[5]/s1[5]:+4.1f})"
    )
    out.append(
        dict(
            outcome=m,
            wider_without=b0[0],
            wider_with=b1[0],
            deeper_without=b0[1],
            deeper_with=b1[1],
            off_with=b1[2],
            closed_with=b1[3],
            crouch=b1[4],
            crouch_z=b1[4] / s1[4],
            knee=b1[5],
            knee_z=b1[5] / s1[5],
        )
    )
pd.DataFrame(out).to_csv("crouch_monthly.csv", index=False)
# ---------------- C. between hitters (season level)
X["year"] = X.month.str[:4]
agg = {c: "sum" for c in NEED}
agg.update(
    dict(
        top=lambda v: np.average(v, weights=X.loc[v.index, "takes"]),
        height_in="first",
        width="mean",
        depth="mean",
        off="mean",
        angle="mean",
    )
)
B = X.groupby(["bs", "year"]).agg(agg).reset_index()
B = B[B.PA >= 250].copy()
for m, (n, d, sc, w, lab) in MET.items():
    B[m] = sc * B[n] / B[d]
B["top_pct"] = 100 * B.top / B.height_in  # zone top as % of height (lower = more crouched)
print(
    f"\nC. BETWEEN HITTERS, season level (2024-25, 250+ PA, {len(B)} hitter-seasons): per 1 point of height LOWER zone top (more crouched),"
)
print("   with year FE, height, and the four stance levers as controls")
for m, (n, d, sc, w, lab) in MET.items():
    r = smf.wls(f"{m} ~ I(-top_pct) + height_in + width + depth + off + angle + C(year)", B, weights=B[w]).fit(
        cov_type="HC1"
    )
    print(f"   {lab:26s} {r.params['I(-top_pct)']:+6.2f} (z {r.tvalues['I(-top_pct)']:+4.1f})")
