"""Matchup stance model (MSM): given a BATTER and a PITCHER, which setup (depth / off-plate / width) is best?

Pre-registered design (written before running, 2026-09-26):
 Unit: plate appearance, 2024-26 regular season. Outcomes: strikeout (pp), wOBA (pts), run value (runs/100 PA);
 mechanism: chase on out-of-zone pitches (pp, pitch level).
 Setup S_k = hitter-season mean (park-adjusted depth and off-plate; width), centred by side-year, per 3 in.
 Identification: hitter-season FE (absorbs each hitter's own level = his EVERYDAY setup effect) + pitcher-season FE
 (absorbs pitcher quality). What is left is how the payoff of a setup CHANGES with the pitcher faced:
   S_k x {same hand, FB velo, breaking %, offspeed %, breaking sweep, FB rise, extension, arm angle, zone %,
          pitcher K%, chase induced, same x breaking %, same x hitter chase, same x hitter whiff}.
 Controls (so a setup is not credited with a hitter-type effect): every hitter trait x pitcher trait, same x hitter
 traits, same x pitcher traits, side x pitcher traits, stance angle x every moderator.
 Hitter traits fixed from 2024-25 (chase, whiff, K%, bat speed, height). Pitcher K%/chase leave-this-batter-out.
 TRAIN 2024-25 -> VALIDATE 2026: lambda = coefficient on the trained matchup index (1 = real & calibrated, 0 = noise).
"""

import numpy as np, pandas as pd, warnings, pickle

warnings.filterwarnings("ignore")
from scipy.stats import chi2
from fe import demean
from stance_lib import pitches

p = pitches()
p["pa"] = p.game_pk.astype(str) + "_" + p.at_bat_number.astype(str)
p["same"] = (p.stand == p.p_throws).astype(float)
FB, BRK, OFF = {"FF", "SI"}, {"SL", "ST", "CU", "KC", "SV", "CS"}, {"CH", "FS", "FO", "SC"}
p["swing"], p["whiff"] = p.swing.astype(bool), p.whiff.astype(bool)
p["is_fb"], p["is_brk"], p["is_off"] = p.pitch_type.isin(FB), p.pitch_type.isin(BRK), p.pitch_type.isin(OFF)
for c in (
    "pfx_x",
    "pfx_z",
    "release_speed",
    "release_extension",
    "arm_angle",
    "bat_speed",
    "delta_run_exp",
    "woba_value",
    "woba_denom",
):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)

# ---------------- plate appearances
last = p[p.events.notna() & (p.events != "")]
last = last[~last.events.isin(["intent_walk", "sac_bunt", "catcher_interf", "sac_bunt_double_play", "truncated_pa"])]
pa = last[["pa", "batter", "pitcher", "stand", "p_throws", "same", "year"]].copy()
pa["K"] = last.events.str.startswith("strikeout").astype(float) * 100
wd = pd.to_numeric(last.woba_denom, errors="coerce").astype(float).to_numpy()
wv = pd.to_numeric(last.woba_value, errors="coerce").astype(float).to_numpy()
pa["woba"] = np.where(wd == 1, wv * 1000, np.nan)
pa["rv"] = pa.pa.map(p.groupby("pa").delta_run_exp.sum()) * 100

# ---------------- pitcher traits (pitcher-season)
g = p.groupby(["pitcher", "year"])
PT = pd.DataFrame(
    {
        "n_p": g.size(),
        "brk": g.is_brk.mean(),
        "off": g.is_off.mean(),
        "ext": g.release_extension.mean(),
        "arm": g.arm_angle.mean(),
        "zone": g.in_zone.mean(),
    }
)
PT["velo"] = p[p.is_fb].groupby(["pitcher", "year"]).release_speed.mean()
PT["rise"] = (p[p.is_fb].pfx_z * 12).groupby([p.pitcher, p.year]).mean()
PT["sweep"] = (p[p.is_brk].pfx_x.abs() * 12).groupby([p.pitcher, p.year]).mean()
PT = PT.reset_index()
# leave-this-batter-out K% and chase induced
k_py = pa.groupby(["pitcher", "year"]).K.agg(["sum", "size"])
k_pyb = pa.groupby(["pitcher", "year", "batter"]).K.agg(["sum", "size"])
oz = p[p.out_zone]
c_py = oz.groupby(["pitcher", "year"]).swing.agg(["sum", "size"])
c_pyb = oz.groupby(["pitcher", "year", "batter"]).swing.agg(["sum", "size"])


def loo(tot, own):
    j = own.join(tot, rsuffix="_t")
    return ((j["sum_t"] - j["sum"]) / (j["size_t"] - j["size"]).replace(0, np.nan)).rename(None)


pa = pa.merge(loo(k_py, k_pyb).rename("Kp").reset_index(), on=["pitcher", "year", "batter"], how="left")
pa = pa.merge(
    loo(c_py, c_pyb.astype(float)).rename("chasep").reset_index(), on=["pitcher", "year", "batter"], how="left"
)
pa = pa.merge(PT, on=["pitcher", "year"], how="left")
pa["sweep"] = pa.sweep.fillna(pa.sweep.median())
pa["velo"] = pa.velo.fillna(pa.velo.median())
pa["rise"] = pa.rise.fillna(pa.rise.median())
pa = pa[(pa.n_p >= 400)].copy()  # ~100 batters faced: stable profile

# ---------------- hitter traits fixed from 2024-25 (batter-side)
h = p[p.year.isin(["2024", "2025"])]
hb = h.groupby(["batter", "stand"])
HT = pd.DataFrame(
    {
        "n_b": hb.size(),
        "whiff_b": h[h.swing].groupby(["batter", "stand"]).whiff.mean(),
        "chase_b": h[h.out_zone].groupby(["batter", "stand"]).swing.mean(),
        "bs_b": h[h.swing].groupby(["batter", "stand"]).bat_speed.mean(),
    }
)
HT["K_b"] = pa[pa.year.isin(["2024", "2025"])].groupby(["batter", "stand"]).K.mean()
HT = HT[HT.n_b >= 600].reset_index()
bio = pd.read_csv("bio_all.csv")[["batter", "height_in"]]
HT = HT.merge(bio, on="batter", how="left").rename(columns={"height_in": "ht_b"})
pa = pa.merge(HT, on=["batter", "stand"])

# ---------------- setup (hitter-season)
s = pd.read_parquet("stance_adj.parquet")
s["width_f"] = s.width_adj.fillna(s.width)
ss = (
    s.groupby(["batter", "side", "year"])
    .agg(
        depth=("depth_adj", "mean"),
        offp=("off_plate_adj", "mean"),
        width=("width_f", "mean"),
        angle=("angle", "mean"),
        days=("depth", "size"),
    )
    .reset_index()
)
ss = ss[ss.days >= 20]
for c in ("depth", "offp", "width", "angle"):
    ss[c] = (ss[c] - ss.groupby(["side", "year"])[c].transform("mean")) / (
        3.0 if c != "angle" else 10.0
    )  # per 3 in / 10 deg
pa = pa.merge(ss.rename(columns={"side": "stand"}), on=["batter", "stand", "year"])
pa = pa.dropna(subset=["Kp", "chasep", "ext", "arm", "whiff_b", "chase_b", "bs_b", "K_b", "ht_b"]).reset_index(
    drop=True
)
pa["bsy"] = pa.batter.astype(str) + pa.stand + pa.year
pa["pyr"] = pa.pitcher.astype(str) + pa.year
pa["isR"] = (pa.stand == "R").astype(float)

PTR = ["velo", "brk", "off", "sweep", "rise", "ext", "arm", "zone", "Kp", "chasep"]
HTR = ["chase_b", "whiff_b", "K_b", "bs_b", "ht_b"]
tr = pa.year.isin(["2024", "2025"])
SCALE = {}
for c in PTR + HTR:  # standardise on TRAINING rows only
    mu, sd = pa.loc[tr, c].mean(), pa.loc[tr, c].std()
    SCALE[c] = (mu, sd)
    pa[c + "_z"] = (pa[c] - mu) / sd
MOD = {"same": pa.same}
for c in PTR:
    MOD[c] = pa[c + "_z"]
MOD["same_x_brk"] = pa.same * pa.brk_z
MOD["same_x_chase_b"] = pa.same * pa.chase_b_z
MOD["same_x_whiff_b"] = pa.same * pa.whiff_b_z
LEV = ["depth", "offp", "width"]
INT = {}
for k in LEV:
    for m, v in MOD.items():
        INT[f"{k}|{m}"] = pa[k] * v
CTL = {"same": pa.same}
for b in HTR:
    CTL[f"same*{b}"] = pa.same * pa[b + "_z"]
    for c in PTR:
        CTL[f"{b}*{c}"] = pa[b + "_z"] * pa[c + "_z"]
for c in PTR:
    CTL[f"same*{c}"] = pa.same * pa[c + "_z"]
    CTL[f"isR*{c}"] = pa.isR * pa[c + "_z"]
CTL["isR*same"] = pa.isR * pa.same
for m, v in MOD.items():
    CTL[f"angle|{m}"] = pa.angle * v
XI = pd.DataFrame(INT)
XC = pd.DataFrame(CTL)
print(
    f"PA sample: {len(pa):,} ({tr.sum():,} train 2024-25, {(~tr).sum():,} test 2026); hitters {pa.batter.nunique()}, "
    f"pitchers {pa.pitcher.nunique()}; interactions {XI.shape[1]}, controls {XC.shape[1]}"
)


def fit(idx, y, cols_int, extra=None, fes=("bsy", "pyr")):
    d = pa.loc[idx]
    ok = d[y].notna().to_numpy()
    d = d[ok]
    Xi = cols_int.loc[d.index] if extra is None else extra.loc[d.index]
    X = np.column_stack([d[y].to_numpy(), Xi.to_numpy(float), XC.loc[d.index].to_numpy(float)])
    codes = [pd.factorize(d[f])[0] for f in fes]
    M = demean(X, codes, iters=100, tol=1e-7)
    yy, Xa = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.pinv(Xa.T @ Xa)
    b = XtX_inv @ Xa.T @ yy
    u = yy - Xa @ b
    gcl = pd.factorize(d.batter)[0]
    G = gcl.max() + 1
    Xu = Xa * u[:, None]
    sums = np.vstack([np.bincount(gcl, weights=Xu[:, j], minlength=G) for j in range(Xa.shape[1])]).T
    n, kk = Xa.shape
    V = XtX_inv @ (sums.T @ sums) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - kk))
    names = list(Xi.columns) + list(XC.columns)
    return pd.Series(b, index=names), pd.DataFrame(V, index=names, columns=names), len(d)


OUT = {"K": ("strikeout %", "pp"), "woba": ("wOBA", "pts"), "rv": ("run value", "runs/100 PA")}
LAB = {
    "same": "same-hand pitcher",
    "velo": "fastball velo",
    "brk": "breaking-ball %",
    "off": "offspeed %",
    "sweep": "breaking sweep",
    "rise": "fastball rise",
    "ext": "extension",
    "arm": "arm angle",
    "zone": "zone %",
    "Kp": "pitcher K%",
    "chasep": "chase induced",
    "same_x_brk": "same-hand x breaking %",
    "same_x_chase_b": "same-hand x hitter chase",
    "same_x_whiff_b": "same-hand x hitter whiff",
}
res = {}
te_idx = pa.index[~tr]
for y, (lab, unit) in OUT.items():
    b, V, n = fit(pa.index[tr], y, XI)
    bi = b[XI.columns]
    Vi = V.loc[XI.columns, XI.columns]
    se = np.sqrt(np.diag(Vi))
    w = bi.values @ np.linalg.pinv(Vi.values) @ bi.values
    print(
        f"\n=== {lab} ({unit}) — TRAIN 2024-25, {n:,} PA. Change in the effect of +3 in per 1 SD of the pitcher trait"
    )
    for k in LEV:
        row = [
            f"{m} {bi[f'{k}|{m}']:+.2f}{'*' if abs(bi[f'{k}|{m}'] / se[list(XI.columns).index(f'{k}|{m}')]) >= 2 else ''}"
            for m in MOD
        ]
        print(f"   {k:6s} " + " ".join(row))
    print(
        f"   joint test (the pitcher changes nothing): chi2({len(bi)}) = {w:.1f}, p = {1 - chi2.cdf(w, len(bi)):.4f}   (* = |z| >= 2)"
    )
    # ---- validate on 2026: trained matchup index, total and per lever
    H = pd.DataFrame({"H_all": XI.loc[te_idx].to_numpy(float) @ bi.values}, index=te_idx)
    for k in LEV:
        cols = [c for c in XI.columns if c.startswith(k + "|")]
        H[f"H_{k}"] = XI.loc[te_idx, cols].to_numpy(float) @ bi[cols].values
    bv, Vv, nv = fit(te_idx, y, None, extra=H[["H_all"]])
    lam, lse = bv["H_all"], np.sqrt(Vv.loc["H_all", "H_all"])
    bk, Vk, _ = fit(te_idx, y, None, extra=H[[f"H_{k}" for k in LEV]])
    per = ", ".join(f"{k} {bk[f'H_{k}']:+.2f} ({np.sqrt(Vk.loc[f'H_{k}', f'H_{k}']):.2f})" for k in LEV)
    print(
        f"   VALIDATION 2026 ({nv:,} PA): lambda = {lam:+.2f} (SE {lse:.2f}, z vs 0 {lam/lse:+.1f}, z vs 1 {(lam-1)/lse:+.1f});  per lever: {per}"
    )
    b26, V26, _ = fit(te_idx, y, XI)
    res[y] = dict(
        b=bi, V=Vi, lam=lam, lse=lse, b26=b26[XI.columns], se26=np.sqrt(np.diag(V26.loc[XI.columns, XI.columns]))
    )
    for k in LEV:
        res[y]["lam_" + k], res[y]["lse_" + k] = bk[f"H_{k}"], np.sqrt(Vk.loc[f"H_{k}", f"H_{k}"])

# ---------------- mechanism: chase on out-of-zone pitches (pitch level), same design, count FE added
print("\n=== MECHANISM: chase on out-of-zone pitches (pp), per +3 in, per 1 SD — key terms, train 2024-25 vs 2026")
ozp = p[p.out_zone][["pa", "swing", "balls", "strikes"]].copy()
ozp["chase"] = ozp.swing.astype(float) * 100
ozp["cnt"] = ozp.balls.astype(str) + ozp.strikes.astype(str)
keep = pa[["pa", "bsy", "pyr", "batter", "year"]].reset_index().rename(columns={"index": "pa_row"})
ozp = ozp.merge(keep, on="pa")
XIo, XCo = XI.loc[ozp.pa_row].reset_index(drop=True), XC.loc[ozp.pa_row].reset_index(drop=True)


def fit_o(mask, Xi):
    d = ozp[mask]
    X = np.column_stack([d.chase.to_numpy(), Xi[mask.to_numpy()].to_numpy(float), XCo[mask.to_numpy()].to_numpy(float)])
    M = demean(X, [pd.factorize(d[f])[0] for f in ("bsy", "pyr", "cnt")], iters=100, tol=1e-7)
    yy, Xa = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.pinv(Xa.T @ Xa)
    b = XtX_inv @ Xa.T @ yy
    u = yy - Xa @ b
    gcl = pd.factorize(d.batter)[0]
    G = gcl.max() + 1
    Xu = Xa * u[:, None]
    sums = np.vstack([np.bincount(gcl, weights=Xu[:, j], minlength=G) for j in range(Xa.shape[1])]).T
    V = XtX_inv @ (sums.T @ sums) @ XtX_inv * (G / (G - 1))
    return pd.Series(b[: Xi.shape[1]], index=Xi.columns), np.sqrt(np.diag(V)[: Xi.shape[1]]), len(d)


m_tr = ozp.year.isin(["2024", "2025"])
bo, so, no = fit_o(m_tr, XIo)
bo6, so6, no6 = fit_o(~m_tr, XIo)
for key in [
    "depth|same",
    "depth|brk",
    "depth|same_x_brk",
    "depth|same_x_chase_b",
    "depth|Kp",
    "depth|chasep",
    "depth|velo",
    "width|same",
    "width|brk",
    "width|same_x_chase_b",
    "offp|same",
    "offp|brk",
    "offp|sweep",
]:
    i = list(XIo.columns).index(key)
    print(
        f"   {key:22s} train {bo[key]:+6.2f} (z {bo[key]/so[i]:+.1f})   2026 {bo6[key]:+6.2f} (z {bo6[key]/so6[i]:+.1f})"
    )
res["chase"] = dict(b=bo, se=so, b26=bo6, se26=so6)
pickle.dump(dict(res=res, SCALE=SCALE, MOD=list(MOD), LEV=LEV, PTR=PTR, HTR=HTR), open("matchup_model.pkl", "wb"))
pa[
    ["pa", "batter", "pitcher", "stand", "p_throws", "same", "year", "K", "woba", "rv", "depth", "offp", "width"]
    + PTR
    + HTR
].to_parquet("matchup_pa.parquet")
