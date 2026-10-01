"""Personalized stance model v2 + out-of-sample validation.
Traits (fixed per hitter, from 2024-25 only): height, weight, bat speed, chase rate, knee height beyond height (leg proxy).
TRAIN (2024-25 monthly panel): double ML, lever effects (+3 in depth / width / off-plate) x traits.
VALIDATE (2026, never seen): residualise 2026 outcomes and levers (cross-fitted), then test whether the TRAINED
personalized part predicts 2026 responses: Y_res ~ sum_k D_res_k [2026 average effect] + lambda * sum_k D_res_k x
(CATE_k - base_k) [trained heterogeneity]. lambda ~ 1 = personalization real & calibrated; ~ 0 = noise."""

import glob, numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

D = pd.read_parquet("front_foot_panel.parquet")
# ---- fixed traits from 2024-25
k = []
for y in ("2024", "2025"):
    d = pd.concat([pd.read_parquet(f, columns=["batter", "stand", "sz_bot"]) for f in glob.glob(f"raw/{y}*.parquet")])
    m = d.groupby(["batter", "stand"]).agg(n=("sz_bot", "size"), knee=("sz_bot", "median")).reset_index()
    k.append(m[m.n >= 300])
K = pd.concat(k).rename(columns={"stand": "side"}).groupby(["batter", "side"]).knee.mean().reset_index()
bio = pd.read_csv("bio_all.csv")[["batter", "height_in", "weight_lb"]]
K = K.merge(bio, on="batter")
K["knee_in"] = K.knee * 12
K["knee_resid"] = K.knee_in - np.polyval(np.polyfit(K.height_in, K.knee_in, 1), K.height_in)
Hs = pd.read_parquet("taxonomy/hitters_pca.parquet")
T = (
    Hs[Hs.year.isin(["2024", "2025"])]
    .groupby(["batter", "side"])
    .agg(bs=("bat_speed", "mean"), ch=("chase", "mean"))
    .reset_index()
)
TRAITS = T.merge(K[["batter", "side", "knee_resid", "weight_lb"]], on=["batter", "side"])
D = D.drop(columns=[c for c in ("weight_lb",) if c in D.columns]).merge(TRAITS, on=["batter", "side"])
D = D.dropna(subset=["height_in", "depth_in", "width_in", "off_plate_in"]).reset_index(drop=True)
TR = {"height": "height_in", "weight": "weight_lb", "batspeed": "bs", "chase": "ch", "legproxy": "knee_resid"}
tr_mask = D.year.isin(["2024", "2025"])
for z, c in TR.items():
    mu, sd = D.loc[tr_mask, c].mean(), D.loc[tr_mask, c].std()
    D[z + "_z"] = (D[c] - mu) / sd
LEV = {"depth": "depth_in", "width": "width_in", "offplate": "off_plate_in"}
for kk, c in LEV.items():
    D[kk] = D[c] / 3.0
for c in list(LEV.values()) + ["front_turn0", "front_turn2", "stance_angle", "stride_len_in"]:
    D[c + "_hm2"] = D.groupby("bsy")[c].transform("mean")
XC = [
    "stance_angle",
    "stride_len_in",
    "front_turn0",
    "front_turn2",
    "front_turn0_hm2",
    "front_turn2_hm2",
    "stance_angle_hm2",
    "stride_len_in_hm2",
    "depth_in_hm2",
    "width_in_hm2",
    "off_plate_in_hm2",
    "K_lag",
    "chase_lag",
    "whiff_lag",
    "xwoba_lag",
    "K_hm",
    "chase_hm",
    "whiff_hm",
    "xwoba_hm",
    "height_in",
    "weight_lb",
    "bs",
    "ch",
    "knee_resid",
    "mo",
    "yr",
    "isR",
]


def residualize(d, y):
    X = d[XC].to_numpy(float)
    targets = [y] + list(LEV)
    r = {t: np.zeros(len(d)) for t in targets}
    for tr, te in GroupKFold(5).split(X, groups=d.batter):
        for t in targets:
            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, random_state=0
            )
            m.fit(X[tr], d[t].to_numpy()[tr])
            r[t][te] = d[t].to_numpy()[te] - m.predict(X[te])
    return r


print(
    f"traits available for {D.batter.nunique()} hitters; train months {tr_mask.sum():,} (2024-25), test months {(~tr_mask).sum():,} (2026)"
)
OUT = {"xwoba": "xwOBA (pts)", "K": "strikeout % (pp)"}
for y, lab in OUT.items():
    tr = D[tr_mask].dropna(subset=[y]).reset_index(drop=True)
    r = residualize(tr, y)
    cols, names = [], []
    for kk in LEV:
        cols.append(r[kk])
        names.append(kk)
        for z in TR:
            cols.append(r[kk] * tr[z + "_z"].to_numpy())
            names.append(f"{kk}x{z}")
    fit = sm.OLS(r[y], np.column_stack(cols)).fit(cov_type="cluster", cov_kwds={"groups": tr.batter.to_numpy()})
    b = pd.Series(fit.params, index=names)
    V = pd.DataFrame(fit.cov_params(), index=names, columns=names)
    se = np.sqrt(np.diag(V))
    print(f"\n=== {lab}: TRAINED on 2024-25 (effect of +3 in; x = change per 1 SD of trait)")
    for kk in LEV:
        row = f"   {kk:9s} base {b[kk]:+6.2f} (z {b[kk]/se[names.index(kk)]:+.1f})"
        for z in TR:
            nm = f"{kk}x{z}"
            row += f" | {z} {b[nm]:+5.2f} (z {b[nm]/se[names.index(nm)]:+.1f})"
        print(row)
    idx = [i for i, n in enumerate(names) if "x" in n and n not in LEV]
    Rm = np.zeros((len(idx), len(b)))
    Rm[np.arange(len(idx)), idx] = 1
    w = (Rm @ b.values) @ np.linalg.inv(Rm @ V.values @ Rm.T) @ (Rm @ b.values)
    print(f"   joint test (traits change nothing): chi2({len(idx)}) = {w:.1f}, p = {1 - chi2.cdf(w, len(idx)):.3f}")
    # ---- validate on 2026
    te = D[~tr_mask].dropna(subset=[y]).reset_index(drop=True)
    rt = residualize(te, y)
    het = np.zeros(len(te))
    A = []
    for kk in LEV:
        cate_dev = sum(b[f"{kk}x{z}"] * te[z + "_z"].to_numpy() for z in TR)  # trained personalized part
        het += rt[kk] * cate_dev
        A.append(rt[kk])
    Xv = np.column_stack(A + [het])
    fv = sm.OLS(rt[y], Xv).fit(cov_type="cluster", cov_kwds={"groups": te.batter.to_numpy()})
    lam, lse = fv.params[-1], fv.bse[-1]
    print(
        f"   VALIDATION on 2026 ({len(te):,} hitter-months, {te.batter.nunique()} hitters): personalization coefficient lambda = {lam:+.2f} "
        f"(SE {lse:.2f}, z {lam/lse:+.1f})   [1 = real & calibrated, 0 = noise]"
    )
    print(
        "   2026 average effects (+3 in): "
        + ", ".join(f"{kk} {fv.params[i]:+.2f} (z {fv.params[i]/fv.bse[i]:+.1f})" for i, kk in enumerate(LEV))
    )
    b.to_csv(f"optimal_stance_v2_{y}_coefs.csv")
