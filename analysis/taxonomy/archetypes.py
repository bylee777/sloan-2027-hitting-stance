import sys, numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture

H, Z, F, bic = pd.read_pickle("taxonomy/_stage.pkl")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
best = min(
    (GaussianMixture(K, covariance_type="full", n_init=5, random_state=s).fit(Z) for s in range(5)),
    key=lambda m: m.bic(Z),
)
post = best.predict_proba(Z)
H["arch"] = post.argmax(1)
H["conf"] = post.max(1)
order = H.groupby("arch").width_in.median().sort_values().index  # label archetypes by setup width
H["arch"] = H.arch.map({a: i + 1 for i, a in enumerate(order)})
print(f"k={K}: mean assignment confidence {H.conf.mean():.2f}; share >= 0.8: {(H.conf >= .8).mean():.0%}")
D = [
    "depth_in",
    "off_plate_in",
    "width_in",
    "stance_angle",
    "front_turn0",
    "stride_len_in",
    "stride_dir",
    "width_land_in",
    "front_turn2",
    "back_foot_move_in",
    "contact_vs_plate_in",
]
S = ["bat_speed", "swing_len", "attack_angle", "attack_dir", "path_tilt", "height_in"]
O = ["K", "BB", "chase", "whiff", "zone_contact", "ev", "sweet", "xwoba_con", "xwoba", "platoon_gap"]
pd.set_option("display.width", 250)
print("\nSETUP & MOVE (medians; in / deg)")
print(H.groupby("arch")[D].median().round(1).T.to_string())
print("\nSWING (medians)")
print(H.groupby("arch")[S].median().round(2).T.to_string())
t = H.groupby("arch")[O].median()
t[["K", "BB", "chase", "whiff", "zone_contact", "sweet"]] *= 100
print("\nOUTCOMES (medians; rates in %)")
print(t.round(3).T.to_string())
print("\nshare of hitter-seasons:", H.arch.value_counts(normalize=True).sort_index().round(2).to_dict())
print("\nclearest examples (highest assignment confidence, >= 450 PA):")
for a, gdf in H[H.PA >= 450].sort_values("conf", ascending=False).groupby("arch"):
    print(f"  {a}: " + "; ".join(f"{r['name']} {r.year}" for _, r in gdf.head(6).iterrows()))
nx = H[["batter", "side", "year", "arch"]].assign(year=lambda d: (d.year.astype(int) - 1).astype(str))
pair = H.merge(nx, on=["batter", "side", "year"], suffixes=("", "_next"))
print(
    f"\nyear-to-year: same archetype next season {(pair.arch == pair.arch_next).mean():.0%} (n {len(pair)}); chance level {sum((H.arch.value_counts(normalize=True))**2):.0%}"
)
print(pd.crosstab(pair.arch, pair.arch_next, normalize="index").round(2).to_string())
for nm in ("Crow-Armstrong", "Jung, Josh", "Altuve", "Burger", "Judge", "Soto"):
    r = H[H.name.str.contains(nm)][
        ["name", "year", "arch", "conf", "depth_in", "width_in", "stride_len_in", "stance_angle"]
    ]
    if len(r):
        print("\n" + r.round(2).to_string(index=False))
H.to_parquet(f"taxonomy/hitters_k{K}.parquet", index=False)
# robustness (paper 5.1): cluster separation, and agreement across methods and across the season the clusters are fit on
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score as ari

Z = np.asarray(Z)
gm = lambda X: GaussianMixture(K, covariance_type="full", n_init=5, random_state=0).fit(X)
lab = {
    "gmm": gm(Z).predict(Z),
    "kmeans": KMeans(K, n_init=10, random_state=0).fit_predict(Z),
    "ward": AgglomerativeClustering(K, linkage="ward").fit_predict(Z),
}
print("\nROBUSTNESS\n  silhouette: " + ", ".join(f"{m} {silhouette_score(Z, l):.3f}" for m, l in lab.items()))
print(
    "  agreement across methods (adjusted Rand): "
    + ", ".join(
        f"{a}-{b} {ari(lab[a], lab[b]):.3f}" for a, b in (("gmm", "kmeans"), ("gmm", "ward"), ("kmeans", "ward"))
    )
)
fits = {y: gm(Z[(H.year == y).values]) for y in sorted(H.year.unique())}
print(
    "  agreement across seasons (fit on each season, label all; adjusted Rand): "
    + ", ".join(
        f"{a}-{b} {ari(fits[a].predict(Z), fits[b].predict(Z)):.3f}"
        for i, a in enumerate(fits)
        for b in list(fits)[i + 1 :]
    )
)
