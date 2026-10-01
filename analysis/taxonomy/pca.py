"""Continuous stance fingerprint (paper Section 4.1 / 5.1): principal components of the 11 start + move features
(taxonomy/build.py: clipped at the 1st/99th percentile and robust-scaled), z-scored, 4 components.
Writes taxonomy/hitters_pca.parquet (hitter-seasons + PC1-PC4) and taxonomy/pca_loadings.csv; prints the variance share and
each component's year-to-year correlation for the same hitter (consecutive seasons)."""

import numpy as np, pandas as pd
from sklearn.decomposition import PCA

H, Z, F, bic = pd.read_pickle("taxonomy/_stage.pkl")
M = (Z - Z.mean(0)) / Z.std(0)
pc = PCA(4).fit(M)
S = pc.transform(M)
out = H.copy()
for k in range(4):
    out[f"PC{k + 1}"] = S[:, k]
out.to_parquet("taxonomy/hitters_pca.parquet", index=False)
pd.DataFrame(np.round(pc.components_.T, 2), index=F, columns=[f"PC{k + 1}" for k in range(4)]).to_csv(
    "taxonomy/pca_loadings.csv"
)
print(
    f"{len(out)} hitter-seasons; variance explained by 4 components: {pc.explained_variance_ratio_.sum():.1%} "
    f"({', '.join(f'{v:.1%}' for v in pc.explained_variance_ratio_)})"
)
o = out.assign(y=out.year.astype(int))
nxt = o.merge(o.assign(y=o.y - 1), on=["batter", "side", "y"], suffixes=("", "_next"))
print(
    f"year-to-year r of each hitter's score ({len(nxt)} consecutive-season pairs): "
    + ", ".join(f"PC{k + 1} {np.corrcoef(nxt[f'PC{k + 1}'], nxt[f'PC{k + 1}_next'])[0, 1]:.2f}" for k in range(4))
)
