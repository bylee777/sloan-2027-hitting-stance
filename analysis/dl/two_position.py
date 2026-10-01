"""Two-position test from the recommender grids (paper §5.4: 'two hand-specific setups add 0.001-0.008 runs per 100 pitches').
For each hitter: pick the best grid spot on half A separately for same-hand and opposite-hand pitchers, score each pitcher on half B with
the spot for its hand group, and compare with the hitter's single best default spot (also picked on half A). Value = two-position gain
minus default gain, runs per 100 pitches, averaged over each hitter's pitcher-sides and then over hitters.
Usage: python two_position.py recommender_grid_<TAG>.parquet [...]"""

import sys, numpy as np, pandas as pd

for path in sys.argv[1:]:
    R = pd.read_parquet(path)
    G = [c for c in R.columns if c.startswith("rv_")]
    R["same"] = (R.p_throws == R.bsy.str[-5]).astype(int)
    A, B = R[R.half == 0].set_index(["bsy", "pss"]), R[R.half == 1].set_index(["bsy", "pss"])
    A, B = A.align(B, join="inner", axis=0)
    default = A[G].groupby(level="bsy").mean().idxmax(axis=1)
    by_hand = A.assign(same=A["same"]).groupby([A.index.get_level_values("bsy"), "same"])[G].mean().idxmax(axis=1)
    cur = B["rv_0_0"]
    two = pd.Series([B.loc[k, by_hand[(k[0], B.loc[k, "same"])]] for k in B.index], index=B.index) - cur
    dfl = pd.Series([B.loc[k, default[k[0]]] for k in B.index], index=B.index) - cur
    per_h = pd.DataFrame({"two": two.groupby(level="bsy").mean(), "default": dfl.groupby(level="bsy").mean()})
    per_h["value"] = per_h.two - per_h.default
    diff_spots = (by_hand.unstack().nunique(axis=1) > 1).mean()
    print(
        f"{path.split('recommender_grid_')[-1].replace('.parquet', ''):22s} two-position value {per_h.value.mean():+.4f} runs/100 "
        f"(default gain {per_h.default.mean():+.3f}); hitters whose two spots differ {diff_spots:.0%}"
    )
