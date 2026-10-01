"""Detect PCA-style sustained mid-season depth moves.

1. Park adjustment: depth ~ batter-side-season FE + park(home team)-season FE; subtract park effect.
2. For each batter-side-season with >= 60 stance games: at every split point k, diff = mean(adj depth
   over next K games) - mean(previous K games), K = 15. Event = the split with the largest |diff|, if
   |diff| >= THRESH and the move persists (median of next 30 games vs previous 30 also >= THRESH - 1).
   Direction: BACK (deeper, +) or UP (toward pitcher, -).
Also detects equally-sized WIDTH / ANGLE changes without a depth move (tinkering placebos).
"""

import numpy as np, pandas as pd, warnings
from stance_lib import stance, pitches
from fe import demean

warnings.filterwarnings("ignore")
K, THRESH = 15, 6.0


def detect(g, col, thresh):
    x = g[col + "_adj"].to_numpy()
    n = len(x)
    if n < 60:
        return None
    best = None
    for k in range(K, n - K + 1):
        d = x[k : k + K].mean() - x[k - K : k].mean()
        if best is None or abs(d) > abs(best[1]):
            best = (k, d)
    k, d = best
    lo, hi = max(0, k - 30), min(n, k + 30)
    pers = np.median(x[k:hi]) - np.median(x[lo:k])
    if abs(d) >= thresh and np.sign(pers) == np.sign(d) and abs(pers) >= thresh - 1:
        return dict(
            bsy=g.bsy.iloc[0],
            batter=g.batter.iloc[0],
            side=g.side.iloc[0],
            year=g.year.iloc[0],
            name=g.name.iloc[0],
            event_date=g.game_date.iloc[k],
            k=k,
            n_games=n,
            move=d,
            persist=pers,
        )
    return None


if __name__ == "__main__":
    s = stance()
    s["bs"] = s.batter.astype(str) + s.side
    s["bsy"] = s.bs + s.year
    # home team for each batter-date (park) from the pitch data
    p = pitches()
    home = p.groupby(["batter", "game_date"]).home_team.first().reset_index()
    s = s.merge(home, on=["batter", "game_date"], how="left").dropna(subset=["home_team"])
    s["park"] = s.home_team + s.year
    for col in ("depth", "width", "angle", "off_plate"):
        codes = [pd.factorize(s.bsy)[0], pd.factorize(s.park)[0]]
        resid = demean(s[[col]].to_numpy(), codes)[:, 0]
        batter_mean = s.groupby("bsy")[col].transform("mean")
        within = s[col] - batter_mean
        s[col + "_park"] = within - resid  # park component of the within-batter variation
        s[col + "_adj"] = s[col] - s[col + "_park"]
    pk = s.groupby("park").depth_park.mean()
    print(
        f"park effect on measured depth: SD across park-seasons {pk.std():.2f} in, range {pk.min():+.2f}..{pk.max():+.2f}"
    )

    s = s.sort_values(["bsy", "game_date"])
    ev = []
    for col, thr in (("depth", THRESH), ("width", THRESH), ("angle", 10.0)):
        for _, g in s.groupby("bsy"):
            r = detect(g, col, thr)
            if r:
                r["kind"] = col
                ev.append(r)
    ev = pd.DataFrame(ev)
    ev["dir"] = np.where(ev.move > 0, "+", "-")
    print("\nevents by kind/direction/season:")
    print(ev.groupby(["kind", "dir", "year"]).size().unstack(fill_value=0).to_string())
    for thr in (4, 5, 6, 8, 10):
        n = sum(1 for _, g in s.groupby("bsy") if (r := detect(g, "depth", thr)) and r["move"] > 0)
        print(f"  depth BACK events at >= {thr} in: {n}")
    print(
        "\nPCA:",
        ev[ev.batter == 691718][["year", "kind", "event_date", "move", "persist"]].round(1).to_string(index=False),
    )
    print("\nlargest depth moves:")
    print(
        ev[ev.kind == "depth"]
        .reindex(ev[ev.kind == "depth"].move.abs().sort_values(ascending=False).index)
        .head(15)[["name", "side", "year", "event_date", "move", "persist", "n_games"]]
        .round(1)
        .to_string(index=False)
    )
    ev.to_csv("events.csv", index=False)
    s.to_parquet("stance_adj.parquet", index=False)
