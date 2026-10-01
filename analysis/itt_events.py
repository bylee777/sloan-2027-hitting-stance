"""ITT re-detection of sustained stance changes (persistence-selection check for the Sloan paper, 2026-09-28).

Original rule (events.detect): per hitter-season and lever, take the split k with the largest |mean(next 15 games) -
mean(previous 15)|; keep it if |diff| >= thresh AND the move persists (median of next 30 games minus median of previous
30 has the same sign and |.| >= thresh - 1). The persistence window sits inside the 60-day outcome window, so changes
that were abandoned (plausibly because they went badly) never become events.

ITT rule here: same scan, same max split, same threshold on the 15-game diff, NO persistence requirement.
Each ITT event is labelled persisted (passes the original rule) or reverted. Inputs replicate the original detectors
exactly: depth = stance_adj.depth_adj (event_study.py, 4 in); width = width.py's refilled + park-adjusted width (4 in);
angle = detect_angle.py's angle_adj.fillna(angle) (10 deg). Writes itt_events.csv (all levers)."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from fe import demean

K = 15


def scan(g, col, thresh):
    """events.detect's scan, returning the max split plus both persistence verdicts."""
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
    if not abs(d) >= thresh:
        return None
    lo, hi = max(0, k - 30), min(n, k + 30)
    pers = np.median(x[k:hi]) - np.median(x[lo:k])
    persisted = bool(np.sign(pers) == np.sign(d) and abs(pers) >= thresh - 1)
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
        persisted=persisted,
    )


def load_stance():
    s = pd.read_parquet("stance_adj.parquet").sort_values(["bsy", "game_date"])
    # width exactly as width.py
    s["width"] = s.width.fillna(s.groupby("bsy").width.transform("mean"))
    codes = [pd.factorize(s.bsy)[0], pd.factorize(s.park)[0]]
    resid = demean(s[["width"]].to_numpy(), codes)[:, 0]
    s["width_adj"] = s.width - ((s.width - s.groupby("bsy").width.transform("mean")) - resid)
    # angle exactly as detect_angle.py (angle_adj is all-NaN in the parquet -> raw angle)
    s["angle_adj"] = s.angle_adj.fillna(s.angle)
    return s


LEVERS = (("depth", 4.0, "BACK", "UP"), ("width", 4.0, "WIDER", "NARROWER"), ("angle", 10.0, "CLOSED", "OPENED"))

if __name__ == "__main__":
    s = load_stance()
    out = []
    for lever, thr, pos, neg in LEVERS:
        rows = [r for _, g in s.groupby("bsy") if (r := scan(g, lever, thr))]
        ev = pd.DataFrame(rows)
        ev["lever"] = lever
        ev["dir"] = np.where(ev.move > 0, pos, neg)
        out.append(ev)
    ev = pd.concat(out, ignore_index=True)
    # reproduction check against the original event files
    orig = {
        "depth": pd.read_csv("events_thr4.csv"),
        "width": pd.read_csv("width_events.csv"),
        "angle": pd.read_csv("angle_events.csv"),
    }
    ok = True
    for lever, o in orig.items():
        p = ev[(ev.lever == lever) & ev.persisted]
        a = set(zip(o.bsy, o.event_date))
        b = set(zip(p.bsy, p.event_date))
        match = a == b
        ok &= match
        print(f"{lever:5s} original file {len(o):3d} | ITT-persisted {len(p):3d} | identical (bsy, date) sets: {match}")
    print("REPRODUCTION", "OK" if ok else "FAILED")
    print("\nITT events by lever / direction / persisted:")
    print(ev.groupby(["lever", "dir", "persisted"]).size().unstack(fill_value=0).to_string())
    ev.to_csv("itt_events.csv", index=False)
