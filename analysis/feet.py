"""Foot-level stance geometry from Savant's batting-stance visual (vizData embedded in the page).

Coordinates (ft): x = lateral (sign flips with batter side; normalised here so + = away from the
plate), y = toward the pitcher. Three phases per foot: 0 = setup, 1 = load/mid, 2 = stride landed
(inferred from the coordinates: phase 2 front foot is ~2 ft toward the pitcher).
Front foot = the pitcher-side foot (R for LHB, L for RHB).
Foot 'turn' = heel->toe direction: 0 deg = toes pointing straight at the plate; + = toes turned
toward the pitcher (OPEN foot); - = toward the catcher.
Stance angle (Savant) = line between the feet; - = open (front foot farther off the plate).
"""

import json, subprocess, time
import numpy as np, pandas as pd

B = "https://baseballsavant.mlb.com/visuals/batting-stance"


def fetch(season, d0, d1, path):
    q = f"seasonStart={season}&seasonEnd={season}&dateStart={d0}&dateEnd={d1}&minSwings=1&minGroupSwings=1"
    subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", f"{B}?{q}", "--create-dirs", "-o", path], check=True)
    s = open(path, encoding="utf-8", errors="ignore").read()
    i = s.find("const vizData = ")
    j = s.find("];", i)
    return json.loads(s[i + len("const vizData = ") : j + 1])


def geometry(r):
    side = r["side"]
    sgn = 1 if side == "L" else -1  # + x = away from plate for either side
    front, back = ("r", "l") if side == "L" else ("l", "r")

    def pt(foot, part, ph):
        return np.array([sgn * r[f"avg_{foot}{part}_x{ph}"], r[f"avg_{foot}{part}_y{ph}"]])

    out = dict(
        depth_in=r["avg_batter_y_position"],
        off_plate_in=r["avg_batter_x_position"],
        width_in=r["avg_foot_sep0"],
        stance_angle=r["avg_foot_angle0"],
        contact_vs_plate_in=r["avg_intercept_y_in_vs_plate"],
    )
    for name, foot in (("front", front), ("back", back)):
        for ph in (0, 2):
            heel = pt(foot, "heel", ph)
            toe = (pt(foot, "bigtoe", ph) + pt(foot, "smalltoe", ph)) / 2
            v = toe - heel  # toes point toward -x (the plate) when square
            out[f"{name}_turn{ph}"] = np.degrees(np.arctan2(v[1], -v[0]))
    f0 = (pt(front, "heel", 0) + pt(front, "bigtoe", 0)) / 2
    f2 = (pt(front, "heel", 2) + pt(front, "bigtoe", 2)) / 2
    d = f2 - f0
    out["stride_len_in"] = np.hypot(*d) * 12
    out["stride_dir"] = np.degrees(np.arctan2(d[0], d[1]))  # + = stride lands AWAY from plate (open step)
    b0 = (pt(back, "heel", 0) + pt(back, "bigtoe", 0)) / 2
    out["landed_angle"] = np.degrees(
        np.arctan2(-(f2 - b0)[0], (f2 - b0)[1])
    )  # line back foot -> landed front foot; - = open
    return out


if __name__ == "__main__":
    rows = []
    for tag, season, d0, d1 in (
        ("2023 season", 2023, "2023-03-01", "2023-11-01"),
        ("2024 pre-move (Apr 1-Jul 26)", 2024, "2024-04-01", "2024-07-26"),
        ("2024 post-move (Jul 29-Sep 30)", 2024, "2024-07-29", "2024-09-30"),
        ("2025 season", 2025, "2025-03-01", "2025-11-01"),
        ("2026 season", 2026, "2026-03-01", "2026-11-01"),
    ):
        d = fetch(season, d0, d1, f"viz/pca_{season}_{d0}.html")
        r = [x for x in d if x["id"] == 691718]
        if r:
            rows.append(dict(window=tag, **geometry(r[0])))
        time.sleep(1.2)
    t = pd.DataFrame(rows).set_index("window").T
    pd.set_option("display.width", 220)
    print("Pete Crow-Armstrong (LHB) stance geometry\n")
    print(t.round(1).to_string())
