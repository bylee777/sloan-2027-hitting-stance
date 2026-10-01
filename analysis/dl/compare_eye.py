"""Paired comparison: perception network with eye z = 0.81 x height (original) vs posture-based eye height (EYE=posture).
Reads robust_<cfg>.txt / robust_<cfg>_eye<L>.txt and the recommender grids; writes compare_eye.txt."""

import re, subprocess, sys, pandas as pd

CFGS = ["s0_2024to2025", "s1_2024to2025", "s2_2024to2025", "s0_2025to2024"]


def parse(path):
    t = open(path).read()
    g = lambda pat: float(re.search(pat, t).group(1)) if re.search(pat, t) else float("nan")
    return dict(
        rv_R2=g(r"rv: test R2 full ([\d.]+)"),
        chase_ll=g(r"chase: test logloss full ([\d.]+)"),
        whiff_ll=g(r"whiff: test logloss full ([\d.]+)"),
        mech_away=g(r"whiff on AWAY ([+-][\d.]+)"),
        mech_in=g(r"IN ([+-][\d.]+) pp \(vs middle\)"),
        default=g(r"SUMMARY \S+ default ([+-][\d.]+)"),
        adjust=g(r"adjust ([+-][\d.]+)"),
        upper=g(r"UPPER BOUND on per-pitcher value: ([+-][\d.]+)"),
        deeper=g(r"deeper_default_share ([\d.]+)"),
    )


def two(grid):
    out = subprocess.run([sys.executable, "two_position.py", grid], capture_output=True, text=True).stdout
    m = re.search(r"two-position value ([+-][\d.]+)", out)
    return float(m.group(1)) if m else float("nan")


rows = []
for c in CFGS:
    for lab, suf in (("original", ""), ("posture eye", "_eye1")):
        try:
            d = parse(f"robust_{c}{suf}.txt")
            d["two_pos"] = two(f"recommender_grid_{c}{suf}.parquet")
            rows.append(dict(cfg=c, eye=lab, **d))
        except FileNotFoundError:
            pass
try:
    d = parse("robust_s0_2024to2025_eye2.txt")
    d["two_pos"] = two("recommender_grid_s0_2024to2025_eye2.parquet")
    rows.append(dict(cfg="s0_2024to2025", eye="posture eye, lean 2", **d))
except FileNotFoundError:
    pass
T = pd.DataFrame(rows)
with open("compare_eye.txt", "w") as f:
    f.write(T.round(4).to_string(index=False) + "\n\n")
    P = T[T.eye.isin(["original", "posture eye"])].pivot(index="cfg", columns="eye")
    for m in ["rv_R2", "chase_ll", "whiff_ll", "default", "adjust", "upper", "two_pos"]:
        if ("posture eye",) and (m, "posture eye") in P.columns:
            d = (P[(m, "posture eye")] - P[(m, "original")]).dropna()
            f.write(
                f"{m:9s} posture minus original, mean over {len(d)} runs: {d.mean():+.4f} (range {d.min():+.4f} to {d.max():+.4f})\n"
            )
    for c in CFGS:
        try:
            eye_lines = [l.strip() for l in open(f"robust_{c}_eye1.txt") if l.startswith("eye ")]
            f.write(f"\n{c} posture-eye counterfactual:\n  " + "\n  ".join(eye_lines))
        except FileNotFoundError:
            pass
print(open("compare_eye.txt").read())
