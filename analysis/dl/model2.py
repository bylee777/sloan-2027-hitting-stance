"""v2 positioning model + recommender (train 2024, evaluate/recommend 2025; pre-ABS).
Networks (MLP 3x128 + batter embedding): RUN VALUE per pitch (MSE), CHASE (OOZ swing), WHIFF (per swing).
Checks: test fit vs zone+count-only net; lateral mechanism vs the measured per-ft pattern (away whiff -3.35,
in whiff +3.47). Recommender: 2025 regulars x pitchers (>= 300 pitches to that batter side), grid depth
{-6,0,+6} in x distance-off-plate {-3,0,+3} in; best spot chosen on half A of each pitcher's pitches,
scored on half B. Value of per-pitcher adjusting = per-pitcher best - hitter's single best default."""

import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, ".")
from features2 import build2, F2, EYE_MODE, EYE_LEAN
from train import Net, Scaler
import os, json

HP_FILE = os.environ.get("HP_FILE", "")
HP = json.load(open(HP_FILE)) if HP_FILE else {}  # Optuna-tuned settings per task (tune_model2.py)


class NetFlex(nn.Module):
    """Net with tunable width/depth/dropout/embedding; depth=3, drop=0, emb=8, h=128 reproduces train.Net exactly."""

    def __init__(self, n_in, n_bat, emb=8, h=128, depth=3, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(n_bat + 1, emb)
        nn.init.normal_(self.emb.weight, 0, 0.01)
        layers, d = [], n_in + emb
        for w in [h] * (depth - 1) + [max(h // 2, 16)]:
            layers += [nn.Linear(d, w), nn.SiLU()] + ([nn.Dropout(drop)] if drop > 0 else [])
            d = w
        self.f = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x, b):
        return self.f(torch.cat([x, self.emb(b)], 1)).squeeze(1)


SEED = int(os.environ.get("SEED", "0"))
TRAIN, TEST = os.environ.get("TRAIN", "2024"), os.environ.get("TEST", "2025")
TAG = f"s{SEED}_{TRAIN}to{TEST}" + ("" if EYE_MODE == "height" else f"_eye{EYE_LEAN:g}") + ("_opt" if HP_FILE else "")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)
t0 = time.time()
df = pd.read_parquet("pitches_feat2.parquet")
df = df[df.year.isin(["2024", "2025"])].reset_index(drop=True)
fb = pd.read_parquet("fb_ref.parquet")
ids = {b: i + 1 for i, b in enumerate(df.batter.unique())}
df["bid"] = df.batter.map(ids).astype(np.int64)
if (
    EYE_MODE != "height"
):  # stored features use eye z = 0.81 x height -> rebuild the view/contact features with posture-based eye height
    f_eye = build2(df, df.depth.to_numpy(float), df.off_plate.to_numpy(float), fb)
    df[F2] = f_eye[F2].to_numpy()
    print(f"EYE={EYE_MODE} lean {EYE_LEAN:g}: features rebuilt with posture-based eye height", flush=True)


def design(d, feats, sc=None):
    X = np.column_stack(
        [
            d[feats].to_numpy(float),
            np.eye(4)[d.balls.clip(0, 3).astype(int)],
            np.eye(3)[d.strikes.clip(0, 2).astype(int)],
            d[["same"]].to_numpy(float),
        ]
    )
    sc = sc or Scaler().fit(X)
    return sc(X), sc


def fit(X, b, y, kind, epochs=10, bs=4096, lr=2e-3, task=None):
    va = np.random.rand(len(y)) < 0.05
    hp = HP.get(task) if task else None
    if hp:
        epochs, bs, lr = 15, hp["bs"], hp["lr"]
        net = NetFlex(X.shape[1], len(ids), emb=hp["emb"], h=hp["h"], depth=hp["depth"], drop=hp["drop"])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=hp["wd"])
    else:
        net = Net(X.shape[1], len(ids))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss() if kind == "bin" else nn.MSELoss()
    Xt, bt, yt = torch.as_tensor(X[~va]), torch.as_tensor(b[~va]), torch.as_tensor(y[~va].astype(np.float32))
    Xv, bv, yv = torch.as_tensor(X[va]), torch.as_tensor(b[va]), torch.as_tensor(y[va].astype(np.float32))
    best, state, bad = 9e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), bs):
            j = perm[i : i + bs]
            opt.zero_grad()
            l = lossf(net(Xt[j], bt[j]), yt[j])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = lossf(net(Xv, bv), yv).item()
        if lv < best - 1e-6:
            best, state, bad = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 2:
            break
    net.load_state_dict(state)
    net.eval()
    return net


def pred(net, X, b, kind):
    with torch.no_grad():
        o = net(torch.as_tensor(X), torch.as_tensor(b))
        return (torch.sigmoid(o) if kind == "bin" else o).numpy()


tr, te = df[df.year == TRAIN], df[df.year == TEST]
ZONE_ONLY = ["plate_xh", "plate_zn"]
M = {}
for task, sel, y, kind in (
    ("rv", lambda d: d.dropna(subset=["rv"]), "rv", "reg"),
    ("chase", lambda d: d[d.out_zone], "swing", "bin"),
    ("whiff", lambda d: d[d.swing == 1], "whiff", "bin"),
):
    a, b_ = sel(tr), sel(te)
    yscale = 10.0 if kind == "reg" else 1.0
    Xa, sc = design(a, F2)
    Xb, _ = design(b_, F2, sc)
    net = fit(Xa, a.bid.to_numpy(), a[y].to_numpy() * yscale, kind, task=task)
    pb = pred(net, Xb, b_.bid.to_numpy(), kind) / yscale
    Za, zsc = design(a, ZONE_ONLY)
    Zb, _ = design(b_, ZONE_ONLY, zsc)
    znet = fit(Za, a.bid.to_numpy(), a[y].to_numpy() * yscale, kind)
    pz = pred(znet, Zb, b_.bid.to_numpy(), kind) / yscale
    yb = b_[y].to_numpy()
    if kind == "reg":
        mse = lambda q: np.mean((yb - q) ** 2)
        base = mse(np.full(len(yb), a[y].mean()))
        print(f"{task}: test R2 full {1 - mse(pb)/base:.4f} | zone+count only {1 - mse(pz)/base:.4f}", flush=True)
    else:
        ll = lambda q: -np.mean(yb * np.log(np.clip(q, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - q, 1e-6, 1)))
        print(f"{task}: test logloss full {ll(pb):.4f} | zone+count only {ll(pz):.4f}", flush=True)
    M[task] = (net, sc, kind, yscale)
print(f"models trained in {time.time()-t0:.0f}s", flush=True)


def predict(task, d, dD=0.0, dL=0.0, dE=0.0):
    net, sc, kind, ysc = M[task]
    f = build2(d, d.depth.to_numpy(float) + dD, d.off_plate.to_numpy(float) + dL, fb, d_eye_in=dE)
    dd = d.copy()
    dd[F2] = f[F2].to_numpy()
    X, _ = design(dd, F2, sc)
    return pred(net, X, d.bid.to_numpy(), kind) / ysc


# ---- mechanism check: 1 ft closer (dL = -12 is far outside the data; use -3 in and scale x4)
band = np.select([te.plate_xh > 0.236, te.plate_xh < -0.236], ["away", "in"], "middle")
sw = te[te.swing == 1]
bsw = band[te.swing.to_numpy() == 1]
dw = (predict("whiff", sw, dL=-3.0) - predict("whiff", sw)) * 100 * 4
mid = dw[bsw == "middle"].mean()
print(
    f"\nmechanism, model: per ft CLOSER, whiff on AWAY {dw[bsw=='away'].mean()-mid:+.2f} pp, IN {dw[bsw=='in'].mean()-mid:+.2f} pp (vs middle)"
    f"   | measured: AWAY -3.35, IN +3.47",
    flush=True,
)

# ---- posture through perception: eyes 1 inch lower (crouch) or higher, everything else fixed (posture-eye runs only)
if EYE_MODE != "height":
    zn = te.plate_zn.to_numpy()
    oz, swm = te.out_zone.to_numpy(bool), te.swing.to_numpy() == 1
    for dE in (-1.0, 1.0):
        drv = (predict("rv", te, dE=dE) - predict("rv", te)) * 100
        dch = (predict("chase", te[oz], dE=dE) - predict("chase", te[oz])) * 100
        dwh = (predict("whiff", te[swm], dE=dE) - predict("whiff", te[swm])) * 100
        hi, lo = zn > 0.75, zn < 0.25
        print(
            f"eye {dE:+.0f} in: run value {drv.mean():+.4f}/100 pitches (high pitches {drv[hi].mean():+.4f}, low {drv[lo].mean():+.4f}); "
            f"chase {dch.mean():+.2f} pp; whiff {dwh.mean():+.2f} pp",
            flush=True,
        )

# ---- recommender
GRID = [(dD, dL) for dD in (-6.0, 0.0, 6.0) for dL in (-3.0, 0.0, 3.0)]
te = te.copy()
te["pss"] = te.pitcher.astype(str) + te.stand
cnt = te.groupby("pss").size()
good = cnt[cnt >= 300].index
scout = (
    te[te.pss.isin(good)].groupby("pss", group_keys=False).apply(lambda g: g.sample(min(len(g), 160), random_state=1))
)
scout["half"] = scout.groupby("pss").cumcount() % 2
reg = te.groupby("bsy").size()
reg = reg[reg >= 1500].index
hit = (
    te[te.bsy.isin(reg)]
    .groupby("bsy")
    .agg(
        batter=("batter", "first"),
        stand=("stand", "first"),
        depth=("depth_season", "first"),
        off_plate=("off_season", "first"),
        body_offset_in=("body_offset_in", "first"),
        height_in=("height_in", "first"),
        sz_top_m=("sz_top_m", "first"),
        sz_bot_m=("sz_bot_m", "first"),
        bid=("bid", "first"),
    )
    .sample(frac=1, random_state=2)
    .head(120)
)
rows = []
cols_h = ["batter", "depth", "off_plate", "body_offset_in", "height_in", "sz_top_m", "sz_bot_m", "bid"]
for i, (bsy, h) in enumerate(hit.iterrows()):
    sc_p = scout[scout.stand == h.stand].copy()
    for c in cols_h:
        sc_p[c] = h[c]
    sc_p["same"] = (sc_p.p_throws == h.stand).astype(np.float32)
    for dD, dL in GRID:
        sc_p[f"rv_{int(dD)}_{int(dL)}"] = predict("rv", sc_p, dD, dL) * 100  # runs per 100 pitches
    g = sc_p.groupby(["pss", "half"])[[f"rv_{int(dD)}_{int(dL)}" for dD, dL in GRID]].mean().reset_index()
    g["bsy"] = bsy
    g["p_throws"] = g.pss.map(sc_p.groupby("pss").p_throws.first())
    rows.append(g)
    if i % 30 == 0:
        print(f"  recommender: {i}/{len(hit)} hitters, {time.time()-t0:.0f}s", flush=True)
R = pd.concat(rows)
R.to_parquet(f"recommender_grid_{TAG}.parquet", index=False)
G = [f"rv_{int(dD)}_{int(dL)}" for dD, dL in GRID]
A, B = R[R.half == 0].set_index(["bsy", "pss"]), R[R.half == 1].set_index(["bsy", "pss"])
A, B = A.align(B, join="inner", axis=0)
cur = B["rv_0_0"]
choice_pp = A[G].idxmax(axis=1)  # per-pitcher choice on half A
gain_pp = pd.Series([B.loc[k, c] for k, c in zip(B.index, choice_pp)], index=B.index) - cur
default = A[G].groupby(level="bsy").mean().idxmax(axis=1)  # hitter's single best default (from half A)
gain_def = pd.Series([B.loc[k, default[k[0]]] for k in B.index], index=B.index) - cur
per_h = pd.DataFrame(
    {"default": gain_def.groupby(level="bsy").mean(), "per_pitcher": gain_pp.groupby(level="bsy").mean()}
)
per_h["adjust_value"] = per_h.per_pitcher - per_h.default
print(
    f"\nRECOMMENDER ({TEST}; {len(hit)} regulars x {B.index.get_level_values('pss').nunique()} pitcher-sides; runs per 100 pitches, scored out-of-half)"
)
print(
    f"  gain from the hitter's single best DEFAULT spot: mean {per_h.default.mean():+.3f} (median {per_h.default.median():+.3f})"
)
print(f"  gain from choosing the best spot PER PITCHER:   mean {per_h.per_pitcher.mean():+.3f}")
print(
    f"  => value of adjusting by pitcher (beyond the default): mean {per_h.adjust_value.mean():+.3f}, "
    f"90th pct hitter {per_h.adjust_value.quantile(.9):+.3f} runs/100 pitches (~x24 for a 2,400-pitch season)"
)
dcount = default.value_counts(normalize=True).round(2)
print("  hitters' best default spot (depth_lateral, inches; lateral - = closer):", dcount.to_dict())
agree = (choice_pp == pd.Series([default[k[0]] for k in choice_pp.index], index=choice_pp.index)).mean()
print(f"  share of hitter-pitcher matchups where the per-pitcher pick = the hitter's default: {agree:.0%}")
per_h.to_csv(f"recommender_value_{TAG}.csv")
full = R.groupby(["bsy", "pss"])[G].mean()
curf = full["rv_0_0"]
bp = full[G].max(axis=1) - curf
dflt = full[G].groupby(level="bsy").mean().idxmax(axis=1)
bd = pd.Series([full.loc[k, dflt[k[0]]] for k in full.index], index=full.index) - curf
print(f"  in-sample UPPER BOUND on per-pitcher value: {(bp - bd).groupby(level='bsy').mean().mean():+.3f}")
print(
    f"SUMMARY {TAG} default {per_h.default.mean():+.4f} per_pitcher {per_h.per_pitcher.mean():+.4f} adjust {per_h.adjust_value.mean():+.4f} "
    f"deeper_default_share {dflt.str.startswith('rv_6').mean():.2f}"
)
