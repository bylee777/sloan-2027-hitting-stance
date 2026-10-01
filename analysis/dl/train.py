"""Hitter-decision models. Tasks: chase (swing | out-of-zone pitch), whiff (whiff | swing).
1. Sufficiency: gradient boosting on PERCEPTION features vs on FULL raw pitch data (trajectory, type, spin,
   release) — train 2024-25, test 2026, no hitter identity in either. Small gap = what the hitter sees at
   the decision instant carries (almost) all the information.
2. Perception network: MLP on perception features + count + same-hand + batter embedding; test 2026
   log loss vs the boosting models.
"""

import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

torch.manual_seed(0)
np.random.seed(0)
PERC = ["d_x", "d_z", "d_vx", "d_vz", "d_vy", "d_elapsed", "sep_x", "sep_z", "sep_y", "c_x", "c_zn"]
FULL = [
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "sz_top_m",
    "sz_bot_m",
    "pt_code",
    "stand_R",
]
CTX = ["balls", "strikes", "same"]


def clean(df):
    m = (
        (df.d_x.abs() < 5)
        & (df.c_x.abs() < 5)
        & df.c_zn.between(-3, 5)
        & df.d_elapsed.between(0.1, 0.6)
        & (df.sep_y.abs() < 20)
    )
    return df[m].copy()


class Net(nn.Module):
    def __init__(self, n_in, n_bat, emb=8, h=128):
        super().__init__()
        self.emb = nn.Embedding(n_bat + 1, emb)  # index 0 = unknown hitter
        nn.init.normal_(self.emb.weight, 0, 0.01)
        self.f = nn.Sequential(
            nn.Linear(n_in + emb, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x, b):
        return self.f(torch.cat([x, self.emb(b)], 1)).squeeze(1)


def fit_net(Xtr, btr, ytr, Xva, bva, yva, n_bat, epochs=12, lr=2e-3, bs=4096):
    net = Net(Xtr.shape[1], n_bat)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, btr, ytr = map(torch.as_tensor, (Xtr, btr, ytr))
    Xva_t, bva_t = torch.as_tensor(Xva), torch.as_tensor(bva)
    best, best_state, bad = 9e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), bs):
            j = perm[i : i + bs]
            opt.zero_grad()
            l = lossf(net(Xtr[j], btr[j]), ytr[j])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva_t, bva_t)).numpy()
        ll = log_loss(yva, np.clip(pv, 1e-6, 1 - 1e-6))
        if ll < best - 1e-5:
            best, best_state, bad = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 2:
            break
    net.load_state_dict(best_state)
    return net, best


class Scaler:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.m) / self.s).astype(np.float32)


def design(df, sc=None):
    X = np.column_stack(
        [
            df[PERC].to_numpy(float),
            np.eye(4)[df.balls.clip(0, 3).astype(int)],
            np.eye(3)[df.strikes.clip(0, 2).astype(int)],
            df[["same"]].to_numpy(float),
        ]
    )
    sc = sc or Scaler().fit(X)
    return sc(X), sc


if __name__ == "__main__":
    df = clean(pd.read_parquet("pitches_feat.parquet"))
    df["pt_code"] = df.pitch_type.astype("category").cat.codes
    df["stand_R"] = (df.stand == "R").astype(float)
    tasks = {"chase": (df[df.out_zone], "swing"), "whiff": (df[df.swing == 1], "whiff")}
    res = []
    for task, (d, y) in tasks.items():
        tr, te = d[d.year == "2024"], d[d.year == "2025"]
        for lab, cols in (("boost: perception", PERC + CTX), ("boost: FULL raw pitch data", FULL + CTX)):
            t0 = time.time()
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
            m.fit(tr[cols], tr[y])
            p = m.predict_proba(te[cols])[:, 1]
            res.append(
                dict(
                    task=task,
                    model=lab,
                    test_logloss=log_loss(te[y], p),
                    test_auc=roc_auc_score(te[y], p),
                    secs=round(time.time() - t0),
                )
            )
        # perception network with batter embedding (batter id, not batter-season, so 2026 hitters are known)
        ids = {b: i + 1 for i, b in enumerate(tr.batter.unique())}
        Xtr, sc = design(tr)
        Xte, _ = design(te, sc)
        btr = tr.batter.map(ids).to_numpy(np.int64)
        bte = te.batter.map(ids).fillna(0).to_numpy(np.int64)
        va = np.random.rand(len(tr)) < 0.05
        t0 = time.time()
        net, _ = fit_net(
            Xtr[~va],
            btr[~va],
            tr[y].to_numpy(np.float32)[~va],
            Xtr[va],
            btr[va],
            tr[y].to_numpy(np.float32)[va],
            len(ids),
        )
        with torch.no_grad():
            p = torch.sigmoid(net(torch.as_tensor(Xte), torch.as_tensor(bte))).numpy()
        res.append(
            dict(
                task=task,
                model="NETWORK: perception + hitter embedding",
                test_logloss=log_loss(te[y], p),
                test_auc=roc_auc_score(te[y], p),
                secs=round(time.time() - t0),
            )
        )
        base = te[y].mean()
        res.append(
            dict(
                task=task,
                model="(constant rate)",
                test_logloss=log_loss(te[y], np.full(len(te), base)),
                test_auc=0.5,
                secs=0,
            )
        )
        print(pd.DataFrame(res[-4:]).round(4).to_string(index=False), flush=True)
    pd.DataFrame(res).to_csv("model_eval.csv", index=False)
