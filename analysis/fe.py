"""Minimal high-dimensional FE OLS: alternating-projection demeaning + batter-clustered SEs."""

import numpy as np, pandas as pd


def demean(M, codes, iters=200, tol=1e-9):
    M = M.astype(float).copy()
    for _ in range(iters):
        delta = 0.0
        for c in codes:
            n = np.bincount(c)
            for j in range(M.shape[1]):
                mu = np.bincount(c, weights=M[:, j]) / np.maximum(n, 1)
                adj = mu[c]
                M[:, j] -= adj
                delta = max(delta, np.abs(adj).max())
        if delta < tol:
            break
    return M


def feols(df, y, xcols, fes, cluster):
    codes = [
        pd.factorize(df[f])[0] if isinstance(f, str) else pd.factorize(df[list(f)].astype(str).agg("|".join, axis=1))[0]
        for f in fes
    ]
    M = demean(np.column_stack([df[y].to_numpy()] + [df[c].to_numpy() for c in xcols]), codes)
    yy, X = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ yy
    u = yy - X @ b
    g = pd.factorize(df[cluster])[0]
    G = g.max() + 1
    S = np.zeros((X.shape[1], X.shape[1]))
    Xu = X * u[:, None]
    sums = np.vstack([np.bincount(g, weights=Xu[:, j], minlength=G) for j in range(X.shape[1])]).T
    S = sums.T @ sums
    n, k = X.shape
    V = XtX_inv @ S @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return pd.Series(b, index=xcols), pd.DataFrame(V, index=xcols, columns=xcols)
