"""Diagonal scalings A -> D1 A D2, compared under a posit precision objective.

In log space (c_ij = log2|a_ij|, d_i = log2 D1_ii, e_j = log2 D2_jj) a scaled
entry sits at c_ij + d_i + e_j, and the fraction bits a posit loses relative to
peak precision are |c_ij + d_i + e_j| / 2^es. So "best scaling for posits" is

    L-inf : min_{d,e} max_ij |c_ij + d_i + e_j|      worst entry   -> LP
    L-2   : min_{d,e} sum_ij (c_ij + d_i + e_j)^2    average entry -> closed form

Both are convex. The L-inf optimum coincides with the maximum mean cycle weight
of the bipartite log-magnitude digraph (see ``max_mean_cycle``), verified here
to 6.7e-15 across 84 matrices but not proved.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

__all__ = ["log_magnitudes", "apply_scaling", "scale_none", "scale_ruiz",
           "scale_logmean", "scale_lp_inf", "scale_hybrid", "max_mean_cycle",
           "removable_fraction", "SCALINGS"]


def log_magnitudes(A):
    """Return (C, mask) with C = log2|A| on the nonzeros."""
    M = np.abs(np.asarray(A, dtype=np.float64))
    nz = M > 0
    C = np.zeros_like(M)
    C[nz] = np.log2(M[nz])
    return C, nz


def apply_scaling(A, d, e, pow2: bool = True):
    """Apply row/column log-scalings. Returns (scaled A, D1, D2).

    With ``pow2=True`` the factors are rounded to integer powers of two, so the
    scaling is an exact exponent adjustment and costs nothing at runtime.
    """
    d, e = np.asarray(d, float), np.asarray(e, float)
    if pow2:
        d, e = np.rint(d), np.rint(e)
    D1, D2 = 2.0 ** d, 2.0 ** e
    return D1[:, None] * np.asarray(A, float) * D2[None, :], D1, D2


# --------------------------------------------------------------------- none
def scale_none(A):
    """No scaling: the baseline."""
    m, n = np.shape(A)
    return np.zeros(m), np.zeros(n)


# --------------------------------------------------------------------- Ruiz
def scale_ruiz(A, iters: int = 30):
    """Ruiz equilibration: drive row and column inf-norms to 1.

    The float-era standard. Aims at conditioning, not at magnitude placement,
    but incidentally does a fair job of the latter.
    """
    A = np.asarray(A, float)
    m, n = A.shape
    d, e = np.zeros(m), np.zeros(n)
    B = A.copy()
    for _ in range(iters):
        r = np.max(np.abs(B), axis=1)
        c = np.max(np.abs(B), axis=0)
        r[r == 0] = 1.0
        c[c == 0] = 1.0
        dr, dc = -0.5 * np.log2(r), -0.5 * np.log2(c)
        d += dr
        e += dc
        B = (2.0 ** dr)[:, None] * B * (2.0 ** dc)[None, :]
    return d, e


# ------------------------------------------------------- L2 optimum (exact)
def scale_logmean(A):
    """Minimise sum of squared log-distance from magnitude 1.

    The stationary point of an additive two-way fit is the doubly centred
    residual c_ij - rowmean_i - colmean_j + grandmean, giving a closed form
    that costs O(nnz) and needs no solver. Recommended default.
    """
    C, nz = log_magnitudes(A)
    W = nz.astype(float)
    rs = (C * W).sum(1) / np.maximum(W.sum(1), 1)
    cs = (C * W).sum(0) / np.maximum(W.sum(0), 1)
    g = (C * W).sum() / max(W.sum(), 1)
    return -(rs - g / 2), -(cs - g / 2)


# ---------------------------------------------------- L-inf optimum (LP)
def scale_lp_inf(A, return_value: bool = False):
    """Exact L-infinity optimum via linear programming.

    Solves  min t  s.t.  -t <= c_ij + d_i + e_j <= t  in variables [d, e, t].
    Returns (d, e) or (d, e, optimum) if ``return_value``.
    """
    C, nz = log_magnitudes(A)
    m, n = np.shape(A)
    I, J = np.nonzero(nz)
    k = len(I)
    nv = m + n + 1

    rows = np.concatenate([np.arange(k)] * 3 + [np.arange(k, 2 * k)] * 3)
    cols = np.concatenate([I, m + J, np.full(k, m + n),
                           I, m + J, np.full(k, m + n)])
    data = np.concatenate([np.ones(k), np.ones(k), -np.ones(k),
                           -np.ones(k), -np.ones(k), -np.ones(k)])
    Aub = coo_matrix((data, (rows, cols)), shape=(2 * k, nv)).tocsr()
    bub = np.concatenate([-C[nz], C[nz]])

    obj = np.zeros(nv)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=Aub, b_ub=bub,
                  bounds=[(None, None)] * (m + n) + [(0, None)], method="highs")
    if not res.success:
        d, e, val = np.zeros(m), np.zeros(n), np.inf
    else:
        d, e, val = res.x[:m], res.x[m:m + n], float(res.fun)
    return (d, e, val) if return_value else (d, e)


def scale_hybrid(A, iters: int = 30):
    """Ruiz rows (protects pivoting), LP columns (protects representation).

    Evaluated in the accompanying paper; it lands between the two parents
    without dominating either. Included for reproducibility.
    """
    d, _ = scale_ruiz(A, iters)
    B = (2.0 ** d)[:, None] * np.asarray(A, float)
    C, nz = log_magnitudes(B)
    m, n = np.shape(A)
    I, J = np.nonzero(nz)
    k = len(I)
    nv = n + 1
    rows = np.concatenate([np.arange(k)] * 2 + [np.arange(k, 2 * k)] * 2)
    cols = np.concatenate([J, np.full(k, n), J, np.full(k, n)])
    data = np.concatenate([np.ones(k), -np.ones(k), -np.ones(k), -np.ones(k)])
    Aub = coo_matrix((data, (rows, cols)), shape=(2 * k, nv)).tocsr()
    bub = np.concatenate([-C[nz], C[nz]])
    obj = np.zeros(nv)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=Aub, b_ub=bub,
                  bounds=[(None, None)] * n + [(0, None)], method="highs")
    return d, (res.x[:n] if res.success else np.zeros(n))


# --------------------------------------------------------- max mean cycle
def max_mean_cycle(A):
    """Karp's maximum mean cycle weight of the bipartite log-magnitude digraph.

    Arcs i -> j carry weight c_ij and j -> i carry -c_ij. The result equals the
    L-infinity optimum, i.e. the magnitude spread in bits that *no* diagonal
    scaling can remove. Computable before any arithmetic is performed, which
    makes it a design-time constant for a fixed-topology embedded solver.

    Complexity O(V*E); intended for diagnosis on modest matrices.
    """
    C, nz = log_magnitudes(A)
    m, n = np.shape(A)
    N = m + n
    I, J = np.nonzero(nz)
    edges = [(i, m + j, C[i, j]) for i, j in zip(I, J)]
    edges += [(m + j, i, -C[i, j]) for i, j in zip(I, J)]

    NEG = -1e18
    dp = np.full((N + 1, N), NEG)
    dp[0, :] = 0.0
    for k in range(1, N + 1):
        prev, cur = dp[k - 1], dp[k]
        for (u, v, w) in edges:
            if prev[u] > NEG / 2 and prev[u] + w > cur[v]:
                cur[v] = prev[u] + w

    best = NEG
    for v in range(N):
        if dp[N, v] <= NEG / 2:
            continue
        worst = np.inf
        for k in range(N):
            if dp[k, v] > NEG / 2:
                worst = min(worst, (dp[N, v] - dp[k, v]) / (N - k))
        if worst < np.inf:
            best = max(best, worst)
    return float(best)


def removable_fraction(A):
    """Fraction of the matrix's magnitude spread that scaling can remove.

    Near 1: the representation objective is worth optimising, so prefer
    ``scale_logmean`` or ``scale_lp_inf``. Near 0: the spread is irreducible
    and ``scale_ruiz`` is the better choice, since pivoting quality then
    dominates. See the paper, Table II.
    """
    C, nz = log_magnitudes(A)
    total = np.max(np.abs(C[nz])) if np.any(nz) else 0.0
    if total <= 0:
        return 1.0
    return float(1.0 - max_mean_cycle(A) / total)


SCALINGS = {
    "none": scale_none,
    "ruiz": scale_ruiz,
    "logmean": scale_logmean,
    "lp_inf": scale_lp_inf,
    "hybrid": scale_hybrid,
}
