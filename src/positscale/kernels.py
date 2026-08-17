"""Linear-algebra kernels executed entirely in a simulated number format."""
from __future__ import annotations
import numpy as np

__all__ = ["lu_solve", "backward_error", "forward_error"]


def lu_solve(A, b, fmt):
    """Right-looking LU with partial pivoting, in-format throughout.

    Every intermediate is rounded to ``fmt``. No quire: the product and the
    subtraction round separately, which is the behaviour of a posit unit
    without exact accumulation.

    Parameters
    ----------
    A, b : array_like
        System to solve.
    fmt : Posit or Float
        Any object exposing ``quantize``.
    """
    q = fmt.quantize
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    U = q(A.copy())
    x = q(np.asarray(b, dtype=np.float64).copy())

    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p]] = U[[p, k]]
            x[[k, p]] = x[[p, k]]
        if U[k, k] == 0:
            continue
        mult = q(U[k + 1:, k] / U[k, k])
        U[k + 1:, k] = mult
        U[k + 1:, k + 1:] = q(U[k + 1:, k + 1:] - q(np.outer(mult, U[k, k + 1:])))
        x[k + 1:] = q(x[k + 1:] - q(mult * x[k]))

    y = x
    for k in range(n - 1, -1, -1):
        if k < n - 1:
            acc = y[k]
            for v in q(U[k, k + 1:] * y[k + 1:]):   # sequential, no quire
                acc = q(acc - v)
            y[k] = acc
        y[k] = 0.0 if U[k, k] == 0 else q(y[k] / U[k, k])
    return y


def backward_error(A, x, b):
    """Componentwise relative residual max_i |r_i| / (|A||x| + |b|)_i.

    Evaluated in binary64 so that the metric itself is not polluted by the
    format under test.
    """
    A = np.asarray(A, float)
    r = np.asarray(b, float) - A @ np.asarray(x, float)
    den = np.abs(A) @ np.abs(x) + np.abs(b)
    den = np.where(den == 0, 1.0, den)
    return float(np.max(np.abs(r) / den))


def forward_error(x, x_true):
    """Relative 2-norm error against a known solution."""
    return float(np.linalg.norm(np.asarray(x, float) - x_true)
                 / np.linalg.norm(x_true))
