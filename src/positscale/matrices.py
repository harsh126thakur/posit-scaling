"""Test matrices with controlled log-magnitude structure.

Entries are +/- u * 2^zeta with u in [1,2), so the base contributes at most one
bit of spread and the *family* alone controls magnitude structure. This is what
lets the experiments separate removable spread from irreducible spread.
"""
from __future__ import annotations
import numpy as np

__all__ = ["gen_matrix", "FAMILIES"]

FAMILIES = ("centered", "tree", "mixed", "cyclic")


def gen_matrix(kind: str, n: int, spread: float, rng: np.random.Generator):
    """Generate an n x n test matrix.

    Parameters
    ----------
    kind : {"centered", "tree", "mixed", "cyclic"}
        ``tree``     zeta_ij = r_i + k_j -- spread lies on a spanning tree and
                     is therefore entirely removable by diagonal scaling.
        ``cyclic``   zeta_ij drawn per entry -- spread lies on cycles and no
                     diagonal scaling can remove it.
        ``mixed``    both components.
        ``centered`` zeta = 0, already at unit magnitude: the null case.
    n : int
        Dimension.
    spread : float
        Total magnitude spread in bits.
    rng : numpy.random.Generator
        Source of randomness; pass a seeded generator for reproducibility.
    """
    base = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(1.0, 2.0, (n, n))
    h = spread / 2

    if kind == "centered":
        return base
    if kind == "tree":
        r, k = rng.uniform(-h, h, n), rng.uniform(-h, h, n)
        return base * 2.0 ** (r[:, None] + k[None, :])
    if kind == "cyclic":
        return base * 2.0 ** rng.uniform(-h, h, (n, n))
    if kind == "mixed":
        r, k = rng.uniform(-h, h, n), rng.uniform(-h, h, n)
        z = rng.uniform(-spread / 6, spread / 6, (n, n))
        return base * 2.0 ** (r[:, None] + k[None, :] + z)
    raise ValueError(f"unknown family {kind!r}; expected one of {FAMILIES}")
