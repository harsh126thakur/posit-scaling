"""Validation of the posit quantiser against brute-force enumeration.

Run with:  pytest -q      (or)   python tests/test_quantizer.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from positscale import Posit, Float


def enumerate_posit16():
    """Every positive posit16 (es=2) value, by direct construction."""
    vals = []
    for k in range(-14, 15):
        L = k + 2 if k >= 0 else -k + 1
        r = 15 - L
        if r < 0:
            continue
        keep = min(2, max(0, r))
        fb = max(0, r - 2)
        for j in range(0, 4, 1 << (2 - keep)):
            sc = k * 4 + j
            if abs(sc) > 56:
                continue
            for f in range(1 << fb):
                vals.append(2.0 ** sc * (1 + f / (1 << fb)))
    vals += [2.0 ** 56, 2.0 ** -56]          # maxpos, minpos
    return np.unique(np.array(vals))


def test_quantizer_matches_brute_force():
    """Rounding must land on the true nearest grid point, every time."""
    p = Posit(16, 2)
    grid = enumerate_posit16()
    rng = np.random.default_rng(0)
    x = 2.0 ** rng.uniform(-56, 56, 40_000) * rng.choice([-1, 1], 40_000)

    got = p.quantize(x)
    idx = np.clip(np.searchsorted(grid, np.abs(x)), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    want = np.sign(x) * np.where(np.abs(x) - lo <= hi - np.abs(x), lo, hi)

    assert np.array_equal(got, want), "quantiser disagrees with enumeration"


def test_precision_profile_is_a_v():
    """27 bits at unit magnitude, one lost per four octaves.

    The V is *not* symmetric: the regime run costs k+2 bits above unit
    magnitude but only -k+1 below it, so posits hold one extra fraction bit
    on the small side.
    """
    p = Posit(32, 2)
    assert int(p.bits_at(1.0)) == 27
    assert int(p.bits_at(2.0 ** 4)) == 26
    assert int(p.bits_at(2.0 ** 40)) == 17
    assert int(p.bits_at(2.0 ** -40)) == int(p.bits_at(2.0 ** 40)) + 1
    assert int(p.bits_at(1.0)) > Float(32).fb          # beats float32 near 1


def test_saturation_not_overflow():
    """Posits saturate; they have no Inf and no NaN."""
    p = Posit(16, 2)
    assert p.quantize(1e300) == p.maxpos
    assert p.quantize(-1e300) == -p.maxpos
    assert p.quantize(1e-300) == p.minpos
    assert p.quantize(0.0) == 0.0
    assert np.isinf(Float(16).quantize(1e30))          # float16, by contrast


def test_idempotent():
    """Quantising an already-representable value must not move it."""
    p = Posit(32, 2)
    rng = np.random.default_rng(1)
    x = p.quantize(2.0 ** rng.uniform(-60, 60, 5_000)
                   * rng.choice([-1, 1], 5_000))
    assert np.array_equal(p.quantize(x), x)


def test_scalar_in_scalar_out():
    p = Posit(32, 2)
    assert np.ndim(p.quantize(1.5)) == 0


def test_lp_equals_max_mean_cycle():
    """The central empirical claim: L-inf optimum == max mean cycle weight."""
    from positscale import gen_matrix, scale_lp_inf, max_mean_cycle
    rng = np.random.default_rng(7)
    for fam in ("centered", "tree", "mixed", "cyclic"):
        for spread in (8, 24, 40):
            A = gen_matrix(fam, 8, spread, rng)
            _, _, lp = scale_lp_inf(A, return_value=True)
            mmc = max_mean_cycle(A)
            assert abs(lp - mmc) <= 1e-8 * max(abs(mmc), 1.0), (fam, spread)


def test_scaling_helps_posit_not_float():
    """Scaling is a posit-specific lever: float32 must be invariant."""
    from positscale import gen_matrix, scale_logmean, apply_scaling
    rng = np.random.default_rng(3)
    A = gen_matrix("tree", 20, 32, rng)
    d, e = scale_logmean(A)
    As, _, _ = apply_scaling(A, d, e)

    def rep(fmt, M):
        return np.max(np.abs(fmt.quantize(M) - M) / np.abs(M))

    p32, f32 = Posit(32, 2), Float(32)
    assert rep(p32, As) < rep(p32, A) / 5          # posits improve a lot
    assert np.isclose(rep(f32, As), rep(f32, A), rtol=0.5)   # floats do not


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
