"""Does removability predict which scaling to use?

Compares Ruiz against the L-infinity LP as a function of how much of the
magnitude spread is removable (computed from the max mean cycle weight).

Writes results/decision.csv
"""
import _common
import numpy as np, pandas as pd
from positscale import (Posit, gen_matrix, apply_scaling, SCALINGS,
                        lu_solve, backward_error, removable_fraction)

P32 = Posit(32, 2)
rows = []
for fam in ("tree", "mixed", "cyclic"):
    for sp in (8, 16, 24, 32, 40, 48):
        for rep in range(5):
            rng = np.random.default_rng(_common.seed(fam, sp, rep, "decision"))
            A = gen_matrix(fam, 24, sp, rng)
            x_true = rng.standard_normal(24)
            b = A @ x_true
            err = {}
            for name in ("ruiz", "lp_inf"):
                d, e = SCALINGS[name](A)
                As, D1, D2 = apply_scaling(A, d, e, pow2=True)
                x = D2 * lu_solve(As, D1 * b, P32)
                err[name] = backward_error(A, x, b)
            rows.append(dict(family=fam, spread=sp,
                             removable=removable_fraction(A),
                             ratio=err["ruiz"] / err["lp_inf"]))

df = pd.DataFrame(rows)
df.to_csv(_common.RESULTS / "decision.csv", index=False)
print("\nRatio > 1 means the posit-optimal LP wins:\n")
print(df.groupby("family")[["removable", "ratio"]].median()
        .to_string(float_format=lambda v: f"{v:.3f}"))
df["bin"] = pd.qcut(df.removable, 3, labels=["low", "mid", "high"])
print("\nFraction of cases where LP beats Ruiz, by removability tercile:")
print(df.groupby("bin", observed=True).ratio.apply(lambda g: (g > 1).mean())
        .to_string(float_format=lambda v: f"{v:.2f}"))
