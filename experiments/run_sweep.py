"""Result 4: the main LU sweep -- 1,728 solves.

Shows that minimising representation error alone does NOT minimise solve
error, because row scaling also governs pivot selection.

Writes results/sweep.csv   (~75 s)
"""
import _common
import numpy as np, pandas as pd, time
from positscale import (Posit, Float, gen_matrix, apply_scaling, SCALINGS,
                        lu_solve, backward_error, forward_error,
                        removable_fraction, FAMILIES)

FMT = {"posit32": Posit(32, 2), "float32": Float(32), "posit16": Posit(16, 2)}
METHODS = ["none", "ruiz", "logmean", "lp_inf"]
SPREADS = [0, 8, 16, 24, 32, 40]
N, REPS = 40, 6

rows, t0 = [], time.time()
for fam in FAMILIES:
    for sp in SPREADS:
        for rep in range(REPS):
            rng = np.random.default_rng(_common.seed(fam, sp, rep, "sweep"))
            A = gen_matrix(fam, N, sp, rng)
            x_true = rng.standard_normal(N)
            b = A @ x_true
            removable = removable_fraction(A) if N <= 24 else np.nan

            for name in METHODS:
                d, e = SCALINGS[name](A)
                As, D1, D2 = apply_scaling(A, d, e, pow2=True)
                for fname, fmt in FMT.items():
                    x = D2 * lu_solve(As, D1 * b, fmt)
                    rows.append(dict(
                        family=fam, spread=sp, rep=rep, scaling=name,
                        fmt=fname,
                        bwd=backward_error(A, x, b),
                        fwd=forward_error(x, x_true),
                        removable=removable,
                        cond=float(np.linalg.cond(A))))
        print(f"  {fam:9s} spread={sp:<3d} ({time.time()-t0:5.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(_common.RESULTS / "sweep.csv", index=False)
print(f"\n{len(df)} solves in {time.time()-t0:.0f}s\n")

piv = df.pivot_table(index=["fmt", "family"], columns="scaling",
                     values="bwd", aggfunc="median")[METHODS]
print("Median backward error:\n")
print(piv.to_string(float_format=lambda v: f"{v:.2e}"))
