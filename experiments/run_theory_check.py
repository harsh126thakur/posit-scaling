"""Result 2: the L-infinity optimum equals the maximum mean cycle weight.

Writes results/theory.csv
"""
import _common
import numpy as np, pandas as pd
from positscale import gen_matrix, scale_lp_inf, max_mean_cycle, FAMILIES

SPREADS = [0, 8, 16, 24, 32, 40, 48]
REPS, N = 3, 10

rows = []
for fam in FAMILIES:
    for sp in SPREADS:
        for rep in range(REPS):
            rng = np.random.default_rng(_common.seed(fam, sp, rep, "theory"))
            A = gen_matrix(fam, N, sp, rng)
            _, _, lp = scale_lp_inf(A, return_value=True)
            rows.append(dict(family=fam, spread=sp, rep=rep,
                             lp=lp, mmc=max_mean_cycle(A)))

df = pd.DataFrame(rows)
df.to_csv(_common.RESULTS / "theory.csv", index=False)

dev = np.max(np.abs(df.lp - df.mmc) / np.maximum(np.abs(df.mmc), 1e-12))
print(f"matrices tested            : {len(df)}")
print(f"max relative deviation     : {dev:.2e}")
print("PASS" if dev < 1e-9 else "FAIL")
