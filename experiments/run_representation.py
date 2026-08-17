"""Results 1 and 3: representation fidelity, and the 16-bit comparison.

Writes results/representation.csv
"""
import _common
import numpy as np, pandas as pd
from positscale import (Posit, Float, gen_matrix, apply_scaling, SCALINGS)

P32, F32, P16, F16 = Posit(32, 2), Float(32), Posit(16, 2), Float(16)
METHODS = ["none", "ruiz", "logmean", "lp_inf"]


def rep_error(fmt, M):
    q = fmt.quantize(M)
    ok = np.isfinite(q) & (q != 0)
    if not ok.any():
        return np.nan, 0.0
    return float(np.max(np.abs(q[ok] - M[ok]) / np.abs(M[ok]))), float(ok.mean())


rows = []
for fam in ("tree", "mixed", "cyclic"):
    for sp in (16, 32, 48):
        for rep in range(6):
            rng = np.random.default_rng(_common.seed(fam, sp, rep, "rep"))
            A = gen_matrix(fam, 40, sp, rng)
            for name in METHODS:
                d, e = SCALINGS[name](A)
                As, _, _ = apply_scaling(A, d, e, pow2=True)
                p32, _ = rep_error(P32, As)
                f32, _ = rep_error(F32, As)
                p16, p16_ok = rep_error(P16, As)
                f16, f16_ok = rep_error(F16, As)
                rows.append(dict(family=fam, spread=sp, scaling=name,
                                 posit32=p32, float32=f32,
                                 posit16=p16, float16=f16,
                                 posit16_representable=p16_ok,
                                 float16_representable=f16_ok,
                                 bits=float(np.mean(P32.bits_at(As[As != 0]))),
                                 cond=float(np.linalg.cond(As))))

df = pd.DataFrame(rows)
df.to_csv(_common.RESULTS / "representation.csv", index=False)

cols = ["posit32", "float32", "posit16", "float16",
        "float16_representable", "bits"]
print("\nMedian over families and spreads (n=40):\n")
print(df.groupby("scaling")[cols].median().reindex(METHODS)
        .to_string(float_format=lambda v: f"{v:.4g}"))
print("\nposit32 gain from best scaling : "
      f"{df[df.scaling=='none'].posit32.median() / df[df.scaling=='logmean'].posit32.median():.1f}x")
print("float32 gain from best scaling : "
      f"{df[df.scaling=='none'].float32.median() / df[df.scaling=='logmean'].float32.median():.1f}x")
