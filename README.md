# posit-scaling

**Optimal diagonal scaling for posit arithmetic.**

Posit precision is *tapered*: it peaks at unit magnitude and decays away from it.
Unlike IEEE 754, therefore, posit accuracy depends on **where** operand
magnitudes sit — which makes diagonal scaling `A → D₁AD₂` a genuine numerical
lever rather than a conditioning formality.

This repository formulates posit-optimal scaling as a convex programme, solves it
exactly, and measures what it buys.

---

## Results

**1. Scaling is a posit-specific lever.** Worst-entry representation error,
median over matrix families and magnitude spreads (n = 40):

| scaling | posit32 | float32 | posit16 | posit32 fraction bits |
|---|---|---|---|---|
| none | 1.90e-07 | 5.81e-08 | 1.38e-02 | 24.9 |
| Ruiz | 2.55e-08 | 5.81e-08 | 1.65e-03 | 26.0 |
| **LogMean** | **6.91e-09** | 5.81e-08 | **4.65e-04** | 26.8 |
| LP (L∞) | 7.01e-09 | 5.81e-08 | 4.69e-04 | 26.7 |

Posit32 improves **27.6×**; float32 does not move at all, exactly as a flat
precision profile predicts. Posit32 crosses over from *worse* than float32
unscaled to **8× better** scaled.

**2. The obstruction has an exact characterisation.** The L∞ optimum equals the
**maximum mean cycle weight** of the bipartite log-magnitude digraph, computable
by Karp's algorithm before any arithmetic runs. Verified across 84 matrices:

> max relative deviation between LP optimum and max mean cycle = **6.0e-15**

**3. The 16-bit case (the embedded motivation).** On the same matrices, unscaled
float16 **fails to represent 13% of entries** (overflow), while posit16 saturates
and represents all of them. After optimal scaling posit16 reaches 4.65e-4 against
float16's 4.78e-4 — float16-class precision *with* range safety.

**4. Negative result: representation is not enough.** On the actual LU solve, the
L∞ optimum wins where spread is removable and loses where it is not:

| family | removable spread | error(Ruiz) / error(LP) |
|---|---|---|
| tree | 0.98 | **1.09** — LP wins |
| mixed | 0.84 | 0.38 |
| cyclic | 0.10 | 0.27 — Ruiz wins |

Row scaling does two jobs: it repositions magnitudes **and** it changes which
pivots partial pivoting selects. The L∞ objective optimises only the first. The
correct objective must be composite — this is the open problem.

---

## The formulation

With `c_ij = log₂|a_ij|`, `δ_i = log₂ D₁ᵢᵢ`, `ε_j = log₂ D₂ⱼⱼ`, the fraction bits
a posit loses on a scaled entry are `|c_ij + δ_i + ε_j| / 2^es`. Convex,
piecewise-linear. Two objectives:

- **L∞** — `min max |c + δ + ε|`. A linear programme; optimum = max mean cycle.
- **L2** — `min Σ (c + δ + ε)²`. Closed form: double centring in log space,
  `O(nnz)`, no solver required.

Scale factors are rounded to powers of two, so `D₁` and `D₂` are exact shifts.

---

## Install

```bash
git clone https://github.com/harsh126thakur/posit-scaling
cd posit-scaling
pip install -r requirements.txt
```

Python ≥ 3.10, NumPy, SciPy, pandas, matplotlib. No compiled dependencies.

## Reproduce

```bash
make test        # validate the quantizer against all 65,536 posit16 values
make theory      # Result 2: LP optimum vs Karp max mean cycle    (~10 s)
make represent   # Results 1 and 3: representation fidelity       (~20 s)
make decision    # Result 4: removability as a decision rule      (~15 s)
make sweep       # 1,728 LU solves                                (~70 s)
make figures     # all figures into paper/figures/
make all         # everything above, ~2 minutes total
```

All seeds are derived deterministically (`experiments/_common.py:seed`), so the
numbers above reproduce exactly rather than approximately.

Every script writes a CSV into `results/` and is independently runnable.

## Usage

```python
import numpy as np
from positscale import Posit, scale_logmean, apply_scaling, max_mean_cycle

A = np.random.randn(40, 40) * 2.0 ** np.random.uniform(-20, 20, (40, 40))
p32 = Posit(32, es=2)

print(max_mean_cycle(A))              # irreducible spread, in bits, a priori

d, e = scale_logmean(A)               # O(nnz), closed form
As, D1, D2 = apply_scaling(A, d, e)   # power-of-two factors

err = lambda M: np.max(np.abs(p32.quantize(M) - M) / np.abs(M))
print(err(A), "→", err(As))           # representation error before / after
```

---

## Validation

The posit quantizer is not a wrapper around SoftPosit — it is an independent
implementation, and it is checked against brute-force enumeration of the complete
posit16 value set:

```
$ make test
7 passed
```

The suite checks exact agreement with enumeration on 40,000 random samples, the
shape of the precision profile, saturation-instead-of-overflow, idempotence,
the max-mean-cycle identity, and that scaling moves posits but not floats.

It reproduces the defining property of the format: posit32 holds 27 fraction bits
at magnitude 1 versus float32's 23, losing one bit per four octaves of distance.
Note that the V is *not* symmetric — the regime run costs `k+2` bits above unit
magnitude but `-k+1` below it, so posits keep one extra fraction bit on the
small side.

Arithmetic is simulated by computing in binary64 and rounding onto the posit
grid. This is faithful because binary64's 53 significand bits exceed posit32's
maximum of 27, so the intermediate is exact to well within half a posit ULP.

## Limitations

- **Synthetic matrices only.** Dense, with controlled log-magnitude structure.
  Sparsity changes the cycle structure the max-mean-cycle result depends on;
  validation on application matrices is the necessary next step.
- **The max-mean-cycle relation is verified numerically, not proved.** It is
  probably a known result in tropical / max-plus linear algebra under another
  name.
- **No quire.** The LU kernel rounds the product and the subtraction separately.
  Exact accumulation would attenuate the growth-factor term and may shift the
  balance back toward the representation objective.
- **Software simulation**, not posit hardware. No performance claims are made.

## Layout

```
src/positscale/   quantizer.py  scaling.py  matrices.py  kernels.py
tests/            brute-force quantizer validation (7 tests)
experiments/      one script per result, each writing a CSV
paper/            IEEE Embedded Systems Letters manuscript + figures
results/          generated CSVs (gitignored; regenerate with `make all`)
```

## Citation

See `CITATION.cff`. The accompanying manuscript is in `paper/`.

## License

MIT — see `LICENSE`.
