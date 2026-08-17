"""Validated posit round-to-nearest-even quantiser.

Arithmetic is simulated by computing in binary64 and rounding the result onto
the posit grid. This is faithful because binary64 carries 53 significand bits
while posit32 tops out at 27, so each intermediate is exact to well within half
a posit ULP.

Positive posit(n, es) values are  x = 2^(k*2^es + e) * (1+f)  where the
run-length-encoded regime k occupies  L(k) = k+2 bits for k >= 0 and -k+1 bits
for k < 0.  The fraction field therefore receives

    m(s) = max(0, n - 1 - es - L(floor(s / 2^es)))

bits at binary scale s, giving the characteristic V-shaped precision profile
with its vertex at magnitude 1.
"""
from __future__ import annotations
import numpy as np

__all__ = ["Posit", "Float", "PositArray"]


class Posit:
    """A posit format. Call ``quantize`` to round values onto its grid.

    Parameters
    ----------
    nbits : int
        Total width, e.g. 16 or 32.
    es : int
        Exponent bits. The 2022 posit standard fixes ``es = 2``.

    Examples
    --------
    >>> p = Posit(32, es=2)
    >>> int(p.bits_at(1.0))          # peak precision at unit magnitude
    27
    >>> int(p.bits_at(2.0 ** 40))    # one bit lost per four octaves
    17
    """

    def __init__(self, nbits: int = 32, es: int = 2):
        if nbits < 4:
            raise ValueError("nbits must be at least 4")
        self.n, self.es = nbits, es
        self.two_es = 1 << es
        self.max_scale = (nbits - 2) * self.two_es
        self.maxpos = 2.0 ** self.max_scale
        self.minpos = 2.0 ** (-self.max_scale)
        self._build_extreme_table()

    def __repr__(self) -> str:
        return f"Posit(nbits={self.n}, es={self.es})"

    # ------------------------------------------------------------------
    def _build_extreme_table(self) -> None:
        """Tabulate values whose regime is long enough to truncate the
        exponent field. There are only a few dozen, so exhaustive is fine."""
        n, es = self.n, self.es
        vals = []
        for k in range(-(n - 2), n - 1):
            L = k + 2 if k >= 0 else -k + 1
            r = n - 1 - L
            if r >= es or r < 0:
                continue
            step = 1 << (es - max(0, r))
            for j in range(0, self.two_es, step):
                sc = k * self.two_es + j
                if abs(sc) <= self.max_scale:
                    vals.append(2.0 ** sc)

        # Bulk = every regime that leaves the exponent field intact (r >= es).
        k_hi = n - 3 - es
        k_lo = -(n - 2 - es)
        self.bulk_hi = 2.0 ** ((k_hi + 1) * self.two_es)   # exclusive
        self.bulk_lo = 2.0 ** (k_lo * self.two_es)         # inclusive

        # The bulk endpoints must also be reachable from the extreme side,
        # otherwise a value just below bulk_lo cannot round up to it.
        vals += [self.maxpos, self.minpos, self.bulk_hi, self.bulk_lo]
        self.extreme = np.unique(np.array(vals, dtype=np.float64))

    # ------------------------------------------------------------------
    def frac_bits(self, scale):
        """Fraction bits available at binary scale ``scale``."""
        scale = np.asarray(scale)
        k = np.floor_divide(scale, self.two_es)
        L = np.where(k >= 0, k + 2, -k + 1)
        return np.maximum(0, self.n - 1 - L - self.es)

    def bits_at(self, x):
        """Fraction bits the format holds for values of magnitude ``x``."""
        a = np.abs(np.asarray(x, dtype=np.float64))
        a = np.where(a == 0, 1.0, a)
        sc = np.floor(np.log2(np.clip(a, self.minpos, self.maxpos))).astype(np.int64)
        return self.frac_bits(sc)

    def unit_roundoff(self, x):
        """Relative rounding error bound at magnitude ``x``."""
        return 2.0 ** (-self.bits_at(x).astype(np.float64) - 1.0)

    # ------------------------------------------------------------------
    def quantize(self, x):
        """Round to nearest representable posit, ties to even.

        Posits saturate rather than overflowing: values above maxpos become
        maxpos and values below minpos become minpos. There is no Inf and no
        NaN, and only exact zero maps to zero.
        """
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.zeros_like(x)

        sign = np.sign(x)
        a = np.abs(x)
        nz = a > 0
        if not np.any(nz):
            return out.item() if scalar else out
        a = np.clip(a, self.minpos, self.maxpos)

        # --- bulk: exponent field intact, round analytically ---
        bulk = nz & (a >= self.bulk_lo) & (a < self.bulk_hi)
        if np.any(bulk):
            ab = a[bulk]
            sc = np.floor(np.log2(ab)).astype(np.int64)
            sc = np.where(2.0 ** sc > ab, sc - 1, sc)          # log2 edge cases
            sc = np.where(2.0 ** (sc + 1) <= ab, sc + 1, sc)
            fb = self.frac_bits(sc)
            m = ab / 2.0 ** sc                                  # in [1, 2)
            scale = 2.0 ** fb
            mq = np.rint(m * scale) / scale                     # ties to even
            carry = mq >= 2.0                                   # rolled a binade
            if np.any(carry):
                sc2 = sc[carry] + 1
                sc[carry], fb[carry] = sc2, self.frac_bits(sc2)
                mq[carry] = 1.0
            out[bulk] = sign[bulk] * np.clip(mq * 2.0 ** sc,
                                             self.minpos, self.maxpos)

        # --- extreme: regime truncates the exponent, use the table ---
        ext = nz & ~bulk
        if np.any(ext):
            ae, tab = a[ext], self.extreme
            idx = np.clip(np.searchsorted(tab, ae), 1, len(tab) - 1)
            lo, hi = tab[idx - 1], tab[idx]
            out[ext] = sign[ext] * np.where(ae - lo <= hi - ae, lo, hi)

        return out.item() if scalar else out

    __call__ = quantize


class Float:
    """IEEE binary16/32/64 baseline exposing the same interface as Posit."""

    _DT = {16: np.float16, 32: np.float32, 64: np.float64}
    _FB = {16: 10, 32: 23, 64: 52}

    def __init__(self, nbits: int = 32):
        if nbits not in self._DT:
            raise ValueError("nbits must be 16, 32 or 64")
        self.n = nbits
        self.dt = self._DT[nbits]
        self.fb = self._FB[nbits]

    def __repr__(self) -> str:
        return f"Float(nbits={self.n})"

    def quantize(self, x):
        with np.errstate(over="ignore"):
            return np.asarray(x, dtype=np.float64).astype(self.dt).astype(np.float64)

    __call__ = quantize

    def bits_at(self, x):
        """Flat precision profile — this is precisely what posits give up."""
        return np.full(np.shape(x), self.fb)

    def unit_roundoff(self, x):
        return np.full(np.shape(x), 2.0 ** (-self.fb - 1))


class PositArray:
    """Convenience wrapper that rounds after every elementary operation.

    Lets existing NumPy code be run in posit arithmetic with minimal edits.
    Slower than calling ``Posit.quantize`` directly on whole arrays, so the
    experiments in this repository use the fast path instead.

    Examples
    --------
    >>> import numpy as np
    >>> x = PositArray([1.0, 2.0, 3.0], Posit(16, 2))
    >>> y = (x * 1.1 + 0.5) / 3.0
    >>> isinstance(y, PositArray)
    True
    """

    __array_priority__ = 100.0

    def __init__(self, values, fmt: Posit | Float | None = None):
        if isinstance(values, PositArray):
            fmt = fmt or values.fmt
            values = values.value
        self.fmt = fmt or Posit(32, 2)
        self.value = self.fmt.quantize(np.asarray(values, dtype=np.float64))

    def _wrap(self, raw):
        return PositArray(raw, self.fmt)

    @staticmethod
    def _raw(other):
        return other.value if isinstance(other, PositArray) else other

    def __add__(self, o):      return self._wrap(self.value + self._raw(o))
    def __radd__(self, o):     return self._wrap(self._raw(o) + self.value)
    def __sub__(self, o):      return self._wrap(self.value - self._raw(o))
    def __rsub__(self, o):     return self._wrap(self._raw(o) - self.value)
    def __mul__(self, o):      return self._wrap(self.value * self._raw(o))
    def __rmul__(self, o):     return self._wrap(self._raw(o) * self.value)
    def __truediv__(self, o):  return self._wrap(self.value / self._raw(o))
    def __rtruediv__(self, o): return self._wrap(self._raw(o) / self.value)
    def __neg__(self):         return self._wrap(-self.value)
    def __matmul__(self, o):   return self._wrap(self.value @ self._raw(o))
    def __getitem__(self, k):  return self._wrap(self.value[k])
    def __len__(self):         return len(self.value)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)

    def __repr__(self) -> str:
        return f"PositArray({self.value!r}, {self.fmt!r})"
