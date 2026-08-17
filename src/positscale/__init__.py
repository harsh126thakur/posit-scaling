"""positscale -- optimal diagonal scaling for posit arithmetic.

Posit precision is tapered: it peaks at unit magnitude and decays away from it.
Diagonal scaling is therefore a genuine numerical lever for posits in a way it
is not for IEEE 754, where the precision profile is flat.

Quick start
-----------
>>> import numpy as np
>>> from positscale import Posit, scale_logmean, apply_scaling
>>> A = np.diag([2.0 ** 20, 2.0 ** -20]) @ np.ones((2, 2))
>>> d, e = scale_logmean(A)
>>> As, D1, D2 = apply_scaling(A, d, e)
"""
from .quantizer import Posit, Float, PositArray
from .scaling import (log_magnitudes, apply_scaling, scale_none, scale_ruiz,
                      scale_logmean, scale_lp_inf, scale_hybrid,
                      max_mean_cycle, removable_fraction, SCALINGS)
from .matrices import gen_matrix, FAMILIES
from .kernels import lu_solve, backward_error, forward_error

__version__ = "0.1.0"

__all__ = [
    "Posit", "Float", "PositArray",
    "log_magnitudes", "apply_scaling", "scale_none", "scale_ruiz",
    "scale_logmean", "scale_lp_inf", "scale_hybrid", "max_mean_cycle",
    "removable_fraction", "SCALINGS",
    "gen_matrix", "FAMILIES",
    "lu_solve", "backward_error", "forward_error",
    "__version__",
]
