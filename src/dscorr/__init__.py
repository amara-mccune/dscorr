"""dscorr: massive de Sitter correlators as finite Mellin-Barnes integrals (v1.0.0).

Tree-level single, double and triple exchange (``dscorr.tree``), the kernel identity
(``dscorr.kernels``), direct Schwinger-Keldysh time integration as the independent check
(``dscorr.direct_sk``), the closed-form squeezed coefficients with templates and the fit
(``dscorr.squeezed``) and the data layer (``dscorr.data``).

Sub-packages: the public modules (this package), the numerical routines
(``dscorr._engines``) and the access layer for the data tables (``dscorr.data``).
"""
from . import data, direct_sk, kernels, squeezed, tree
from ._validate import TESTED, ValidityWarning
from .direct_sk import double_exchange_direct, single_exchange_direct, triple_exchange_direct
from .kernels import (IdentityCheck, NumericVsClosedForm, kernel_identity_check,
                      kernel_identity_lhs, kernel_identity_rhs)
from .squeezed import M1, M2, M3, SqueezedFit, extract, signal, template
from .tree import (DoubleExchange, SingleExchange, TripleExchange, double_exchange,
                   single_exchange, triple_exchange)

__version__ = "1.0.0"

__all__ = [
    "single_exchange", "double_exchange", "triple_exchange",
    "SingleExchange", "DoubleExchange", "TripleExchange",
    "ValidityWarning", "TESTED",
    "M1", "M2", "M3", "template", "signal", "extract", "SqueezedFit",
    "single_exchange_direct", "double_exchange_direct", "triple_exchange_direct",
    "kernel_identity_lhs", "kernel_identity_rhs", "kernel_identity_check",
    "IdentityCheck", "NumericVsClosedForm",
    "kernels", "squeezed", "tree", "direct_sk", "data",
]
