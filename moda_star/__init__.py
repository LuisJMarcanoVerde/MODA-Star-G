"""
MODA*-G — Mixed-Data Outlier Divergence Analysis with Softmax Gating
=====================================================================
Version 0.6  |  March 2026  |  Luis J. Marcano Verde

Public API
----------
    from moda_star import moda_star, adaptive_threshold, classify
    from moda_star import score_mcd, score_pena, score_gower, score_sdc
    from moda_star import softmax_gating

Quick start
-----------
    import numpy as np
    from moda_star import moda_star, adaptive_threshold

    X_num = np.random.randn(300, 4)           # 4 numerical features
    X_cat = np.random.randint(0, 3, (300, 2)) # 2 categorical features

    scores = moda_star(X_num, X_cat, k_temp=3.0)
    tau    = adaptive_threshold(scores)
    labels = (scores >= tau).astype(int)
"""

from .moda_star import (
    moda_star,
    adaptive_threshold,
    classify,
    score_mcd,
    score_pena,
    score_gower,
    score_sdc,
    softmax_gating,
    phi,
    robust_mad,
    robust_scale,
)

__version__  = "0.6.0"
__author__   = "Luis J. Marcano Verde"
__email__    = "luisjmarcanoverde@gmail.com"
__license__  = "MIT"

__all__ = [
    "moda_star",
    "adaptive_threshold",
    "classify",
    "score_mcd",
    "score_pena",
    "score_gower",
    "score_sdc",
    "softmax_gating",
    "phi",
    "robust_mad",
    "robust_scale",
]
