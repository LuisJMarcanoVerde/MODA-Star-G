"""Competitor methods for MODA* v0.6 benchmark."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder


def _encode_mixed(X_num, X_cat):
    if X_cat is None or X_cat.ndim < 2 or X_cat.shape[1] == 0:
        return X_num
    X_enc = np.zeros_like(X_cat, dtype=float)
    for k in range(X_cat.shape[1]):
        le = LabelEncoder()
        X_enc[:, k] = le.fit_transform(
            X_cat[:, k].astype(str)).astype(float)
    return np.hstack([X_num, X_enc])


def score_isolation_forest(X_num, X_cat=None,
                            contamination=0.1, random_state=42):
    X   = _encode_mixed(X_num, X_cat)
    eps = float(np.clip(contamination, 0.001, 0.499))
    clf = IsolationForest(contamination=eps,
                          random_state=random_state, n_jobs=-1)
    clf.fit(X)
    raw = -clf.decision_function(X)
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros(len(X))


def score_lof(X_num, X_cat=None,
              n_neighbors=20, contamination=0.1):
    X   = _encode_mixed(X_num, X_cat)
    eps = float(np.clip(contamination, 0.001, 0.499))
    k   = min(n_neighbors, len(X) - 1)
    clf = LocalOutlierFactor(n_neighbors=k, contamination=eps)
    clf.fit_predict(X)
    raw = -clf.negative_outlier_factor_
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros(len(X))
