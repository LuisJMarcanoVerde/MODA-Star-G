"""
MODA* Simulation Study v0.6 — Softmax Gating
Tests k_temp in {1, 2, 3, 5, 8, 10} across four canonical scenarios.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from moda_star import (moda_star, score_mcd, score_pena,
                        score_gower, score_sdc, adaptive_threshold)
from competitors import score_isolation_forest, score_lof

# Temperature values to test
K_VALUES = [1, 2, 3, 5, 8, 10]


# ─── Data generators ─────────────────────────────────────────────────────────

def generate_O1(n=300, p=4, q=2, eps=0.05, seed=None):
    rng     = np.random.default_rng(seed)
    n_out   = max(1, int(np.floor(n * eps)))
    n_clean = n - n_out
    X_num_c = rng.multivariate_normal(np.zeros(p), np.eye(p), n_clean)
    v       = rng.standard_normal(p); v /= np.linalg.norm(v)
    X_num_o = rng.multivariate_normal(6.0*v, np.eye(p)*0.1, n_out)
    X_cat_c = rng.choice([0,1,2], size=(n_clean,q), p=[0.6,0.3,0.1])
    X_cat_o = rng.choice([0,1,2], size=(n_out,  q), p=[0.6,0.3,0.1])
    X_num   = np.vstack([X_num_c, X_num_o])
    X_cat   = np.vstack([X_cat_c, X_cat_o])
    labels  = np.array([0]*n_clean + [1]*n_out)
    idx     = rng.permutation(n)
    return X_num[idx], X_cat[idx], labels[idx]


def generate_O2(n=300, p=4, q=2, eps=0.10, seed=None):
    rng     = np.random.default_rng(seed)
    n_out   = max(2, int(np.floor(n * eps)))
    n_clean = n - n_out
    X_num_c = rng.multivariate_normal(np.zeros(p), np.eye(p), n_clean)
    X_num_o = rng.multivariate_normal(np.ones(p)*4.0, np.eye(p)*0.05, n_out)
    X_cat_c = rng.choice([0,1,2], size=(n_clean,q), p=[0.6,0.3,0.1])
    X_cat_o = rng.choice([0,1,2], size=(n_out,  q), p=[0.6,0.3,0.1])
    X_num   = np.vstack([X_num_c, X_num_o])
    X_cat   = np.vstack([X_cat_c, X_cat_o])
    labels  = np.array([0]*n_clean + [1]*n_out)
    idx     = rng.permutation(n)
    return X_num[idx], X_cat[idx], labels[idx]


def generate_O3(n=300, p=4, q=2, eps=0.05, seed=None):
    rng     = np.random.default_rng(seed)
    n_out   = max(1, int(np.floor(n * eps)))
    n_clean = n - n_out
    X_num_c = rng.multivariate_normal(np.zeros(p), np.eye(p), n_clean)
    v       = rng.standard_normal(p); v /= np.linalg.norm(v)
    X_num_o = rng.multivariate_normal(2.0*v, np.eye(p)*0.5, n_out)
    X_cat_c = rng.choice([0,1,2], size=(n_clean,q), p=[0.6,0.3,0.1])
    X_cat_o = np.full((n_out,q), 3, dtype=int)
    X_num   = np.vstack([X_num_c, X_num_o])
    X_cat   = np.vstack([X_cat_c, X_cat_o])
    labels  = np.array([0]*n_clean + [1]*n_out)
    idx     = rng.permutation(n)
    return X_num[idx], X_cat[idx], labels[idx]


def generate_O4(n=300, p=4, q=2, eps=0.05, seed=None):
    rng     = np.random.default_rng(seed)
    n_out   = max(1, int(np.floor(n * eps)))
    n_clean = n - n_out
    X_num_c = rng.multivariate_normal(np.zeros(p), np.eye(p), n_clean)
    X_num_o = rng.multivariate_normal(np.zeros(p), np.eye(p)*0.1, n_out)
    X_cat_c = rng.choice([0,1,2], size=(n_clean,q), p=[0.6,0.3,0.1])
    X_cat_o = np.full((n_out,q), 99, dtype=int)
    X_num   = np.vstack([X_num_c, X_num_o])
    X_cat   = np.vstack([X_cat_c, X_cat_o])
    labels  = np.array([0]*n_clean + [1]*n_out)
    idx     = rng.permutation(n)
    return X_num[idx], X_cat[idx], labels[idx]


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(scores, labels):
    out = {}
    if len(np.unique(labels)) > 1:
        try:
            out['auc'] = float(roc_auc_score(labels, scores))
        except Exception:
            out['auc'] = 0.5
    else:
        out['auc'] = float('nan')
    tau  = adaptive_threshold(scores)
    pred = (scores >= tau).astype(int)
    tp   = int(((pred==1)&(labels==1)).sum())
    fp   = int(((pred==1)&(labels==0)).sum())
    fn   = int(((pred==0)&(labels==1)).sum())
    tn   = int(((pred==0)&(labels==0)).sum())
    out['tpr'] = tp/(tp+fn) if (tp+fn)>0 else 0.0
    out['fpr'] = fp/(fp+tn) if (fp+tn)>0 else 0.0
    return out


# ─── Scenario runner ──────────────────────────────────────────────────────────
GENERATORS = {
    'O1_isolated_numerical':  generate_O1,
    'O2_clustered_numerical': generate_O2,
    'O3_moderate_mixed':      generate_O3,
    'O4_pure_categorical':    generate_O4,
}

CONTAMINATION_RATES = [0.01, 0.05, 0.10, 0.15, 0.20]
N_REPS  = 100
N_OBS   = 300


def run_scenario(scenario_name, generator, eps, n_reps=N_REPS, n=N_OBS):
    """
    For each repetition, compute:
    - MODA*-G for each k in K_VALUES
    - MODA* base (linear, v0.3)
    - IF, LOF
    - Individual engines
    """
    method_keys = ([f'G_k{k}' for k in K_VALUES] +
                   ['BASE', 'IF', 'LOF',
                    'SDN_mcd', 'SDN_pena', 'SDG_mad', 'SDC'])
    records = {m: {'auc':[], 'tpr':[], 'fpr':[]} for m in method_keys}

    for rep in range(n_reps):
        seed = rep * 1000 + int(eps * 100)
        X_num, X_cat, labels = generator(n=n, eps=eps, seed=seed)
        if labels.sum() == 0:
            continue

        # MODA*-G for each temperature k
        try:
            # Compute engine scores once
            sc_base, s_mcd, s_pena, s_gow, s_sdc, _, _ = moda_star(
                X_num, X_cat, use_gating=False, return_components=True)

            # Base linear
            m = compute_metrics(sc_base, labels)
            for kk in ['auc','tpr','fpr']:
                records['BASE'][kk].append(m[kk])

            # Individual engines
            for name, s in [('SDN_mcd', s_mcd), ('SDN_pena', s_pena),
                             ('SDG_mad', s_gow), ('SDC', s_sdc)]:
                m2 = compute_metrics(s, labels)
                for kk in ['auc','tpr','fpr']:
                    records[name][kk].append(m2[kk])

            # Gating for each k
            for k_val in K_VALUES:
                sc_g, *_ = moda_star(
                    X_num, X_cat,
                    use_gating=True, k_temp=float(k_val),
                    return_components=True)
                m = compute_metrics(sc_g, labels)
                for kk in ['auc','tpr','fpr']:
                    records[f'G_k{k_val}'][kk].append(m[kk])

        except Exception:
            pass

        # Isolation Forest
        try:
            s_if = score_isolation_forest(
                X_num, X_cat, contamination=min(eps, 0.499))
            m = compute_metrics(s_if, labels)
            for kk in ['auc','tpr','fpr']:
                records['IF'][kk].append(m[kk])
        except Exception:
            pass

        # LOF
        try:
            s_lof = score_lof(
                X_num, X_cat, contamination=min(eps, 0.499))
            m = compute_metrics(s_lof, labels)
            for kk in ['auc','tpr','fpr']:
                records['LOF'][kk].append(m[kk])
        except Exception:
            pass

    summary = {'scenario': scenario_name, 'eps': eps, 'n_reps': n_reps}
    for method in method_keys:
        for metric in ['auc','tpr','fpr']:
            vals = [v for v in records[method][metric] if not np.isnan(v)]
            summary[f'{method}_{metric}']     = np.mean(vals) if vals else float('nan')
            summary[f'{method}_{metric}_std'] = np.std(vals)  if vals else float('nan')
    return summary


def run_full_simulation(n_reps=N_REPS):
    results = []
    total   = len(GENERATORS) * len(CONTAMINATION_RATES)
    done    = 0
    for scenario_name, generator in GENERATORS.items():
        for eps in CONTAMINATION_RATES:
            done += 1
            print(f"[{done}/{total}] {scenario_name} | "
                  f"eps={eps:.2f} | {n_reps} reps...",
                  end=' ', flush=True)
            r = run_scenario(scenario_name, generator, eps, n_reps)
            results.append(r)
            # Show key k values and base
            k3   = r.get('G_k3_auc',   float('nan'))
            k5   = r.get('G_k5_auc',   float('nan'))
            k10  = r.get('G_k10_auc',  float('nan'))
            base = r.get('BASE_auc',   float('nan'))
            print(f"k3={k3:.3f}  k5={k5:.3f}  k10={k10:.3f}  base={base:.3f}")
    return pd.DataFrame(results)
