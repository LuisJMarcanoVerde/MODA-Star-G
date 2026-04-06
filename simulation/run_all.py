"""
MODA* Simulation Study v0.6 — Softmax Gating — Main Runner
Run: python run_all.py

Tests Softmax temperature k in {1, 2, 3, 5, 8, 10}.
Key question: which k maximises AUC across all four scenarios?

Expected behavior:
  k=1:  near-uniform weights, similar to base
  k=3:  moderate amplification, best balance
  k=5:  strong amplification, O4 improves, O1-O3 stay strong
  k=10: near-argmax, O4 optimal, possible noise sensitivity

Outputs in results/:
  simulation_results_v06.csv
  table_sensitivity_k.csv      <- KEY TABLE for paper
  table_by_scenario.csv
  figure_sensitivity_k.png     <- AUC vs k for each scenario
  figure_heatmap_v06.png
  figure_gating_weights_O4.png <- how weights change with k in O4
  figure_score_distributions.png
  summary_v06.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from simulation import (run_full_simulation, GENERATORS,
                         CONTAMINATION_RATES, N_OBS, K_VALUES,
                         generate_O1, generate_O2,
                         generate_O3, generate_O4)
from moda_star import moda_star, adaptive_threshold, softmax_gating
from moda_star import score_mcd, score_pena, score_gower, score_sdc

os.makedirs('results', exist_ok=True)

N_REPS = 100   # set to 10 for quick test

SCENARIO_LABELS = {
    'O1_isolated_numerical':  'O1 — Isolated Numerical',
    'O2_clustered_numerical': 'O2 — Clustered Numerical',
    'O3_moderate_mixed':      'O3 — Mixed Moderate',
    'O4_pure_categorical':    'O4 — Pure Categorical',
}
SCENARIOS  = list(GENERATORS.keys())
GEN_FUNCS  = [generate_O1, generate_O2, generate_O3, generate_O4]
K_COLORS   = {1:'#aec6cf', 2:'#7fb3d3', 3:'#1a4a8a',
               5:'#f39c12', 8:'#e67e22', 10:'#c0392b'}

# ─── Run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("MODA*-G  v0.6 — Softmax Gating Sensitivity Analysis")
print(f"n={N_OBS} | {N_REPS} reps per cell")
print(f"k values tested: {K_VALUES}")
print("=" * 70)

df = run_full_simulation(n_reps=N_REPS)
df.to_csv('results/simulation_results_v06.csv', index=False)
print(f"\nResults → results/simulation_results_v06.csv")

# ─── Sensitivity table — k vs scenario (eps<=0.15) ────────────────────────────
sub15 = df[df['eps'] <= 0.15]

sens_rows = []
for k in K_VALUES:
    row = {' k': k}
    for sc in SCENARIOS:
        sub = sub15[sub15['scenario']==sc]
        vals = sub[f'G_k{k}_auc'].dropna().values
        row[SCENARIO_LABELS[sc][:4]] = round(float(np.mean(vals)),3) \
            if len(vals) else float('nan')
    base_vals = sub15['BASE_auc'].dropna().values
    row['Mean'] = round(float(np.mean(
        [sub15[sub15['scenario']==sc][f'G_k{k}_auc'].dropna().values.mean()
         for sc in SCENARIOS])), 3)
    sens_rows.append(row)

# Add base row
base_row = {' k': 'base'}
for sc in SCENARIOS:
    vals = sub15[sub15['scenario']==sc]['BASE_auc'].dropna().values
    base_row[SCENARIO_LABELS[sc][:4]] = round(float(np.mean(vals)),3)
base_row['Mean'] = round(float(np.mean(
    [sub15[sub15['scenario']==sc]['BASE_auc'].dropna().values.mean()
     for sc in SCENARIOS])), 3)
sens_rows.insert(0, base_row)

t_sens = pd.DataFrame(sens_rows)
t_sens.to_csv('results/table_sensitivity_k.csv', index=False)

print("\n" + "─"*70)
print("SENSITIVITY TABLE — Mean AUC-ROC by k (eps <= 0.15)")
print("─"*70)
print(t_sens.to_string(index=False))

# ─── By scenario full table ───────────────────────────────────────────────────
rows = []
for sc in SCENARIOS:
    sub = df[df['scenario']==sc]
    for eps in CONTAMINATION_RATES:
        sub_e = sub[sub['eps']==eps]
        row   = {'Scenario': SCENARIO_LABELS[sc][:4], 'eps': eps}
        for k in K_VALUES:
            vals = sub_e[f'G_k{k}_auc'].dropna().values
            row[f'k={k}'] = round(float(np.mean(vals)),3) if len(vals) else float('nan')
        vals_b = sub_e['BASE_auc'].dropna().values
        row['base'] = round(float(np.mean(vals_b)),3) if len(vals_b) else float('nan')
        vals_if = sub_e['IF_auc'].dropna().values
        row['IF'] = round(float(np.mean(vals_if)),3) if len(vals_if) else float('nan')
        rows.append(row)
pd.DataFrame(rows).to_csv('results/table_by_scenario.csv', index=False)

# ─── Figure 1: Sensitivity — AUC vs k ────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('MODA*-G v0.6 — AUC-ROC vs Temperature k  (eps <= 0.15)',
             fontsize=13, fontweight='bold')

x_k = K_VALUES
for ai, sc in enumerate(SCENARIOS):
    ax  = axes[ai]
    sub = sub15[sub15['scenario']==sc]

    # Gating lines for each eps
    for eps in CONTAMINATION_RATES:
        sub_e = sub[sub['eps']==eps]
        y = [sub_e[f'G_k{k}_auc'].dropna().values.mean()
             if len(sub_e[f'G_k{k}_auc'].dropna()) > 0 else np.nan
             for k in K_VALUES]
        alpha_line = 0.4 + 0.12 * CONTAMINATION_RATES.index(eps)
        ax.plot(x_k, y, 'o-', alpha=alpha_line,
                color='#1a4a8a', linewidth=1.5,
                label=f'ε={eps:.0%}')

    # Base line
    base_val = sub['BASE_auc'].dropna().values.mean()
    ax.axhline(base_val, color='gray', linestyle='--',
               linewidth=1.5, label=f'Base={base_val:.3f}')

    # IF line
    if_val = sub['IF_auc'].dropna().values.mean()
    ax.axhline(if_val, color='#c0392b', linestyle=':',
               linewidth=1.5, label=f'IF={if_val:.3f}')

    ax.set_title(SCENARIO_LABELS[sc], fontsize=10)
    ax.set_xlabel('Temperature k', fontsize=9)
    ax.set_ylabel('AUC-ROC', fontsize=9)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.2, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figure_sensitivity_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure → results/figure_sensitivity_k.png")

# ─── Figure 2: Heatmap — k vs scenario ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('MODA*-G — Mean AUC-ROC by Temperature k and Scenario (eps<=0.15)',
             fontsize=12, fontweight='bold')

matrix = np.zeros((len(K_VALUES)+1, len(SCENARIOS)))
row_labels = ['base'] + [f'k={k}' for k in K_VALUES]

for j, sc in enumerate(SCENARIOS):
    sub = sub15[sub15['scenario']==sc]
    base_v = sub['BASE_auc'].dropna().values
    matrix[0, j] = np.mean(base_v) if len(base_v) else np.nan
    for i, k in enumerate(K_VALUES):
        vals = sub[f'G_k{k}_auc'].dropna().values
        matrix[i+1, j] = np.mean(vals) if len(vals) else np.nan

im = ax.imshow(matrix, vmin=0.3, vmax=1.0, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(SCENARIOS)))
ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in SCENARIOS], fontsize=9)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)

for i in range(len(row_labels)):
    for j in range(len(SCENARIOS)):
        v = matrix[i, j]
        if not np.isnan(v):
            color = 'black' if v > 0.6 else 'white'
            weight = 'bold' if i == 0 else 'normal'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)

# Highlight base row
ax.add_patch(plt.Rectangle((-0.5,-0.5), len(SCENARIOS), 1,
                             fill=False, edgecolor='gray',
                             linewidth=2, linestyle='--'))

plt.colorbar(im, ax=ax, label='AUC-ROC', fraction=0.03)
plt.tight_layout()
plt.savefig('results/figure_heatmap_v06.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure → results/figure_heatmap_v06.png")

# ─── Figure 3: Gating weights in O4 as k increases ───────────────────────────
fig, axes = plt.subplots(1, len(K_VALUES), figsize=(18, 4))
fig.suptitle('MODA*-G — Mean Gating Weights in O4 by Temperature k (eps=0.05)',
             fontsize=12, fontweight='bold')

X_num_ex, X_cat_ex, labels_ex = generate_O4(n=300, eps=0.05, seed=42)
eng_names  = ['MCD', 'Peña', 'Gower', 'SDC']
eng_colors = ['#1a4a8a','#c0392b','#1a6b3c','#d4a830']

for ai, k_val in enumerate(K_VALUES):
    ax = axes[ai]
    _, s_mcd, s_pena, s_gow, s_sdc, gw, mw = moda_star(
        X_num_ex, X_cat_ex,
        use_gating=True, k_temp=float(k_val),
        return_components=True)

    # Mean weights for outliers vs normals
    w_out  = gw[:, labels_ex==1].mean(axis=1)
    w_norm = gw[:, labels_ex==0].mean(axis=1)

    x_pos = np.arange(4)
    w_bar = 0.35
    ax.bar(x_pos - w_bar/2, w_out,  w_bar,
           color=eng_colors, alpha=0.9, label='Outlier')
    ax.bar(x_pos + w_bar/2, w_norm, w_bar,
           color=eng_colors, alpha=0.4, label='Normal')
    ax.set_title(f'k = {k_val}', fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(eng_names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Mean weight', fontsize=9) if ai==0 else None
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    ax.grid(axis='y', alpha=0.3)
    if ai == 0:
        ax.legend(fontsize=8)

    # Annotate SDC weight for outliers
    ax.text(3 - w_bar/2, w_out[3] + 0.02,
            f'{w_out[3]:.2f}',
            ha='center', fontsize=8, fontweight='bold',
            color='#d4a830')

plt.tight_layout()
plt.savefig('results/figure_gating_weights_O4.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure → results/figure_gating_weights_O4.png")

# ─── Figure 4: Score distributions for best k ────────────────────────────────
# Determine best k for each scenario (eps<=0.15)
best_k_per_sc = {}
for sc in SCENARIOS:
    sub = sub15[sub15['scenario']==sc]
    aucs = {k: sub[f'G_k{k}_auc'].dropna().values.mean() for k in K_VALUES}
    best_k_per_sc[sc] = max(aucs, key=aucs.get)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MODA*-G v0.6 — Score Distributions at Optimal k (eps=0.05)',
             fontsize=13, fontweight='bold')

for ai, (sc, gf) in enumerate(zip(SCENARIOS, GEN_FUNCS)):
    ax   = axes.flatten()[ai]
    best_k = best_k_per_sc[sc]
    X_num, X_cat, labels = gf(n=500, eps=0.05, seed=42)
    scores, *_, gw, mw = moda_star(X_num, X_cat,
                                    use_gating=True,
                                    k_temp=float(best_k),
                                    return_components=True)
    tau  = adaptive_threshold(scores)
    bins = np.linspace(0,1,30)
    ax.hist(scores[labels==0], bins=bins, alpha=0.6,
            color='#1a4a8a', label='Normal', density=True)
    ax.hist(scores[labels==1], bins=bins, alpha=0.8,
            color='#c0392b', label='Outlier', density=True)
    ax.axvline(tau, color='#b8860b', linestyle='--', lw=2,
               label=f'τ*={tau:.3f}')
    ax.set_title(f"{SCENARIO_LABELS[sc]}  [best k={best_k}]", fontsize=10)
    ax.set_xlabel('MODA*-G Score', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figure_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure → results/figure_score_distributions.png")

# ─── Final summary ────────────────────────────────────────────────────────────
lines = []
def pr(s): print(s); lines.append(s)

pr("\n" + "="*70)
pr("MODA*-G  v0.6 — FINAL SUMMARY")
pr("="*70)

pr("\nSENSITIVITY TABLE — Mean AUC-ROC (eps <= 0.15)")
pr(t_sens.to_string(index=False))

pr("\nBEST k PER SCENARIO (eps<=0.15):")
for sc in SCENARIOS:
    sub = sub15[sub15['scenario']==sc]
    aucs = {}
    for k in K_VALUES:
        vals = sub[f'G_k{k}_auc'].dropna().values
        aucs[k] = np.mean(vals) if len(vals) else float('nan')
    best_k  = max(aucs, key=lambda kk: aucs[kk] if not np.isnan(aucs[kk]) else -1)
    best_auc = aucs[best_k]
    base_auc = sub['BASE_auc'].dropna().values.mean()
    pr(f"  {SCENARIO_LABELS[sc]:32s}: best k={best_k}  "
       f"AUC={best_auc:.3f}  base={base_auc:.3f}  "
       f"Delta={best_auc-base_auc:+.3f}")

pr("\nOVERALL BEST k (mean over all scenarios, eps<=0.15):")
overall = {}
for k in K_VALUES:
    vals = []
    for sc in SCENARIOS:
        v = sub15[sub15['scenario']==sc][f'G_k{k}_auc'].dropna().values
        if len(v): vals.append(np.mean(v))
    overall[k] = np.mean(vals) if vals else float('nan')
best_overall = max(overall, key=lambda kk: overall[kk] if not np.isnan(overall[kk]) else -1)
pr(f"  Best k overall: k={best_overall}  AUC={overall[best_overall]:.3f}")
for k in K_VALUES:
    pr(f"  k={k:2d}: {overall[k]:.3f}")

pr("\n✓ All outputs saved in results/")
pr("✓ Simulation complete — MODA*-G v0.6")

with open('results/summary_v06.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Summary → results/summary_v06.txt")
