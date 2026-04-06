MODA*-G — Mixed-Data Outlier Divergence Analysis with Softmax Gating
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)
Author: Luis J. Marcano Verde · Independent Statistical Consultant · New York, NY  
Paper: Zenodo preprint
---
What is MODA*-G?
MODA*-G is a novel unsupervised anomaly detection framework for tabular
data containing both numerical and categorical variables.
Most outlier detection methods (Isolation Forest, LOF, COPOD, EIF) require
encoding categorical features as numbers — distorting the metric structure.
MODA*-G operates natively in $\mathbb{R}^p \times \mathcal{C}^q$
without any encoding.
Four statistical engines
Engine	Type	Detection target
SDN_mcd	Robust MCD Mahalanobis	Isolated numerical outliers
SDN_peña	Directional kurtosis (Peña & Prieto 2001)	Clustered numerical outliers
SDG_mad	Gower-MAD distance	Mixed numerical + categorical
SDC	Categorical entropy score	Pure categorical anomalies
Softmax Gating (key innovation)
Instead of fixed linear weights, MODA*-G computes observation-level
adaptive weights via softmax:
$$w_i(x; k) = \frac{\exp(k \cdot S_i(x))}{\sum_j \exp(k \cdot S_j(x))}$$
$$\text{MODA*-G}(x; k) = \sum_i w_i(x; k) \cdot S_i(x)$$
Temperature $k=3$ is the recommended default. When one engine strongly
dominates (e.g. SDC scores 0.95 for a categorical outlier), the softmax
automatically amplifies it — without any heuristic rules.
---
Results
Monte Carlo Simulation (Section 6 of paper)
Four canonical outlier types, $n=300$, $p=4$, $q=2$, 100 replicates, $\varepsilon \leq 0.15$:
Scenario	MODA*-G (k=3)	IF	LOF	Δ vs IF
O1 — Isolated numerical	0.957	0.903	0.659	+0.054
O2 — Clustered numerical	0.993	0.957	0.670	+0.036
O3 — Mixed moderate	0.986	0.930	0.622	+0.056
O4 — Pure categorical	0.977	0.855	0.577	+0.122
Mean	0.978	0.911	0.623	+0.067
Real-Data Benchmark — Annthyroid (Section 7 of paper)
$n=7{,}200$, $p=6$ numerical $+$ $q=15$ categorical, $7.42%$ outliers.  
50 bootstrap replicates, $n_\text{sub}=2{,}000$.  
Dataset: Campos et al. (2016), LMU benchmark version.
Method	AUC-ROC	±	vs MODA*-G
MODA*-G (k=3)	0.939	0.010	— ★
Isolation Forest	0.632	0.021	−0.307
LOF	0.620	0.023	−0.320
CatBoost-AD	0.554	0.024	−0.385
COPOD	0.553	0.094	−0.386
EIF	0.496	0.064	−0.443
Best of Campos et al. (2016)	~0.750	—	−0.189
> MODA\*-G surpasses the best method reported by Campos et al. (2016)
> by **+0.189 AUC points**.
---
Formal guarantees
Five theorems established in the paper:
Result	Property
Theorem 1	Consistency: $\text{MODA*-G}_n \xrightarrow{P} \text{MODA*-G}$
Theorem 2	Breakdown point: $\varepsilon^* = 50%$
Theorem 3	Partial affine equivariance
Theorem 4	Score comparability across engines
Theorem 5	Stochastic dominance over any individual engine
Proposition 1	Softmax gating weight → 1 as $k \to \infty$
Proposition 2	Convergence rate $O_P(n^{-1/2})$
---
Installation
```bash
git clone https://github.com/luisjmarcano/MODA-Star-G.git
cd MODA-Star-G
pip install -r requirements.txt
```
No package installation required — import directly from the cloned repo.
---
Quick start
```python
import numpy as np
import sys
sys.path.insert(0, 'path/to/MODA-Star-G')

from moda_star import moda_star, adaptive_threshold

# Example: 300 observations, 4 numerical + 2 categorical features
X_num = np.random.randn(300, 4)
X_cat = np.random.randint(0, 3, (300, 2))

# Compute MODA*-G scores (higher = more anomalous)
scores = moda_star(X_num, X_cat, k_temp=3.0)

# Adaptive threshold: median + 3*MAD
tau    = adaptive_threshold(scores)
labels = (scores >= tau).astype(int)

print(f"Detected {labels.sum()} outliers ({100*labels.mean():.1f}%)")
```
---
Reproducing paper results
Section 6 — Monte Carlo Simulation
```bash
cd simulation
python run_all.py
# Results → simulation/results/
# Runtime: ~20-40 minutes (100 reps × 4 scenarios × 5 eps × 6 k values)
```
For a quick test (10 reps):
```bash
# Edit simulation/run_all.py: change N_REPS = 100 to N_REPS = 10
python run_all.py
# Runtime: ~3-5 minutes
```
Section 7 — Annthyroid Benchmark
```bash
cd benchmark
python run_benchmark.py
# Results → benchmark/results/
# Runtime: ~15-25 minutes (50 reps × 6 methods)
```
For a quick test (10 reps):
```bash
# Edit benchmark/run_benchmark.py: change N_BOOT = 50 to N_BOOT = 10
python run_benchmark.py
# Runtime: ~3-5 minutes
```
---
Repository structure
```
MODA-Star-G/
│
├── moda_star/                   ← Core framework
│   ├── __init__.py              ← Public API
│   └── moda_star.py             ← MODA*-G v0.6 implementation
│       ├── score_mcd()          ← Engine I: MCD Mahalanobis
│       ├── score_pena()         ← Engine II: Directional kurtosis
│       ├── score_gower()        ← Engine III: Gower-MAD distance
│       ├── score_sdc()          ← Engine IV: Categorical entropy
│       ├── softmax_gating()     ← Softmax Gating mechanism
│       └── moda_star()          ← Combined scorer
│
├── simulation/                  ← Monte Carlo study (Section 6)
│   ├── moda_star.py             ← Framework copy (self-contained)
│   ├── simulation.py            ← Four scenarios O1-O4
│   ├── competitors.py           ← IF and LOF
│   ├── run_all.py               ← Main runner + figures
│   └── results/                 ← Output directory (auto-created)
│
├── benchmark/                   ← Real-data study (Section 7)
│   ├── run_benchmark.py         ← Annthyroid benchmark
│   ├── Annthyroid_07.arff       ← Dataset (Campos et al. 2016)
│   └── results/                 ← Output directory (auto-created)
│
├── paper/
│   └── MODA_G_final.tex         ← LaTeX source (arXiv/Zenodo version)
│
├── notebooks/                   ← Jupyter notebooks (coming soon)
│
├── requirements.txt
├── LICENSE                      ← MIT
├── CITATION.cff                 ← Machine-readable citation
├── .gitignore
└── README.md
```
---
Citation
If you use MODA*-G in your research, please cite:
```bibtex
@misc{marcano2026modag,
  author       = {Marcano Verde, Luis J.},
  title        = {{MODA*-G}: A Robust Mixed-Data Outlier Detection
                  Framework with Attention-based Softmax Gating},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX},
}
```
---
References
Campos et al. (2016) — On the evaluation of unsupervised outlier detection.
Data Mining and Knowledge Discovery, 30(4), 891–927.
Hariri et al. (2019) — Extended Isolation Forest.
IEEE TKDE, 33(4), 1479–1489.
Li et al. (2020) — COPOD: Copula-Based Outlier Detection.
IEEE ICDM, 1118–1123.
Liu et al. (2008) — Isolation Forest.
IEEE ICDM, 413–422.
Marcano & Fermín (2013) — Comparación de métodos de detección de datos
anómalos multivariantes mediante un estudio de simulación.
Saber, 25(2), 192–201.
Peña & Prieto (2001) — Multivariate outlier detection and robust
covariance matrix estimation. Technometrics, 43(3), 286–310.
Rousseeuw (1985) — Multivariate estimation with high breakdown point.
Mathematical Statistics and Applications, 8, 283–297.
---
License
MIT License — see LICENSE
