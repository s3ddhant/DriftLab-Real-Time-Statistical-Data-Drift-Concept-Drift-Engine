# DriftLab: Real-Time Statistical Distribution Shift & Data Drift Auditor

**Academic Admissions Portfolio Project**  
*M.Sc. Data Science & Machine Learning / Applied Artificial Intelligence*  
*Target Universities: Technical University of Munich (TUM), LMU Munich, TU Berlin, RWTH Aachen*

---

## 1. Executive Abstract & Research Motivation

In production machine learning systems, statistical invariants assumed during offline training rarely hold indefinitely. Unmonitored non-stationary phenomena—namely **Covariate Shift** $P_t(X) \neq P_0(X)$ and **Concept Drift** $P_t(Y \mid X) \neq P_0(Y \mid X)$—lead to silent model degradation, catastrophic prediction failures, and financial liability.

Traditional heuristics (e.g. monitoring mean and standard deviation) frequently fail to capture higher-order geometric shifts, multimodality, or subtle quantile expansions. **DriftLab** introduces an academic-grade, vectorized non-parametric audit architecture that simultaneously tracks:
1. **Infinitesimal supremum divergence** via the Two-Sample Kolmogorov-Smirnov test with Bonferroni correction for multi-hypothesis testing.
2. **Physical transport effort** via the Wasserstein-1 (Earth Mover's) metric under the Kantorovich-Rubinstein duality.
3. **Information-theoretic bucket stability** via the Population Stability Index (PSI) derived from symmetric Kullback-Leibler divergence.

The framework operates in real time with zero external API keys, zero cloud dependencies, and a modern **Neo-Bento Dark Mode** visual interface.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Two-Sample Kolmogorov-Smirnov (K-S) Test
For a continuous feature $X \in \mathbb{R}$, let $F_0(x)$ denote the empirical cumulative distribution function (eCDF) of the reference baseline (sample size $n$), and $F_t(x)$ denote the eCDF of the sliding production window (sample size $m$):

$$F_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}_{(-\infty, x]}(X_i)$$

The two-sample Kolmogorov-Smirnov test statistic $D_{n, m}$ is defined as the supremum norm between both eCDFs:

$$D_{n, m} = \sup_{x \in \mathbb{R}} |F_n(x) - F_m(x)|$$

Under the null hypothesis $H_0: P_0(X) = P_t(X)$, the asymptotic distribution of $\sqrt{\frac{nm}{n+m}} D_{n,m}$ converges to the Kolmogorov distribution.

#### Bonferroni Family-Wise Error Rate (FWER) Correction
When testing $M$ parallel continuous features simultaneously:

$$\alpha_{\text{adjusted}} = \frac{\alpha_{\text{nominal}}}{M} = \frac{0.05}{4} = 0.0125$$

This guarantees that the probability of at least one false positive drift detection across all monitored dimensions is bounded by $\le \alpha_{\text{nominal}}$.

---

### 2.2 Wasserstein-1 Distance (Earth Mover's Distance)
While the KS test is scale-invariant and bounded in $[0, 1]$, it provides no information about *how far in feature space* probability mass has migrated. The 1st Wasserstein metric measures the minimum cost of transporting probability distribution $u$ into $v$:

$$\mathcal{W}_1(u, v) = \inf_{\gamma \in \Pi(u, v)} \mathbb{E}_{(X, Y) \sim \gamma}[\|X - Y\|]$$

By the **Kantorovich-Rubinstein Duality Theorem**, for one-dimensional continuous measures:

$$\mathcal{W}_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| \, dx$$

where $U(x)$ and $V(x)$ are the respective cumulative distribution functions. DriftLab normalizes $\mathcal{W}_1$ by the baseline standard deviation $\sigma_0$ ($\mathcal{W}_1 / \sigma_0$) to enable cross-feature comparison on commensurate scales.

---

### 2.3 Population Stability Index (PSI)
The Population Stability Index is a discrete, bin-level measure of shift derived directly from symmetric Kullback-Leibler (KL) divergence. The baseline support is partitioned into $K=10$ quantile bins $B_k$. Let $E_k$ be the expected fraction of samples in bin $k$ from baseline, and $A_k$ be the actual fraction in the current sliding window:

$$\text{PSI} = \sum_{k=1}^K (A_k - E_k) \cdot \ln\left(\frac{A_k + \epsilon}{E_k + \epsilon}\right)$$

where $\epsilon = 10^{-4}$ is a Laplace regularizer to guarantee computational stability.

#### Operational Decision Boundaries:
- $\text{PSI} < 0.10$: **Stable Invariant** (No model retraining required, green status).
- $0.10 \le \text{PSI} < 0.25$: **Moderate Drift** (Early warning, active queue inspection).
- $\text{PSI} \ge 0.25$: **Severe Drift / Distribution Shock** (Immediate human-in-the-loop audit or automated fallbacks).

---

## 3. Streaming Engine Architecture

```
                       [ Synthetic Data Stream ]
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     Batches [00 - 10]    Batches [11 - 20]    Batches [21 - 30]
     Stationary Normal    Covariate Shift      Severe Shock / Crisis
     Mean = 120, σ = 28   Mean → 215, σ → 68   Bimodal Mixture Spike
              │                    │                    │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
             [ Vectorized Statistical Audit Engine ]
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
   Non-Parametric Tests                           Density Estimation
   • Scipy KS 2-Sample                            • Scikit-Learn KDE
   • Bonferroni Multi-Correction                  • Divergence Shading
   • Wasserstein-1 (EMD)                          • Dual Plotly Visuals
   • 10-Quantile Binned PSI                       • Bento Metric Cards
```

---

## 4. Setup & Local Execution

### Prerequisites
- Python 3.10 or higher
- Clean virtual environment (`venv` or `conda`)

```bash
# 1. Clone repository
git clone https://github.com/admissions-portfolio/driftlab.git
cd driftlab

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install academic dependencies
pip install -r requirements.txt

# 4. Launch Streamlit Neo-Bento Dashboard
streamlit run app.py
```

The application will immediately bind to `http://localhost:8501` in your browser.

---

## 5. Directory Structure

```
├── app.py               # Complete single-file Neo-Bento Streamlit application
├── requirements.txt     # Scientific dependencies (scipy, scikit-learn, plotly, etc.)
├── README.md            # Academic portfolio documentation & mathematical proofs
├── src/                 # Interactive live web dashboard (React / Vite / Tailwind)
└── metadata.json        # AI Studio platform configuration
```

---

## 6. Author Information
- **Author**: Siddhant Bhandare
- **Email**: [siddhantbhandare1@gmail.com](mailto:siddhantbhandare1@gmail.com)
- **Portfolio Focus**: M.Sc. Data Science & Machine Learning (Targeting TUM, LMU Munich, TU Berlin, RWTH Aachen)
- **Core Competencies**: Statistical Signal Processing, Robust MLOps, Non-Parametric Inference, Optimal Transport
- **System Operational Mode**: 100% Offline · Zero API keys, zero cloud dependencies, air-gapped deterministic computation.
