"""
DriftLab: Real-Time Statistical Distribution Shift & Data Drift Auditor
Academic-grade streaming distribution auditor for Machine Learning Systems.
Author: Siddhant Bhandare (siddhantbhandare1@gmail.com)
System: 100% Offline · Zero external API keys or cloud dependencies
Tech Stack: Python 3.10+, Scikit-Learn, SciPy, Pandas, NumPy, Plotly, Streamlit.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
from typing import Dict, Tuple, List

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DriftLab | Distribution Shift Auditor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# NEO-BENTO CUSTOM CSS INJECTION
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global resets & pitch black theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0B0C !important;
    color: #F4F4F5 !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

footer {
    display: none !important;
}

/* Hide Streamlit default padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 1400px !important;
}

/* Base Bento Card */
.bento-card {
    background-color: #17171A;
    border: 1px solid #26262B;
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.bento-card:hover {
    border-color: #3F3F46;
}

/* Bento Hero Accents */
.bento-hero-yellow {
    background-color: #E6F852 !important;
    color: #0B0B0C !important;
    border-radius: 24px;
    padding: 26px;
    border: none;
    box-shadow: 0 0 35px -6px rgba(230, 248, 82, 0.35);
}

.bento-hero-green {
    background-color: #1FE075 !important;
    color: #0B0B0C !important;
    border-radius: 24px;
    padding: 26px;
    border: none;
    box-shadow: 0 0 35px -6px rgba(31, 224, 117, 0.35);
}

.bento-hero-pink {
    background-color: #FF2E74 !important;
    color: #0B0B0C !important;
    border-radius: 24px;
    padding: 26px;
    border: none;
    box-shadow: 0 0 35px -6px rgba(255, 46, 116, 0.35);
}

.bento-hero-blue {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
    border-radius: 24px;
    padding: 26px;
    border: none;
    box-shadow: 0 0 35px -6px rgba(59, 130, 246, 0.35);
}

/* Metric Badges (Micro-Tags) */
.badge-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    text-transform: uppercase;
}

.badge-green {
    background-color: rgba(31, 224, 117, 0.15);
    color: #1FE075;
    border: 1px solid rgba(31, 224, 117, 0.35);
}

.badge-yellow {
    background-color: rgba(230, 248, 82, 0.15);
    color: #E6F852;
    border: 1px solid rgba(230, 248, 82, 0.35);
}

.badge-pink {
    background-color: rgba(255, 46, 116, 0.15);
    color: #FF2E74;
    border: 1px solid rgba(255, 46, 116, 0.35);
}

/* Custom Table Styles */
.audit-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 12px;
    font-size: 14px;
}

.audit-table th {
    background-color: #121215;
    color: #A1A1AA;
    text-align: left;
    padding: 14px 18px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #26262B;
}

.audit-table th:first-child {
    border-top-left-radius: 12px;
}
.audit-table th:last-child {
    border-top-right-radius: 12px;
}

.audit-table td {
    padding: 16px 18px;
    border-bottom: 1px solid #1E1E24;
    color: #E4E4E7;
    font-family: 'JetBrains Mono', monospace;
}

.audit-table tr:hover td {
    background-color: rgba(255, 255, 255, 0.02);
}

/* Floating Bottom Navigation Pill */
.floating-pill-bar {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(24, 24, 27, 0.85);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid #27272A;
    border-radius: 9999px;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    z-index: 999;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
}

/* Streamlit Button override */
div.stButton > button {
    background-color: #FF2E74 !important;
    color: #0B0B0C !important;
    font-weight: 700 !important;
    border-radius: 9999px !important;
    border: none !important;
    padding: 8px 24px !important;
    letter-spacing: 0.04em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

div.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(255, 46, 116, 0.5) !important;
}

/* Slider Track override */
div[data-testid="stSlider"] > div {
    color: #E6F852 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 1: SYNTHETIC NON-STATIONARY STREAMING ENGINE
# -----------------------------------------------------------------------------
FEATURE_NAMES = [
    "Transaction Amount",
    "Velocity",
    "Account Balance",
    "Risk Score"
]

@st.cache_data
def generate_synthetic_stream(
    n_batches: int = 31,
    samples_per_batch: int = 400,
    random_seed: int = 42
) -> Dict[int, pd.DataFrame]:
    """
    Generates non-stationary multi-dimensional continuous stream simulation.
    - Batches 0-10: Stationary Baseline Gaussian distribution
    - Batches 11-20: Progressive Covariate Shift (mean drift & variance expansion)
    - Batches 21-30: Severe Sudden Distribution Shock (simulating market crash / fraud wave)
    """
    rng = np.random.default_rng(random_seed)
    batches = {}

    for b in range(n_batches):
        if b <= 10:
            # Baseline Regime (Stationary)
            amt = rng.normal(loc=120.0, scale=28.0, size=samples_per_batch)
            vel = rng.normal(loc=4.5, scale=1.2, size=samples_per_batch)
            bal = rng.normal(loc=4800.0, scale=1100.0, size=samples_per_batch)
            rsk = rng.beta(a=2.0, b=8.0, size=samples_per_batch)  # Mean ~ 0.20
        elif 11 <= b <= 20:
            # Progressive Covariate Shift
            progress = (b - 10) / 10.0  # 0.1 -> 1.0
            amt = rng.normal(
                loc=120.0 + progress * 95.0,  # Mean drifts to 215
                scale=28.0 + progress * 40.0, # Variance expands
                size=samples_per_batch
            )
            vel = rng.normal(loc=4.5 + progress * 2.8, scale=1.2 + progress * 0.8, size=samples_per_batch)
            bal = rng.normal(loc=4800.0 - progress * 900.0, scale=1100.0 + progress * 300.0, size=samples_per_batch)
            rsk = rng.beta(a=2.0 + progress * 2.5, b=8.0 - progress * 2.0, size=samples_per_batch)
        else:
            # Late batches 21-30: Severe Sudden Distribution Shock (Bimodal Fraud / Crisis Spike)
            shock_weight = (b - 20) / 10.0
            # Bimodal mixture for Transaction Amount
            comp1 = rng.normal(loc=130.0, scale=30.0, size=int(samples_per_batch * 0.4))
            comp2 = rng.normal(loc=460.0 + shock_weight * 80.0, scale=95.0, size=samples_per_batch - len(comp1))
            amt = np.concatenate([comp1, comp2])
            rng.shuffle(amt)

            vel = rng.normal(loc=12.5 + shock_weight * 3.5, scale=3.5, size=samples_per_batch)
            bal = rng.normal(loc=2600.0 - shock_weight * 800.0, scale=1800.0, size=samples_per_batch)
            rsk = np.clip(rng.beta(a=6.0 + shock_weight * 4.0, b=2.2, size=samples_per_batch), 0.0, 1.0)

        # Enforce realistic boundary constraints
        amt = np.clip(amt, 5.0, 2500.0)
        vel = np.clip(vel, 0.1, 50.0)
        bal = np.clip(bal, 50.0, 20000.0)
        rsk = np.clip(rsk, 0.001, 0.999)

        batches[b] = pd.DataFrame({
            "Transaction Amount": amt,
            "Velocity": vel,
            "Account Balance": bal,
            "Risk Score": rsk
        })

    return batches


# -----------------------------------------------------------------------------
# MODULE 2: NON-PARAMETRIC STATISTICAL TESTING & DRIFT SCORING
# -----------------------------------------------------------------------------
def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Population Stability Index (PSI) using quantile binning from reference.
    PSI = sum((A_i - E_i) * ln(A_i / E_i)) with epsilon Laplace smoothing.
    """
    # Create bin boundaries using quantiles from expected (reference) data
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    # Ensure strictly monotonic bin edges to avoid zero-width bins
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(np.min(expected), np.max(expected), num_bins + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractional probabilities with epsilon smoothing
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize after adding epsilon
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    # PSI calculation
    psi_bins = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_bins))

    return total_psi, bin_edges, expected_pct, actual_pct


def run_drift_audit(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Runs multi-feature non-parametric drift audit:
    - 2-Sample Kolmogorov-Smirnov Test with Bonferroni correction
    - Wasserstein-1 Distance (Earth Mover's Distance)
    - Population Stability Index (PSI) with strict categorizations
    """
    m = len(FEATURE_NAMES)
    bonferroni_alpha = alpha / m

    audit_records = []

    for feat in FEATURE_NAMES:
        ref_data = baseline_df[feat].to_numpy()
        cur_data = current_df[feat].to_numpy()

        # 1. Kolmogorov-Smirnov Test
        ks_res = stats.ks_2samp(ref_data, cur_data)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. Wasserstein-1 Distance
        w1_dist = float(stats.wasserstein_distance(ref_data, cur_data))

        # 3. Population Stability Index (PSI)
        psi_val, _, _, _ = calculate_psi(ref_data, cur_data, num_bins=10)

        # Categorize drift state
        if psi_val < 0.10 and p_val > bonferroni_alpha:
            status = "STABLE"
            pill_class = "badge-green"
        elif psi_val < 0.25:
            status = "MODERATE SHIFT"
            pill_class = "badge-yellow"
        else:
            status = "SEVERE DRIFT"
            pill_class = "badge-pink"

        audit_records.append({
            "Feature": feat,
            "KS Statistic": ks_stat,
            "p-value": p_val,
            "Bonferroni Significant": p_val < bonferroni_alpha,
            "Wasserstein Distance": w1_dist,
            "PSI Score": psi_val,
            "Status": status,
            "Pill Class": pill_class
        })

    return pd.DataFrame(audit_records)


# -----------------------------------------------------------------------------
# PLOTLY CHART BUILDERS (DARK NEO-BENTO AESTHETICS)
# -----------------------------------------------------------------------------
def build_wasserstein_timeline_chart(
    stream_data: Dict[int, pd.DataFrame],
    baseline_df: pd.DataFrame,
    current_batch: int
) -> go.Figure:
    """
    Bento Card 1: Dynamic Plotly line chart tracing Wasserstein Distance over time
    with electric green neon glow trace and transparent background.
    """
    batches_x = list(range(current_batch + 1))
    fig = go.Figure()

    colors = {
        "Transaction Amount": "#1FE075",  # Neon Green
        "Velocity": "#E6F852",            # Acid Yellow
        "Account Balance": "#3B82F6",     # Cobalt Blue
        "Risk Score": "#FF2E74"           # Cyber Pink
    }

    for feat in FEATURE_NAMES:
        y_vals = []
        ref_arr = baseline_df[feat].to_numpy()
        # Scale for normalized display comparability across different physical units
        std_ref = np.std(ref_arr) if np.std(ref_arr) > 0 else 1.0

        for b in batches_x:
            b_arr = stream_data[b][feat].to_numpy()
            raw_w = stats.wasserstein_distance(ref_arr, b_arr)
            norm_w = raw_w / std_ref
            y_vals.append(norm_w)

        # Primary Line
        fig.add_trace(go.Scatter(
            x=batches_x,
            y=y_vals,
            mode='lines+markers',
            name=feat,
            line=dict(color=colors[feat], width=3),
            marker=dict(size=6, color=colors[feat]),
            hovertemplate=f"<b>{feat}</b><br>Batch: %{{x}}<br>Norm W1: %{{y:.3f}}<extra></extra>"
        ))

    # Add Drift Alarm Threshold line (e.g. 0.50 normalized shift)
    fig.add_hline(
        y=0.45,
        line_dash="dot",
        line_color="#FF2E74",
        annotation_text="Critical Shift Threshold (0.45 σ)",
        annotation_position="bottom right",
        annotation_font=dict(color="#FF2E74", size=10)
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#A1A1AA")
        ),
        xaxis=dict(
            title="Simulation Batch Tick (0 → 30)",
            title_font=dict(color="#71717A", size=11),
            tickfont=dict(color="#A1A1AA", size=10),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)',
            range=[-0.5, 30.5]
        ),
        yaxis=dict(
            title="Normalized Earth Mover's Distance (W₁ / σ)",
            title_font=dict(color="#71717A", size=11),
            tickfont=dict(color="#A1A1AA", size=10),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        hovermode="x unified"
    )

    return fig


def build_kde_divergence_chart(
    ref_series: np.ndarray,
    cur_series: np.ndarray,
    feature_name: str
) -> go.Figure:
    """
    Bento Card 2: Interactive Kernel Density Estimation comparison curves
    (Baseline vs. Current Window) highlighting divergence area in yellow/pink.
    """
    # Evaluate KDE over common support
    min_val = min(np.percentile(ref_series, 0.5), np.percentile(cur_series, 0.5))
    max_val = max(np.percentile(ref_series, 99.5), np.percentile(cur_series, 99.5))
    x_grid = np.linspace(min_val, max_val, 300)

    # Fit Gaussian KDE
    kde_ref = stats.gaussian_kde(ref_series)
    kde_cur = stats.gaussian_kde(cur_series)

    pdf_ref = kde_ref(x_grid)
    pdf_cur = kde_cur(x_grid)

    fig = go.Figure()

    # Baseline PDF trace (Subtle Cobalt / Slate)
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=pdf_ref,
        mode='lines',
        name='Baseline P(X) [Batches 0-10]',
        line=dict(color='#3B82F6', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.12)',
        hovertemplate="Baseline Density: %{y:.4f}<extra></extra>"
    ))

    # Current Window PDF trace (Vibrant Acid Yellow or Hot Pink if drifted)
    cur_color = '#FF2E74' if np.abs(np.mean(cur_series) - np.mean(ref_series)) > 0.5 * np.std(ref_series) else '#E6F852'
    cur_fill = 'rgba(255, 46, 116, 0.20)' if cur_color == '#FF2E74' else 'rgba(230, 248, 82, 0.20)'

    fig.add_trace(go.Scatter(
        x=x_grid,
        y=pdf_cur,
        mode='lines',
        name='Current Window Q(X) [Sliding]',
        line=dict(color=cur_color, width=2.8),
        fill='tozeroy',
        fillcolor=cur_fill,
        hovertemplate="Current Density: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#A1A1AA")
        ),
        xaxis=dict(
            title=f"{feature_name} Support Domain",
            title_font=dict(color="#71717A", size=11),
            tickfont=dict(color="#A1A1AA", size=10),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        yaxis=dict(
            title="Probability Density f(x)",
            title_font=dict(color="#71717A", size=11),
            tickfont=dict(color="#A1A1AA", size=10),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        hovermode="x unified"
    )

    return fig


# -----------------------------------------------------------------------------
# APPLICATION MAIN CONTROLLER
# -----------------------------------------------------------------------------
def main():
    # Session state for shock injection
    if "shock_injected" not in st.session_state:
        st.session_state.shock_injected = False

    # 1. Generate & Cache Simulation Stream
    stream_data = generate_synthetic_stream(n_batches=31, samples_per_batch=400)
    baseline_df = stream_data[0]

    # Header Title
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 24px;">
        <div>
            <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.12em; color: #E6F852; text-transform: uppercase; margin-bottom: 4px;">
                Admissions Portfolio · M.Sc. Data Science & Machine Learning
            </div>
            <h1 style="margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.03em; color: #FFFFFF;">
                DriftLab: Real-Time Statistical Distribution Shift Auditor
            </h1>
            <p style="margin: 4px 0 0 0; color: #A1A1AA; font-size: 14px;">
                Vectorized Non-Parametric Hypothesis Testing (KS-Test), Earth Mover's Distance (Wasserstein-1), and Population Stability Index (PSI).
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge-pill badge-green" style="font-size: 12px; padding: 6px 14px;">
                ● Streaming Engine Active
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TIME-STEP SCRUBBER & DRIFT SHOCK CONTROLS
    # -------------------------------------------------------------------------
    st.markdown('<div class="bento-card" style="padding: 18px 24px; margin-bottom: 22px;">', unsafe_allow_html=True)
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([6, 3, 3])

    with c_ctrl1:
        selected_batch = st.slider(
            "CHRONOLOGICAL TIMELINE BATCH SCRUBBER (0 TO 30)",
            min_value=0,
            max_value=30,
            value=25 if st.session_state.shock_injected else 14,
            help="Step through the data stream batches. 0-10: Baseline, 11-20: Covariate Shift, 21-30: Severe Shock."
        )

    with c_ctrl2:
        st.write("")
        st.write("")
        if st.button("⚡ Inject Distribution Shock", use_container_width=True):
            st.session_state.shock_injected = True
            st.rerun()

    with c_ctrl3:
        st.write("")
        st.write("")
        if st.button("↺ Reset to Baseline (Batch 0)", use_container_width=True):
            st.session_state.shock_injected = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Current Window Evaluation
    current_df = stream_data[selected_batch]
    audit_results = run_drift_audit(baseline_df, current_df, alpha=0.05)

    # Compute overall system health metrics
    severe_count = sum(audit_results["Status"] == "SEVERE DRIFT")
    moderate_count = sum(audit_results["Status"] == "MODERATE SHIFT")
    stable_count = sum(audit_results["Status"] == "STABLE")

    if severe_count >= 2:
        stability_pct = 42
        sys_status_text = "CRITICAL SHOCK DETECTED"
        hero_theme = "bento-hero-pink"
    elif severe_count == 1 or moderate_count >= 2:
        stability_pct = 74
        sys_status_text = "MODERATE DRIFT OBSERVED"
        hero_theme = "bento-hero-yellow"
    else:
        stability_pct = 98
        sys_status_text = "SYSTEM STATUS: 98% STABLE"
        hero_theme = "bento-hero-yellow" if stability_pct < 95 else "bento-hero-green"

    # -------------------------------------------------------------------------
    # TOP HEADER & BENTO METRICS GRID (3 ACCENT CARDS)
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="bento-hero-yellow" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;">
                REGIME STABILITY INDEX
            </div>
            <div style="font-size: 32px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;">
                {stability_pct}% STABLE
            </div>
            <div style="font-size: 12px; font-weight: 700; opacity: 0.85;">
                Current Batch: #{selected_batch:02d} · Window n={len(current_df)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="bento-hero-green" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;">
                TELEMETRY & INVARIANTS
            </div>
            <div style="font-size: 32px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;">
                4 FEATURES MONITORED
            </div>
            <div style="font-size: 12px; font-weight: 700; opacity: 0.85;">
                {stable_count} Stable · {moderate_count} Moderate Shift
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="bento-hero-pink" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;">
                CRITICAL THRESHOLD BREACHES
            </div>
            <div style="font-size: 32px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;">
                {severe_count} FEATURES CRITICAL
            </div>
            <div style="font-size: 12px; font-weight: 700; opacity: 0.85;">
                Bonferroni α/4 = 0.0125 Threshold
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # -------------------------------------------------------------------------
    # DUAL VISUAL ANALYSIS CARDS
    # -------------------------------------------------------------------------
    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        st.markdown("""
        <div class="bento-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 14px; font-weight: 700; color: #FFFFFF;">
                    Wasserstein-1 Distance (Earth Mover's Trajectory)
                </div>
                <span class="badge-pill badge-green">Dynamic W₁ / σ</span>
            </div>
            <p style="color: #71717A; font-size: 12px; margin-top: -6px; margin-bottom: 12px;">
                Quantifies minimal transport energy required to transform empirical sample into baseline support.
            </p>
        """, unsafe_allow_html=True)

        w1_fig = build_wasserstein_timeline_chart(stream_data, baseline_df, selected_batch)
        st.plotly_chart(w1_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with g_col2:
        # Feature selector for KDE comparison
        st.markdown("""
        <div class="bento-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-size: 14px; font-weight: 700; color: #FFFFFF;">
                    Kernel Density Estimation (Baseline vs. Current Window)
                </div>
                <span class="badge-pill badge-yellow">Non-Parametric KDE</span>
            </div>
        """, unsafe_allow_html=True)

        selected_feature = st.selectbox(
            "Select Feature for Distribution Overlay:",
            options=FEATURE_NAMES,
            index=0,
            label_visibility="collapsed"
        )

        kde_fig = build_kde_divergence_chart(
            baseline_df[selected_feature].to_numpy(),
            current_df[selected_feature].to_numpy(),
            selected_feature
        )
        st.plotly_chart(kde_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DETAILED FEATURE AUDIT TABLE
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="bento-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div>
                <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #FFFFFF;">
                    Rigorous Feature-Level Statistical Audit
                </h3>
                <p style="margin: 2px 0 0 0; color: #71717A; font-size: 12px;">
                    Null Hypothesis H₀: P(X) = Q(X) across current sliding batch vs. reference baseline.
                </p>
            </div>
            <span class="badge-pill badge-yellow">
                Bonferroni Corrected α = 0.0125
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Construct HTML table for ultra-clean bento formatting
    rows_html = ""
    for _, row in audit_results.iterrows():
        p_val_display = f"{row['p-value']:.2e}" if row['p-value'] < 0.001 else f"{row['p-value']:.4f}"
        rows_html += f"""
        <tr>
            <td style="font-weight: 600; color: #FFFFFF; font-family: 'Plus Jakarta Sans', sans-serif;">{row['Feature']}</td>
            <td>{row['KS Statistic']:.4f}</td>
            <td style="color: {'#FF2E74' if row['Bonferroni Significant'] else '#A1A1AA'}; font-weight: {'700' if row['Bonferroni Significant'] else '400'};">
                {p_val_display} {'*' if row['Bonferroni Significant'] else ''}
            </td>
            <td>{row['Wasserstein Distance']:.3f}</td>
            <td style="font-weight: 700;">{row['PSI Score']:.4f}</td>
            <td>
                <span class="badge-pill {row['Pill Class']}">{row['Status']}</span>
            </td>
        </tr>
        """

    table_html = f"""
    <table class="audit-table">
        <thead>
            <tr>
                <th>Feature Name</th>
                <th>KS Statistic (D)</th>
                <th>Asymptotic p-value</th>
                <th>Wasserstein-1 (EMD)</th>
                <th>PSI Score</th>
                <th>Audit Decision</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    <div style="margin-top: 14px; font-size: 11px; color: #71717A; display: flex; justify-content: space-between;">
        <span>* Indicates p < α_bonferroni (0.0125). Rejects H₀: Distribution shift confirmed.</span>
        <span>Decision Bounds: PSI < 0.10 (Stable) | 0.10 ≤ PSI < 0.25 (Moderate) | PSI ≥ 0.25 (Action Required)</span>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # AUTHOR & PROJECT FOOTER
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; margin-top: 48px; margin-bottom: 30px; font-size: 13px; color: #71717A; font-family: 'Plus Jakarta Sans', sans-serif;">
        100% Offline Statistical Engine · Zero Cloud API Dependencies<br/>
        Made by <strong style="color: #FFFFFF;">Siddhant Bhandare</strong> · <a href="mailto:siddhantbhandare1@gmail.com" style="color: #E6F852; text-decoration: none; font-weight: 600;">siddhantbhandare1@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # FLOATING BOTTOM NAVIGATION PILL
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div class="floating-pill-bar">
        <span style="font-size: 12px; font-weight: 700; color: #E6F852;">
            ⚡ DriftLab v2.4-Academic
        </span>
        <span style="color: #3F3F46;">|</span>
        <span style="font-size: 12px; color: #A1A1AA;">
            Batch <b style="color: #FFFFFF;">#{selected_batch:02d} / 30</b>
        </span>
        <span style="color: #3F3F46;">|</span>
        <span style="font-size: 12px; color: {'#1FE075' if severe_count == 0 else '#FF2E74'}; font-weight: 700;">
            {severe_count} Critical Alerts
        </span>
        <span style="color: #3F3F46;">|</span>
        <span style="font-size: 11px; color: #A1A1AA;">
            By Siddhant Bhandare
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
