import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import io

# ============================================================
# PAGE CONFIGURATION & THEME
# ============================================================

st.set_page_config(
    page_title="Stock Price Random Walk Simulation | BDA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-pill {
        display: inline-block;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 12px;
        background-color: #2196F3;
        color: white;
    }
    .highlight-box {
        background-color: rgba(33, 150, 243, 0.08);
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">📈 One-Dimensional Stock Price Random Walk Simulation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header"><b>Mathematical Foundation of Big Data Analytics (BDA)</b> &bull; Monte Carlo Simulation, Law of Large Numbers (LLN) & Central Limit Theorem (CLT) Analysis</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - PRESETS & SIMULATION PARAMETERS
# ============================================================

st.sidebar.header("⚙️ Simulation Controls")

# Market Presets
preset_choice = st.sidebar.selectbox(
    "🎯 Market Scenario Preset",
    [
        "Custom Configuration",
        "⚖️ Fair Market (Martingale, p=0.50)",
        "🐂 Bullish Market (Positive Drift, p=0.60)",
        "🐻 Bearish Market (Negative Drift, p=0.40)",
        "⚡ High Volatility Speculation (d=5, p=0.52)",
        "🔬 Large-Scale Monte Carlo (M=3000, n=200)"
    ],
    index=0
)

# Preset values dictionary
presets = {
    "⚖️ Fair Market (Martingale, p=0.50)": {"s0": 100, "trials": 1000, "steps": 100, "d": 1, "p": 0.50, "model": "Discrete Random Walk (±d)"},
    "🐂 Bullish Market (Positive Drift, p=0.60)": {"s0": 100, "trials": 1500, "steps": 150, "d": 2, "p": 0.60, "model": "Discrete Random Walk (±d)"},
    "🐻 Bearish Market (Negative Drift, p=0.40)": {"s0": 100, "trials": 1500, "steps": 150, "d": 2, "p": 0.40, "model": "Discrete Random Walk (±d)"},
    "⚡ High Volatility Speculation (d=5, p=0.52)": {"s0": 100, "trials": 1000, "steps": 100, "d": 5, "p": 0.52, "model": "Discrete Random Walk (±d)"},
    "🔬 Large-Scale Monte Carlo (M=3000, n=200)": {"s0": 100, "trials": 3000, "steps": 200, "d": 1, "p": 0.50, "model": "Discrete Random Walk (±d)"}
}

# Determine default values based on preset
if preset_choice in presets:
    cfg = presets[preset_choice]
    def_s0 = cfg["s0"]
    def_trials = cfg["trials"]
    def_steps = cfg["steps"]
    def_d = cfg["d"]
    def_p = cfg["p"]
    def_model = cfg["model"]
else:
    def_s0 = 100
    def_trials = 1000
    def_steps = 100
    def_d = 1
    def_p = 0.50
    def_model = "Discrete Random Walk (±d)"

# Simulation Model Selector
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Discrete Random Walk (±d)", "Geometric Brownian Motion (GBM)"],
    index=0 if def_model == "Discrete Random Walk (±d)" else 1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

initial_price = st.sidebar.slider(
    "Initial Stock Price ($S_0$ in ₹)",
    min_value=10,
    max_value=1000,
    value=def_s0,
    step=10
)

trials = st.sidebar.slider(
    "Number of Trials ($M$ Paths)",
    min_value=10,
    max_value=5000,
    value=def_trials,
    step=50
)

steps = st.sidebar.slider(
    "Time Steps ($n$)",
    min_value=10,
    max_value=500,
    value=def_steps,
    step=10
)

if model_type == "Discrete Random Walk (±d)":
    price_change = st.sidebar.slider(
        "Step Increment ($d$ in ₹)",
        min_value=1,
        max_value=25,
        value=def_d,
        step=1
    )
    prob_up = st.sidebar.slider(
        "Upward Probability ($p$)",
        min_value=0.0,
        max_value=1.0,
        value=float(def_p),
        step=0.01
    )
    prob_down = 1.0 - prob_up
    st.sidebar.caption(f"Downward Probability ($1-p$): **{prob_down:.2f}**")
    
    drift_type = "Zero Drift (Martingale)" if abs(prob_up - 0.5) < 1e-4 else ("Bullish (+ Drift)" if prob_up > 0.5 else "Bearish (- Drift)")
    st.sidebar.info(f"Regime: **{drift_type}**")
else:
    annual_mu = st.sidebar.slider(
        "Annual Expected Return ($\mu$ %)",
        min_value=-50.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    ) / 100.0
    annual_vol = st.sidebar.slider(
        "Annual Volatility ($\sigma$ %)",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    ) / 100.0
    price_change = 1
    prob_up = 0.5

# Advanced options
with st.sidebar.expander("🛠️ Advanced Settings", expanded=False):
    enforce_zero_floor = st.checkbox("Enforce Price Floor ≥ ₹0 (Bankruptcy Barrier)", value=False)
    use_random_seed = st.checkbox("Set Fixed Random Seed (Reproducibility)", value=False)
    seed_value = st.number_input("Seed Value", min_value=0, max_value=999999, value=42, step=1) if use_random_seed else None

run_simulation = st.sidebar.button(
    "🚀 Run Simulation",
    type="primary",
    use_container_width=True
)


# ============================================================
# VECTORIZED SIMULATION ENGINES
# ============================================================

def simulate_discrete_random_walk(s0, m_trials, n_steps, step_size, p_up, zero_floor=False, seed=None):
    """
    Vectorized Discrete Random Walk simulation using NumPy.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate random matrix M x n
    rnd = np.random.random((m_trials, n_steps))
    # Vectorized step movements (+d if < p, else -d)
    steps_matrix = np.where(rnd < p_up, step_size, -step_size)
    
    # Pre-allocate array (M x n+1)
    paths = np.zeros((m_trials, n_steps + 1))
    paths[:, 0] = s0
    
    if not zero_floor:
        paths[:, 1:] = s0 + np.cumsum(steps_matrix, axis=1)
    else:
        # Step-by-step enforcement of zero floor
        current_p = np.full(m_trials, float(s0))
        for t in range(n_steps):
            current_p = np.maximum(0.0, current_p + steps_matrix[:, t])
            paths[:, t + 1] = current_p
            
    return paths

def simulate_gbm(s0, m_trials, n_steps, mu, sigma, dt=1/252, seed=None):
    """
    Geometric Brownian Motion (GBM): S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """
    if seed is not None:
        np.random.seed(seed)
        
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    z = np.random.normal(0, 1, size=(m_trials, n_steps))
    increments = drift + diffusion * z
    
    log_paths = np.zeros((m_trials, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    
    return np.exp(log_paths)


# ============================================================
# RUN & CACHE SIMULATION
# ============================================================

current_params = {
    "initial_price": initial_price,
    "trials": trials,
    "steps": steps,
    "price_change": price_change,
    "prob_up": prob_up,
    "model_type": model_type,
    "enforce_zero_floor": enforce_zero_floor,
    "seed": seed_value if use_random_seed else None
}

if run_simulation or "stock_prices" not in st.session_state or st.session_state.get("params") != current_params:
    if model_type == "Discrete Random Walk (±d)":
        paths = simulate_discrete_random_walk(
            initial_price, trials, steps, price_change, prob_up, 
            zero_floor=enforce_zero_floor, seed=seed_value if use_random_seed else None
        )
    else:
        paths = simulate_gbm(
            initial_price, trials, steps, annual_mu, annual_vol, 
            seed=seed_value if use_random_seed else None
        )
        
    st.session_state.stock_prices = paths
    st.session_state.params = current_params

paths = st.session_state.stock_prices
p_cfg = st.session_state.params
final_prices = paths[:, -1]
s0 = p_cfg["initial_price"]
n = p_cfg["steps"]
d = p_cfg["price_change"]
p = p_cfg["prob_up"]
m = p_cfg["trials"]


# ============================================================
# STATISTICAL CALCULATIONS (EMPIRICAL & THEORETICAL)
# ============================================================

# Empirical values
mean_final = np.mean(final_prices)
median_final = np.median(final_prices)
var_final = np.var(final_prices)
std_final = np.std(final_prices)
min_final = np.min(final_prices)
max_final = np.max(final_prices)
skewness = stats.skew(final_prices)
kurtosis = stats.kurtosis(final_prices)

mean_path = np.mean(paths, axis=0)
q10_path = np.percentile(paths, 10, axis=0)
q25_path = np.percentile(paths, 25, axis=0)
q75_path = np.percentile(paths, 75, axis=0)
q90_path = np.percentile(paths, 90, axis=0)

# Theoretical values (for Discrete Walk)
theo_step_expected = d * (2 * p - 1)
theo_final_expected = s0 + n * theo_step_expected
theo_step_var = 4 * (d**2) * p * (1 - p)
theo_final_var = n * theo_step_var
theo_final_std = np.sqrt(theo_final_var)

# Profitability & Risk
profitable_trials = np.sum(final_prices > s0)
losing_trials = np.sum(final_prices < s0)
breakeven_trials = np.sum(final_prices == s0)
prob_profit = (profitable_trials / m) * 100
prob_loss = (losing_trials / m) * 100
prob_breakeven = (breakeven_trials / m) * 100

var_95 = np.percentile(final_prices, 5)
var_99 = np.percentile(final_prices, 1)

# Maximum Drawdown (MDD) calculation across all paths
peak_paths = np.maximum.accumulate(paths, axis=1)
drawdowns = (peak_paths - paths) / np.maximum(peak_paths, 1e-6)
max_drawdowns_per_path = np.max(drawdowns, axis=1) * 100
mean_mdd = np.mean(max_drawdowns_per_path)


# ============================================================
# TOP KPI METRICS BAR
# ============================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    label="Initial Price (S₀)",
    value=f"₹{s0:.2f}"
)

kpi2.metric(
    label="Empirical Mean Price",
    value=f"₹{mean_final:.2f}",
    delta=f"{(mean_final - s0):+.2f} ({(mean_final - s0)/s0 * 100:+.1f}%)"
)

kpi3.metric(
    label="Theoretical Expected E[Sₙ]",
    value=f"₹{theo_final_expected:.2f}" if model_type.startswith("Discrete") else "N/A (GBM)",
    delta=f"Err: {abs(mean_final - theo_final_expected):.2f}" if model_type.startswith("Discrete") else None,
    delta_color="inverse"
)

kpi4.metric(
    label="Empirical Std Dev (σ)",
    value=f"₹{std_final:.2f}"
)

kpi5.metric(
    label="Win Rate (P > S₀)",
    value=f"{prob_profit:.1f}%",
    delta=f"{prob_profit - 50.0:+.1f}% vs 50/50"
)

st.markdown("---")


# ============================================================
# MAIN APPLICATION TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trajectories & Fan Chart",
    "📊 Distribution & CLT Normal Fit",
    "⚖️ Theoretical vs Empirical (LLN)",
    "🛡️ Risk & Profitability Metrics",
    "🧮 Mathematical Proofs & Theory",
    "💾 Export Simulation Data"
])


# ------------------------------------------------------------
# TAB 1: TRAJECTORIES & FAN CHART
# ------------------------------------------------------------
with tab1:
    st.subheader("📈 Stochastic Multi-Path Trajectories & Quantile Dispersion")
    
    col_plot, col_info = st.columns([3, 1])
    
    with col_plot:
        fig1, ax1 = plt.subplots(figsize=(11, 5.5), facecolor='none')
        ax1.set_facecolor('#0E1117' if st.get_option("theme.base") == "dark" else '#F8F9FA')
        
        time_axis = np.arange(n + 1)
        
        # Plot quantile fan bands (dispersion cone)
        ax1.fill_between(time_axis, q10_path, q90_path, color='#1E88E5', alpha=0.15, label="10th - 90th Percentile Band")
        ax1.fill_between(time_axis, q25_path, q75_path, color='#1E88E5', alpha=0.25, label="25th - 75th Percentile (IQR)")
        
        # Plot subset of individual paths (up to 300 to avoid clutter)
        sample_paths_count = min(300, m)
        for i in range(sample_paths_count):
            ax1.plot(time_axis, paths[i], color='#90CAF9', alpha=0.04, linewidth=0.8)
            
        # Plot Best and Worst paths
        best_idx = np.argmax(final_prices)
        worst_idx = np.argmin(final_prices)
        ax1.plot(time_axis, paths[best_idx], color='#00E676', linewidth=1.5, linestyle=':', label=f"Max Path (₹{max_final:.1f})")
        ax1.plot(time_axis, paths[worst_idx], color='#FF5252', linewidth=1.5, linestyle=':', label=f"Min Path (₹{min_final:.1f})")
        
        # Plot Mean trajectory and Initial Price baseline
        ax1.plot(time_axis, mean_path, color='#FFD600', linewidth=2.8, label=f"Empirical Mean Path (Final: ₹{mean_final:.2f})")
        ax1.axhline(s0, color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#333333', linestyle='--', linewidth=1.6, label=f"Initial S₀ (₹{s0})")
        
        ax1.set_title(f"Monte Carlo Simulation: {m:,} Stochastic Trajectories over {n} Steps", fontsize=13, fontweight='bold', pad=12)
        ax1.set_xlabel("Time Step (t)", fontsize=11)
        ax1.set_ylabel("Stock Price (₹)", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc="upper left", framealpha=0.8, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig1)
        
    with col_info:
        st.markdown("### 💡 Trajectory Insights")
        st.markdown(f"""
        - **Total Paths:** `{m:,}`
        - **Total Time Steps:** `{n}`
        - **Diffusion Cone:** The expanding shaded bands illustrate **variance dispersion** $\sigma \propto \sqrt{t}$.
        - **Trajectory Range:** 
          - 🔼 **Peak:** ₹{max_final:.2f}
          - 🔽 **Trough:** ₹{min_final:.2f}
          - 📊 **Spread:** ₹{max_final - min_final:.2f}
        """)
        
        if p > 0.5:
            st.success(f"**Bullish Drift Detected:** With $p={p:.2f} > 0.50$, the mean path slopes upwards at $+₹{theo_step_expected:.2f}$/step.")
        elif p < 0.5:
            st.error(f"**Bearish Drift Detected:** With $p={p:.2f} < 0.50$, the mean path slopes downwards at $-₹{abs(theo_step_expected):.2f}$/step.")
        else:
            st.info("**Martingale Property:** With $p=0.50$, the process has zero expected drift ($\mathbb{E}[S_t] = S_0$).")


# ------------------------------------------------------------
# TAB 2: DISTRIBUTION & CLT NORMAL FIT
# ------------------------------------------------------------
with tab2:
    st.subheader("📊 Terminal Price Distribution & Central Limit Theorem (CLT) Validation")
    
    col_hist, col_stats = st.columns([3, 1])
    
    with col_hist:
        fig2, ax2 = plt.subplots(figsize=(11, 5.5), facecolor='none')
        ax2.set_facecolor('#0E1117' if st.get_option("theme.base") == "dark" else '#F8F9FA')
        
        num_bins = min(50, max(20, int(np.sqrt(m))))
        
        # Plot Histogram
        count, bins, patches = ax2.hist(
            final_prices,
            bins=num_bins,
            density=True,
            alpha=0.65,
            edgecolor='black',
            color='#1E88E5',
            label="Empirical Distribution (Normalized)"
        )
        
        # Color code profit vs loss bars
        for patch, bin_left in zip(patches, bins[:-1]):
            if bin_left >= s0:
                patch.set_facecolor('#2E7D32')
                patch.set_alpha(0.6)
            else:
                patch.set_facecolor('#C62828')
                patch.set_alpha(0.6)
                
        # Overlay Theoretical Normal Distribution (CLT Gaussian Curve)
        x_norm = np.linspace(min_final - 5, max_final + 5, 500)
        
        if model_type.startswith("Discrete") and theo_final_std > 0:
            y_norm = stats.norm.pdf(x_norm, loc=theo_final_expected, scale=theo_final_std)
            ax2.plot(x_norm, y_norm, color='#FFD600', linewidth=2.8, linestyle='-', label=f"Theoretical CLT Normal Fit $\mathcal{{N}}(\mu={theo_final_expected:.1f}, \sigma={theo_final_std:.1f})$")
        else:
            y_norm = stats.norm.pdf(x_norm, loc=mean_final, scale=std_final)
            ax2.plot(x_norm, y_norm, color='#FFD600', linewidth=2.8, linestyle='-', label=f"Empirical Gaussian Fit $\mathcal{{N}}(\mu={mean_final:.1f}, \sigma={std_final:.1f})$")
            
        # Vertical markers
        ax2.axvline(s0, color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000', linestyle='--', linewidth=1.8, label=f"Initial Price S₀ (₹{s0})")
        ax2.axvline(mean_final, color='#00E676', linestyle='-', linewidth=2, label=f"Sample Mean (₹{mean_final:.2f})")
        ax2.axvline(median_final, color='#FF9100', linestyle=':', linewidth=2, label=f"Median (₹{median_final:.2f})")
        
        ax2.set_title(f"Terminal Price Distribution vs. Theoretical Normal PDF (M={m:,} Trials)", fontsize=13, fontweight='bold', pad=12)
        ax2.set_xlabel("Terminal Stock Price Sₙ (₹)", fontsize=11)
        ax2.set_ylabel("Probability Density", fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc="upper right", framealpha=0.85, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        
    with col_stats:
        st.markdown("### 🔬 Normality Diagnostics")
        st.markdown(f"""
        - **Green Zone:** Profit ($S_n \ge S_0$)
        - **Red Zone:** Loss ($S_n < S_0$)
        - **Skewness:** `{skewness:.3f}` *(0 = perfectly symmetric)*
        - **Excess Kurtosis:** `{kurtosis:.3f}` *(0 = mesokurtic / normal)*
        """)
        
        if abs(skewness) < 0.2 and abs(kurtosis) < 0.5:
            st.success("✅ **CLT Confirmation:** Distribution exhibits strong asymptotic convergence to a Gaussian Bell Curve.")
        else:
            st.warning("⚠️ **Skew/Fat Tails:** Moderate deviation from normality due to finite step size or asymmetric probability bias.")


# ------------------------------------------------------------
# TAB 3: THEORETICAL VS EMPIRICAL (LLN)
# ------------------------------------------------------------
with tab3:
    st.subheader("⚖️ Asymptotic Verification: Empirical Results vs. Mathematical Theorems")
    
    st.markdown("""
    Under the **Law of Large Numbers (LLN)**, as the number of Monte Carlo trials $M \to \infty$, 
    the sample averages converge almost surely to their analytical expected values.
    """)
    
    comp_data = {
        "Metric / Parameter": [
            "Expected Final Price 𝔼[Sₙ]",
            "Step Expected Value 𝔼[Xₜ]",
            "Standard Deviation σ(Sₙ)",
            "Variance Var(Sₙ)",
            "Median Price",
            "Minimum Realization",
            "Maximum Realization"
        ],
        "Theoretical Formula": [
            "S₀ + n · d(2p - 1)",
            "d(2p - 1)",
            "2d · √(n · p · (1-p))",
            "4n · d² · p · (1-p)",
            "Equal to Mean (if symmetric)",
            "S₀ - n · d (Absolute Min)",
            "S₀ + n · d (Absolute Max)"
        ],
        "Theoretical Value": [
            f"₹{theo_final_expected:.2f}",
            f"₹{theo_step_expected:.2f}",
            f"{theo_final_std:.2f}",
            f"{theo_final_var:.2f}",
            f"₹{theo_final_expected:.2f}",
            f"₹{s0 - n * d:.2f}",
            f"₹{s0 + n * d:.2f}"
        ],
        "Empirical Monte Carlo": [
            f"₹{mean_final:.2f}",
            f"₹{(mean_final - s0) / n:.2f}",
            f"{std_final:.2f}",
            f"{var_final:.2f}",
            f"₹{median_final:.2f}",
            f"₹{min_final:.2f}",
            f"₹{max_final:.2f}"
        ],
        "Absolute Error": [
            f"₹{abs(mean_final - theo_final_expected):.2f}",
            f"₹{abs((mean_final - s0)/n - theo_step_expected):.4f}",
            f"{abs(std_final - theo_final_std):.2f}",
            f"{abs(var_final - theo_final_var):.2f}",
            f"₹{abs(median_final - theo_final_expected):.2f}",
            "-",
            "-"
        ],
        "Relative Error (%)": [
            f"{abs(mean_final - theo_final_expected) / max(abs(theo_final_expected), 1e-4) * 100:.2f}%",
            f"{abs((mean_final - s0)/n - theo_step_expected) / max(abs(theo_step_expected), 1e-4) * 100:.2f}%" if abs(theo_step_expected) > 0 else "0.00%",
            f"{abs(std_final - theo_final_std) / max(theo_final_std, 1e-4) * 100:.2f}%",
            f"{abs(var_final - theo_final_var) / max(theo_final_var, 1e-4) * 100:.2f}%",
            "-",
            "-",
            "-"
        ]
    }
    
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    # Convergence rate metric callout
    err_pct = abs(mean_final - theo_final_expected) / max(abs(theo_final_expected), 1e-4) * 100
    if err_pct < 1.0:
        st.success(f"🎯 **High-Precision Convergence:** Empirical Mean is within **{err_pct:.2f}%** of the theoretical expectation ($M = {m:,}$ trials).")
    else:
        st.info(f"ℹ️ **Sampling Variance:** Current Mean deviation is **{err_pct:.2f}%**. Increasing trial count $M$ will further minimize sampling noise via $\mathcal{{O}}(1/\sqrt{{M}})$.")


# ------------------------------------------------------------
# TAB 4: RISK & PROFITABILITY METRICS
# ------------------------------------------------------------
with tab4:
    st.subheader("🛡️ Financial Risk & Quantitative Performance Profiling")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    col_r1.metric(
        "🟢 Probability of Profit",
        f"{prob_profit:.2f}%",
        f"{profitable_trials:,} of {m:,} trials"
    )
    
    col_r2.metric(
        "🔴 Probability of Loss",
        f"{prob_loss:.2f}%",
        f"{losing_trials:,} of {m:,} trials"
    )
    
    col_r3.metric(
        "⚠️ Value at Risk (95% VaR)",
        f"₹{var_95:.2f}",
        f"Loss: ₹{max(0.0, s0 - var_95):.2f}"
    )
    
    col_r4.metric(
        "📉 Average Max Drawdown",
        f"{mean_mdd:.2f}%",
        "Peak-to-trough drop"
    )
    
    st.markdown("---")
    
    col_pie, col_dd = st.columns(2)
    
    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4), facecolor='none')
        labels = ['Profitable (> S₀)', 'Loss (< S₀)', 'Break-even (= S₀)']
        sizes = [profitable_trials, losing_trials, breakeven_trials]
        colors = ['#2E7D32', '#C62828', '#FFB300']
        explode = (0.05, 0.05, 0)
        
        # Filter out 0 slices
        filtered = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0]
        if filtered:
            f_labels, f_sizes, f_colors, f_explode = zip(*filtered)
            ax_pie.pie(
                f_sizes, labels=f_labels, autopct='%1.1f%%',
                colors=f_colors, explode=f_explode, startangle=140,
                textprops={'fontsize': 9, 'color': 'white' if st.get_option("theme.base") == "dark" else 'black'}
            )
        ax_pie.set_title("Outcome Probability Breakdown", fontsize=11, fontweight='bold')
        st.pyplot(fig_pie)
        
    with col_dd:
        fig_mdd, ax_mdd = plt.subplots(figsize=(6, 4), facecolor='none')
        ax_mdd.set_facecolor('#0E1117' if st.get_option("theme.base") == "dark" else '#F8F9FA')
        
        ax_mdd.hist(max_drawdowns_per_path, bins=25, color='#E53935', alpha=0.7, edgecolor='black')
        ax_mdd.axvline(mean_mdd, color='#FFD600', linestyle='--', linewidth=2, label=f"Mean MDD = {mean_mdd:.1f}%")
        ax_mdd.set_title("Maximum Drawdown (MDD) Distribution", fontsize=11, fontweight='bold')
        ax_mdd.set_xlabel("Max Drawdown (%)", fontsize=10)
        ax_mdd.set_ylabel("Frequency", fontsize=10)
        ax_mdd.grid(True, linestyle='--', alpha=0.3)
        ax_mdd.legend(fontsize=9)
        st.pyplot(fig_mdd)


# ------------------------------------------------------------
# TAB 5: MATHEMATICAL PROOFS & THEORY
# ------------------------------------------------------------
with tab5:
    st.subheader("🧮 Mathematical Foundations & Formal Proofs")
    
    st.markdown(r"""
    ### 1. The One-Dimensional Random Walk Model
    Let $\{X_t\}_{t=1}^n$ be a sequence of independent and identically distributed (**i.i.d.**) discrete random variables defining price changes:
    
    $$
    X_t = \begin{cases} 
    +d & \text{with probability } p \\
    -d & \text{with probability } 1-p 
    \end{cases}
    $$
    
    The cumulative price process after $n$ discrete periods is:
    
    $$S_n = S_0 + \sum_{t=1}^n X_t$$
    
    ---
    
    ### 2. Derivation of Expected Value $\mathbb{E}[S_n]$
    For an individual step $X_t$:
    $$\mathbb{E}[X_t] = (+d) \cdot p + (-d) \cdot (1 - p) = d(p - 1 + p) = d(2p - 1)$$
    
    By linearity of expectation:
    $$\mathbb{E}[S_n] = \mathbb{E}\left[S_0 + \sum_{t=1}^n X_t\right] = S_0 + \sum_{t=1}^n \mathbb{E}[X_t] = S_0 + n \cdot d(2p - 1)$$
    
    - If $p = 0.5 \implies \mathbb{E}[S_n] = S_0$ (**Fair Game / Martingale**)
    - If $p > 0.5 \implies \mathbb{E}[S_n] > S_0$ (**Positive Drift / Submartingale**)
    - If $p < 0.5 \implies \mathbb{E}[S_n] < S_0$ (**Negative Drift / Supermartingale**)
    
    ---
    
    ### 3. Derivation of Variance $\text{Var}(S_n)$ and Volatility $\sigma(S_n)$
    Since $\mathbb{E}[X_t^2] = (+d)^2 p + (-d)^2 (1-p) = d^2(p + 1 - p) = d^2$:
    $$\text{Var}(X_t) = \mathbb{E}[X_t^2] - (\mathbb{E}[X_t])^2 = d^2 - [d(2p-1)]^2 = d^2 [1 - (4p^2 - 4p + 1)] = 4d^2 p(1 - p)$$
    
    Due to pairwise independence of $\{X_t\}$:
    $$\text{Var}(S_n) = \sum_{t=1}^n \text{Var}(X_t) = 4n d^2 p(1 - p)$$
    
    $$\sigma(S_n) = \sqrt{\text{Var}(S_n)} = 2d \sqrt{n p (1 - p)}$$
    
    *Notice that the standard deviation scales with the square root of time ($\sigma \propto \sqrt{n}$), the fundamental law of Brownian diffusion.*
    
    ---
    
    ### 4. Asymptotic Convergence to the Normal Distribution (Central Limit Theorem)
    By the **Lindeberg-Lévy Central Limit Theorem**, as $n \to \infty$:
    $$\frac{S_n - \mathbb{E}[S_n]}{\sqrt{\text{Var}(S_n)}} \xrightarrow{d} \mathcal{N}(0, 1)$$
    
    Hence, for sufficiently large $n$, $S_n \sim \mathcal{N}\left(S_0 + nd(2p-1), \; 4nd^2p(1-p)\right)$.
    """)


# ------------------------------------------------------------
# TAB 6: EXPORT SIMULATION DATA
# ------------------------------------------------------------
with tab6:
    st.subheader("💾 Export Simulation Trajectories & Big Data Summaries")
    
    st.markdown("Download high-throughput simulation datasets for external big data analytics, Excel modeling, or academic reporting.")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("#### 📄 Trajectory Paths Dataset (CSV)")
        st.caption(f"Export full matrix of {m:,} trials × {n+1} time steps.")
        
        # Prepare DataFrame for paths
        paths_df = pd.DataFrame(paths, columns=[f"Step_{t}" for t in range(n + 1)])
        paths_df.insert(0, "Trial_ID", np.arange(1, m + 1))
        
        csv_buffer = io.StringIO()
        # Limit download size if very large
        sample_export = paths_df.head(1000) if m > 1000 else paths_df
        sample_export.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label=f"📥 Download Trajectories CSV ({'First 1,000' if m > 1000 else 'All'} Trials)",
            data=csv_buffer.getvalue(),
            file_name=f"stock_random_walk_trajectories_M{m}_N{n}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_exp2:
        st.markdown("#### 📊 Statistical Summary Report (CSV)")
        st.caption("Export summary metrics and distribution parameters.")
        
        summary_df = pd.DataFrame([
            {"Parameter": "Initial Price (S0)", "Value": s0},
            {"Parameter": "Trials (M)", "Value": m},
            {"Parameter": "Steps (n)", "Value": n},
            {"Parameter": "Price Step Increment (d)", "Value": d},
            {"Parameter": "Upward Probability (p)", "Value": p},
            {"Parameter": "Empirical Mean Final Price", "Value": round(mean_final, 4)},
            {"Parameter": "Theoretical Expected Value", "Value": round(theo_final_expected, 4)},
            {"Parameter": "Empirical Std Deviation", "Value": round(std_final, 4)},
            {"Parameter": "Theoretical Std Deviation", "Value": round(theo_final_std, 4)},
            {"Parameter": "Minimum Final Price", "Value": round(min_final, 4)},
            {"Parameter": "Maximum Final Price", "Value": round(max_final, 4)},
            {"Parameter": "Profit Probability (%)", "Value": round(prob_profit, 2)},
            {"Parameter": "Loss Probability (%)", "Value": round(prob_loss, 2)},
            {"Parameter": "Value at Risk 95%", "Value": round(var_95, 2)},
            {"Parameter": "Mean Max Drawdown (%)", "Value": round(mean_mdd, 2)}
        ])
        
        summary_csv = io.StringIO()
        summary_df.to_csv(summary_csv, index=False)
        
        st.download_button(
            label="📥 Download Statistical Summary CSV",
            data=summary_csv.getvalue(),
            file_name=f"simulation_summary_report_M{m}_N{n}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<center><small style='color: #888;'>Mathematical Foundation of Big Data Analytics Mini-Project &bull; Vectorized NumPy Monte Carlo Simulation Framework</small></center>",
    unsafe_allow_html=True
)
