import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Price Random Walk",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 One-Dimensional Stock Price Random Walk")
st.markdown(
    """
    **Mathematical Foundation of Big Data Mini Project**

    This application simulates stock price movements using a
    one-dimensional random walk and performs multiple trials
    to analyze the resulting price distribution.
    """
)

st.divider()


# ============================================================
# SIDEBAR - SIMULATION PARAMETERS
# ============================================================

st.sidebar.header("⚙️ Simulation Parameters")

initial_price = st.sidebar.slider(
    "Initial Stock Price (₹)",
    min_value=10,
    max_value=1000,
    value=100,
    step=10
)

trials = st.sidebar.slider(
    "Number of Trials",
    min_value=10,
    max_value=5000,
    value=1000,
    step=10
)

steps = st.sidebar.slider(
    "Number of Time Steps",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

price_change = st.sidebar.slider(
    "Price Change per Step (₹)",
    min_value=1,
    max_value=20,
    value=1,
    step=1
)

prob_up = st.sidebar.slider(
    "Probability of Price Increase",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

prob_down = 1 - prob_up

st.sidebar.write(f"Probability of decrease: **{prob_down:.2f}**")

run_simulation = st.sidebar.button(
    "🚀 Generate Simulation",
    use_container_width=True
)


# ============================================================
# SIMULATION FUNCTION
# ============================================================

def simulate_random_walk(
    initial_price,
    trials,
    steps,
    price_change,
    prob_up
):
    """
    Generates multiple one-dimensional stock price random walks.
    """

    # Generate random numbers between 0 and 1
    random_values = np.random.random((trials, steps))

    # Determine whether each movement is UP or DOWN
    movements = np.where(
        random_values < prob_up,
        price_change,
        -price_change
    )

    # Create array to store stock prices
    stock_prices = np.zeros((trials, steps + 1))

    # Every trial starts at the initial stock price
    stock_prices[:, 0] = initial_price

    # Calculate cumulative price movement
    stock_prices[:, 1:] = (
        initial_price +
        np.cumsum(movements, axis=1)
    )

    return stock_prices


# ============================================================
# RUN SIMULATION
# ============================================================

if run_simulation or "stock_prices" not in st.session_state:

    st.session_state.stock_prices = simulate_random_walk(
        initial_price,
        trials,
        steps,
        price_change,
        prob_up
    )

    st.session_state.parameters = {
        "initial_price": initial_price,
        "trials": trials,
        "steps": steps,
        "price_change": price_change,
        "prob_up": prob_up
    }


# Retrieve simulation data
stock_prices = st.session_state.stock_prices
params = st.session_state.parameters

final_prices = stock_prices[:, -1]


# ============================================================
# STATISTICAL CALCULATIONS
# ============================================================

mean_final = np.mean(final_prices)
median_final = np.median(final_prices)
variance = np.var(final_prices)
std_dev = np.std(final_prices)
minimum = np.min(final_prices)
maximum = np.max(final_prices)

mean_path = np.mean(stock_prices, axis=0)


# ============================================================
# KEY PARAMETERS
# ============================================================

st.subheader("📋 Simulation Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Initial Price",
    f"₹{params['initial_price']}"
)

col2.metric(
    "Trials",
    f"{params['trials']:,}"
)

col3.metric(
    "Time Steps",
    f"{params['steps']:,}"
)

col4.metric(
    "Price Change",
    f"₹{params['price_change']}"
)


st.divider()


# ============================================================
# RANDOM WALK GRAPH
# ============================================================

st.subheader("📈 Multiple Random Walks")

fig1, ax1 = plt.subplots(figsize=(12, 6))

# Plot each trial
for i in range(len(stock_prices)):
    ax1.plot(
        stock_prices[i],
        alpha=0.03
    )

# Plot mean path
ax1.plot(
    mean_path,
    linewidth=3,
    label="Mean Path"
)

# Initial price line
ax1.axhline(
    params["initial_price"],
    linestyle="--",
    linewidth=2,
    label="Initial Price"
)

ax1.set_xlabel("Time Step")
ax1.set_ylabel("Stock Price (₹)")
ax1.set_title(
    f"{params['trials']:,} Simulated Stock Price Random Walks"
)

ax1.legend()
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)


st.info(
    """
    Each line represents one possible stock-price path.
    The paths start from the same initial price but move
    differently because each movement is random.
    """
)


# ============================================================
# STATISTICS
# ============================================================

st.subheader("📊 Statistical Analysis")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean Final Price",
    f"₹{mean_final:.2f}"
)

col2.metric(
    "Median Final Price",
    f"₹{median_final:.2f}"
)

col3.metric(
    "Standard Deviation",
    f"{std_dev:.2f}"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Variance",
    f"{variance:.2f}"
)

col5.metric(
    "Minimum Final Price",
    f"₹{minimum:.2f}"
)

col6.metric(
    "Maximum Final Price",
    f"₹{maximum:.2f}"
)


st.divider()


# ============================================================
# FINAL PRICE DISTRIBUTION
# ============================================================

st.subheader("📊 Distribution of Final Stock Prices")

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.hist(
    final_prices,
    bins=20,
    edgecolor="black"
)

ax2.axvline(
    mean_final,
    linestyle="--",
    linewidth=2,
    label=f"Mean = ₹{mean_final:.2f}"
)

ax2.set_xlabel("Final Stock Price (₹)")
ax2.set_ylabel("Number of Trials")
ax2.set_title("Distribution of Final Stock Prices")

ax2.legend()
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)


# ============================================================
# MATHEMATICAL FOUNDATION
# ============================================================

st.divider()

st.subheader("🧮 Mathematical Foundation")

st.markdown(
    r"""
    The stock price follows a one-dimensional random walk.

    The price at the next time step is given by:

    $$S_{t+1} = S_t + X_t$$

    where:

    $$X_t =
    \begin{cases}
    +d & \text{with probability } p \\
    -d & \text{with probability } 1-p
    \end{cases}
    $$

    Here:

    - $S_t$ = stock price at time $t$
    - $d$ = price change per step
    - $p$ = probability of price increase
    - $1-p$ = probability of price decrease

    After $n$ steps:

    $$S_n = S_0 + \sum_{t=1}^{n} X_t$$
    """
)


# ============================================================
# OBSERVATIONS
# ============================================================

st.subheader("🔍 Observations")

st.write(
    f"""
    • The simulation starts with a stock price of ₹{params['initial_price']}.

    • {params['trials']:,} independent random walk trials were generated.

    • Each trial contains {params['steps']:,} time steps.

    • The probability of an increase is {params['prob_up']:.2f},
      while the probability of a decrease is {1 - params['prob_up']:.2f}.

    • The mean final stock price obtained from the simulation is
      ₹{mean_final:.2f}.

    • The standard deviation of final prices is {std_dev:.2f}.

    • Increasing the number of trials provides more observations
      and gives a more stable estimate of the overall behavior.
    """
)


# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.subheader("✅ Conclusion")

st.markdown(
    """
    The simulation demonstrates how a stock price can be modeled
    as a one-dimensional random walk. By performing multiple trials,
    we can visualize different possible price paths and analyze
    their statistical properties using mean, variance and
    standard deviation.

    The project demonstrates the application of probability,
    random variables, statistical analysis and data visualization
    in a Big Data context.
    """
)
