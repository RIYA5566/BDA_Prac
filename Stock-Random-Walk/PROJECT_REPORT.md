# Mathematical Foundation of Big Data Analytics (BDA) Mini-Project
# One-Dimensional Stock Price Random Walk Simulation & Stochastic Modeling

---

## 1. Abstract
This project presents an advanced computational and mathematical simulation platform for analyzing asset price dynamics using a **one-dimensional discrete random walk model** and **Geometric Brownian Motion (GBM)**. Implemented in Python leveraging **Streamlit**, **NumPy**, **Matplotlib**, **Pandas**, and **SciPy**, the application executes high-throughput Monte Carlo simulations across thousands of independent stochastic paths. The framework analyzes statistical convergence, drift properties, quantitative risk metrics (Value at Risk, Maximum Drawdown), and empirical validation of the **Law of Large Numbers (LLN)** and the **Central Limit Theorem (CLT)**.

---

## 2. Project Objectives
- Model discrete asset price trajectories using high-performance vectorized stochastic processes.
- Analyze how probability asymmetry ($p \neq 0.5$) induces systematic drift in price paths.
- Quantitatively evaluate statistical moments (Mean, Median, Variance, Std Deviation, Skewness, Excess Kurtosis) across up to 5,000 Monte Carlo paths.
- Verify asymptotic convergence to the theoretical Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ via real-time CLT probability density overlays.
- Quantify financial risk and profitability parameters (Win/Loss probability ratios, 95% Value at Risk, Maximum Drawdown).
- Provide automated Big Data export capabilities (trajectory matrices and statistical summary reports as CSV).

---

## 3. Mathematical Foundation

### 3.1 Discrete Random Walk Formulation
Let $S_t$ denote the stock price at discrete time step $t \in \{0, 1, 2, \dots, n\}$.
The price evolves according to the recurrence relation:
$$S_{t+1} = S_t + X_t$$

Where the step increment $X_t$ is an independent and identically distributed (i.i.d.) discrete random variable:
$$X_t = \begin{cases} +d & \text{with probability } p \\ -d & \text{with probability } 1-p \end{cases}$$

Here:
- $S_0$: Initial stock price (₹)
- $d$: Step change magnitude per interval (₹)
- $p$: Probability of price increase ($P(\text{UP})$)
- $1-p$: Probability of price decrease ($P(\text{DOWN})$)

### 3.2 Price after $n$ Steps
After $n$ discrete steps, the cumulative price realization is:
$$S_n = S_0 + \sum_{t=1}^{n} X_t$$

### 3.3 Expected Value and Drift Analysis
1. **Expected Step Increment**:
   $$\mathbb{E}[X_t] = (+d)p + (-d)(1-p) = d(2p - 1)$$
2. **Expected Final Stock Price**:
   $$\mathbb{E}[S_n] = S_0 + n \cdot \mathbb{E}[X_t] = S_0 + n \cdot d(2p - 1)$$
   - When $p = 0.5$ (Symmetric Martingale): $\mathbb{E}[S_n] = S_0$ (Zero drift / Fair Game).
   - When $p > 0.5$ (Submartingale): $\mathbb{E}[S_n] > S_0$ (Positive Bullish Drift).
   - When $p < 0.5$ (Supermartingale): $\mathbb{E}[S_n] < S_0$ (Negative Bearish Drift).

### 3.4 Variance and Standard Deviation (Diffusion Scaling)
1. **Step Variance**:
   $$\text{Var}(X_t) = \mathbb{E}[X_t^2] - (\mathbb{E}[X_t])^2 = d^2 - [d(2p-1)]^2 = 4d^2 p(1-p)$$
2. **Cumulative Terminal Variance**:
   $$\text{Var}(S_n) = \sum_{t=1}^n \text{Var}(X_t) = 4 n d^2 p(1-p)$$
3. **Standard Deviation (Volatility Diffusion)**:
   $$\sigma(S_n) = \sqrt{\text{Var}(S_n)} = 2d \sqrt{n p (1-p)}$$
   *Note: Standard deviation scales with the square root of time ($\sigma \propto \sqrt{n}$), the fundamental property of Brownian diffusion.*

### 3.5 Central Limit Theorem (CLT) Convergence
By the **Lindeberg-Lévy Central Limit Theorem**, as the number of independent steps $n \to \infty$:
$$\frac{S_n - \mathbb{E}[S_n]}{\sqrt{\text{Var}(S_n)}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Consequently:
$$S_n \sim \mathcal{N}\left(S_0 + n \cdot d(2p - 1), \; 4 n d^2 p(1 - p)\right)$$

---

## 4. System Architecture

```text
               +------------------------------------------+
               |        Interactive Streamlit UI          |
               | - Initial Price (S0) | Trials (M)        |
               | - Time Steps (N)     | Step Change (d)   |
               | - Probability of Increase (p)            |
               | - Scenario Presets & Model Selection     |
               +--------------------+---------------------+
                                    |
                                    v
               +------------------------------------------+
               |     High-Speed NumPy Simulation Engine   |
               |  1. Generate Uniform Random Matrix (M×N) |
               |  2. Vectorized Thresholding (UP / DOWN)  |
               |  3. Cumulative Path Summation (np.cumsum)|
               +--------------------+---------------------+
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
+-----------------------+   +-----------------------+   +-----------------------+
| Visualizations        |   | Statistical & CLT     |   | Risk & Data Export    |
| - Quantile Fan Bands  |   | - LLN Error Table     |   | - Win/Loss Probabilities|
| - Mean & Min/Max Paths|   | - CLT Normal PDF Fit  |   | - 95% Value at Risk   |
| - Trajectory Sampling |   | - Skewness & Kurtosis |   | - Trajectory CSV / Summary|
+-----------------------+   +-----------------------+   +-----------------------+
```

---

## 5. Implementation & Technical Features
- **Vectorized Generation**: Utilizes NumPy's 2D array generation `np.random.random((trials, steps))` for high computational throughput, eliminating slow Python `for`-loops.
- **Quantile Fan Chart**: Computes 10th, 25th, 75th, and 90th percentile trajectories to visually illustrate variance diffusion over time.
- **Analytical Normal PDF Overlay**: Uses SciPy's `stats.norm.pdf` to directly compare empirical histogram frequencies with theoretical CLT predictions.
- **Quantitative Risk Analysis**: Computes 95% Value at Risk (VaR), Maximum Drawdown (MDD) distributions, and win/loss outcome breakdowns.
- **Data Export**: Built-in CSV generation for simulated path matrices and summary metrics using Pandas and Python IO buffers.

---

## 6. Experimental Observations & Viva Demonstration

### Experiment 1: Symmetric Walk ($p = 0.5$)
- **Setup**: $S_0 = 100$, $n = 100$, $M = 1000$, $p = 0.5, d = 1$.
- **Observation**: The mean final price closely hovers around ₹100. The distribution is bell-shaped (approaching Gaussian per CLT) with skewness near 0.

### Experiment 2: Upward Bias ($p = 0.8$) vs. Downward Bias ($p = 0.2$)
- **Observation**: At $p = 0.8$, $\mathbb{E}[X_t] = d(1.6 - 1) = +0.6d > 0$, causing trajectories and mean curve to slope upward with high probability of profit. At $p = 0.2$, paths trend downwards.

### Experiment 3: Effect of Sample Size (Law of Large Numbers)
- **Observation**: Demonstrates the **Law of Large Numbers (LLN)**. At $M=10$, the empirical mean deviates noticeably from the theoretical expectation. At $M=3000$, empirical mean matches theoretical $\mathbb{E}[S_n]$ with $<0.5\%$ error.

---

## 7. Viva Q&A Cheat Sheet

1. **Q: Is this a stock market price predictor?**
   - *A:* No. It is a mathematical stochastic simulation of a 1D discrete random walk. It models the behavior of random variables, drift, and probability distributions, not real-world market microstructure.

2. **Q: How does this relate to Big Data Analytics (BDA)?**
   - *A:* It covers foundational concepts in BDA: Monte Carlo simulation, statistical aggregation over large datasets, probability mass functions, convergence theorems (LLN and CLT), high-throughput vectorization, and interactive visualization.

3. **Q: Why does the distribution look normal (bell-shaped)?**
   - *A:* According to the **Central Limit Theorem**, the sum of a large number of independent step increments $X_t$ converges to a Normal distribution $\mathcal{N}(\mu, \sigma^2)$.

4. **Q: What is the computational complexity?**
   - *A:* Generating the random matrix is $\mathcal{O}(M \times n)$, and the cumulative sum is $\mathcal{O}(M \times n)$, running in milliseconds via NumPy C-extensions.
