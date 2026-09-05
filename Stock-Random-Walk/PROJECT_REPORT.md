# Mathematical Foundation of Big Data (BDA) Mini-Project
# One-Dimensional Stock Price Random Walk Simulation

---

## 1. Abstract
This project presents a computational and mathematical simulation of stock price dynamics using a **one-dimensional discrete random walk model**. Implemented in Python with Streamlit, NumPy, and Matplotlib, the application generates multiple independent stochastic paths to analyze the statistical behavior, convergence, and price distributions under varying probabilities and time horizons. The project illustrates core concepts in Big Data Analytics including probability distributions, stochastic processes, Monte Carlo trials, and the **Law of Large Numbers**.

---

## 2. Project Objectives
- To model asset price trajectories using a discrete 1D random walk.
- To analyze how probability asymmetry ($p \neq 0.5$) induces drift (trend) in asset prices.
- To demonstrate statistical aggregation (Mean, Variance, Standard Deviation) over large-scale trials ($N = 10 \text{ to } 5000$).
- To verify the **Law of Large Numbers (LLN)** and the **Central Limit Theorem (CLT)** empirically via final price distributions.

---

## 3. Mathematical Foundation

### 3.1 Discrete Random Walk Formulation
Let $S_t$ denote the stock price at discrete time step $t \in \{0, 1, 2, \dots, n\}$.
The price evolves according to:
$$S_{t+1} = S_t + X_t$$

Where the step increment $X_t$ is an independent and identically distributed (i.i.d.) random variable:
$$X_t = \begin{cases} +d & \text{with probability } p \\ -d & \text{with probability } 1-p \end{cases}$$

Here:
- $S_0$: Initial stock price (e.g., ₹100)
- $d$: Price step change (₹)
- $p$: Probability of price increase ($P(\text{UP})$)
- $1-p$: Probability of price decrease ($P(\text{DOWN})$)

### 3.2 Price after $n$ Steps
After $n$ discrete steps, the cumulative price is:
$$S_n = S_0 + \sum_{t=1}^{n} X_t$$

### 3.3 Expected Value and Variance
1. **Expected Step Increment**:
   $$E[X_t] = (+d)p + (-d)(1-p) = d(2p - 1)$$
2. **Expected Final Stock Price**:
   $$E[S_n] = S_0 + n \cdot E[X_t] = S_0 + n \cdot d(2p - 1)$$
   - When $p = 0.5$ (Symmetric / Martingale property): $E[S_n] = S_0$ (Zero drift).
   - When $p > 0.5$: Positive drift (Upward trend).
   - When $p < 0.5$: Negative drift (Downward trend).
3. **Variance of Final Price**:
   $$\text{Var}(X_t) = E[X_t^2] - (E[X_t])^2 = d^2 - [d(2p-1)]^2 = 4d^2 p(1-p)$$
   $$\text{Var}(S_n) = n \cdot \text{Var}(X_t) = 4 n d^2 p(1-p)$$
   $$\sigma(S_n) = 2d \sqrt{n p (1-p)}$$

---

## 4. System Architecture & Flowchart

```text
               +----------------------------------+
               | User Interface (Streamlit)       |
               | - Initial Price (S0)             |
               | - Trials (M), Time Steps (N)     |
               | - Step Change (d), Prob (p)      |
               +-----------------+----------------+
                                 |
                                 v
               +----------------------------------+
               | Stochastic Simulation Engine     |
               | NumPy 2D Array Matrix            |
               | - Uniform Random Gen (M x N)     |
               | - Vectorized Thresholding        |
               | - Cumulative Summation (np.cumsum|
               +-----------------+----------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
+-------------------------------+             +-------------------------------+
| Visualizations (Matplotlib)   |             | Statistical Aggregation       |
| - M Paths Overlay (Alpha 0.03)|             | - Mean Final Price            |
| - Empirical Mean Trajectory   |             | - Variance & Std Deviation    |
| - Final Price Histogram (CLT) |             | - Min & Max Realizations      |
+-------------------------------+             +-------------------------------+
```

---

## 5. Implementation Details
- **Vectorized Generation**: Utilizes NumPy's 2D array generation `np.random.random((trials, steps))` for high computational efficiency, eliminating slow Python `for`-loops during path generation.
- **Fast Cumulative Path Calculation**: `np.cumsum(movements, axis=1)` computes all $M \times N$ discrete coordinates simultaneously.
- **Interactive UI**: Sliders and real-time metric cards built using Streamlit.

---

## 6. Experimental Observations & Viva Demonstration

### Experiment 1: Symmetric Walk ($p = 0.5$)
- **Setup**: $S_0 = 100$, $N = 100$, $M = 1000$, $p = 0.5$.
- **Observation**: The mean final price closely hovers around ₹100. The distribution is bell-shaped (Binomial approximating Gaussian per CLT).

### Experiment 2: Upward Bias ($p = 0.8$) vs. Downward Bias ($p = 0.2$)
- **Observation**: At $p = 0.8$, $E[X_t] = d(1.6 - 1) = +0.6d > 0$, causing trajectories and mean curve to slope upward. At $p = 0.2$, paths trend downwards.

### Experiment 3: Effect of Sample Size (Trials $M = 10 \to 1000$)
- **Observation**: Demonstrates the **Law of Large Numbers (LLN)**. At $M=10$, the empirical mean deviates noticeably from the theoretical expectation. At $M=1000$, empirical mean matches theoretical $E[S_n]$ with minimal error.

---

## 7. Viva Q&A Cheat Sheet

1. **Q: Is this a stock market price predictor?**
   - *A:* No. It is a mathematical stochastic simulation of a 1D discrete random walk. It models the behavior of random variables, drift, and probability distributions, not real-world market microstructure.

2. **Q: How does this relate to Big Data Analytics (BDA)?**
   - *A:* It covers foundational concepts in BDA: Monte Carlo simulation, statistical aggregation over large datasets, probability mass functions, convergence theorems (LLN and CLT), and interactive visualization.

3. **Q: Why does the distribution look normal (bell-shaped)?**
   - *A:* According to the **Central Limit Theorem**, the sum of a large number of independent step increments $X_t$ converges to a Normal distribution $\mathcal{N}(\mu, \sigma^2)$.

4. **Q: What is the computational complexity?**
   - *A:* Generating the random matrix is $\mathcal{O}(M \times N)$, and the cumulative sum is $\mathcal{O}(M \times N)$, which runs in milliseconds using NumPy C-extensions.
