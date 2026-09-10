# 📈 One-Dimensional Stock Price Random Walk Simulation
### *Mathematical Foundation of Big Data Analytics (BDA) — Mini-Project*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Export-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Modeling-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features & Enhancements](#-key-features--enhancements)
- [Mathematical Formulation](#-mathematical-formulation)
  - [1. Discrete Stochastic Walk](#1-discrete-stochastic-walk)
  - [2. Expected Value \& Drift Analysis](#2-expected-value--drift-analysis)
  - [3. Variance \& Standard Deviation](#3-variance--standard-deviation)
  - [4. Central Limit Theorem (CLT) Convergence](#4-central-limit-theorem-clt-convergence)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation \& Quickstart](#-installation--quickstart)
- [Application Modules](#-application-modules)
- [Experimental Case Studies](#-experimental-case-studies)
- [Viva \& Interview Q\&A](#-viva--interview-qa)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview

This project implements a computational and mathematical simulation of stock price dynamics using a **one-dimensional discrete random walk model** and **Geometric Brownian Motion (GBM)**. Powered by **Streamlit**, **NumPy**, **Matplotlib**, **Pandas**, and **SciPy**, the application performs high-throughput Monte Carlo simulations across thousands of stochastic trajectories to analyze price convergence, drift dynamics, risk metrics, and probability distributions.

The project demonstrates foundational Big Data Analytics (BDA) principles:
1. **Stochastic Processes & Monte Carlo Methods**
2. **The Law of Large Numbers (LLN)**
3. **The Central Limit Theorem (CLT)**
4. **Vectorized Matrix Computation**

---

## ✨ Key Features & Enhancements

- **⚡ Fully Vectorized Simulation Engine**: Leverages NumPy 2D array broadcasting and `np.cumsum` to simulate up to 5,000 independent price paths across hundreds of time steps in milliseconds without Python loop bottlenecks.
- **🎯 Scenario Presets**: Instant one-click selection for:
  - ⚖️ *Fair Market (Martingale, $p=0.50$)*
  - 🐂 *Bullish Market (Positive Drift, $p=0.60$)*
  - 🐻 *Bearish Market (Negative Drift, $p=0.40$)*
  - ⚡ *High Volatility Speculation ($d=5, p=0.52$)*
  - 🔬 *Large-Scale Monte Carlo ($M=3000, n=200$)*
- **📈 Quantile Fan Chart**: Shaded 10th–90th and 25th–75th percentile confidence diffusion cones showing variance growth over time ($\sigma \propto \sqrt{t}$).
- **📊 CLT Normal Distribution Overlay**: Overlays the theoretical continuous Gaussian probability density function $\mathcal{N}(\mathbb{E}[S_n], \text{Var}(S_n))$ atop the empirical histogram with color-coded profit/loss regions.
- **⚖️ Theoretical vs. Empirical Convergence Suite**: Computes real-time analytical equations vs. Monte Carlo realizations, reporting absolute and percentage error rates.
- **🛡️ Quantitative Risk & Profitability Profiling**:
  - Probability of Profit ($S_n > S_0$), Loss ($S_n < S_0$), and Break-even.
  - 95% and 99% **Value at Risk (VaR)** calculation.
  - Path-level **Maximum Drawdown (MDD)** distribution.
  - Skewness and Excess Kurtosis normality diagnostics.
- **💾 Big Data Export**: Download full trajectory matrices ($M \times n$) and summary statistics as CSV for external data analysis and academic reporting.

---

## 🧮 Mathematical Formulation

### 1. Discrete Stochastic Walk

Let $S_t$ denote the stock price at discrete time step $t \in \{0, 1, 2, \dots, n\}$.

The recurrence relation governing price updates is:
$$S_{t+1} = S_t + X_t$$

Where step increment $X_t$ is an independent and identically distributed (i.i.d.) random variable:
$$X_t = \begin{cases} +d & \text{with probability } p \\ -d & \text{with probability } 1-p \end{cases}$$

After $n$ discrete time steps, the cumulative price is:
$$S_n = S_0 + \sum_{t=1}^{n} X_t$$

---

### 2. Expected Value & Drift Analysis

The expected value of an individual step increment $X_t$:
$$\mathbb{E}[X_t] = (+d)p + (-d)(1-p) = d(2p - 1)$$

The expected terminal stock price after $n$ steps:
$$\mathbb{E}[S_n] = S_0 + n \cdot \mathbb{E}[X_t] = S_0 + n \cdot d(2p - 1)$$

| Probability Condition | Market Drift Regime | Mathematical Behavior |
| :--- | :--- | :--- |
| **$p = 0.5$** | **Zero Drift (Martingale)** | $\mathbb{E}[S_n] = S_0$. Price fluctuates symmetrically around initial price. |
| **$p > 0.5$** | **Positive Drift (Bullish / Submartingale)** | $\mathbb{E}[S_n] > S_0$. Price trajectory slopes upward on average. |
| **$p < 0.5$** | **Negative Drift (Bearish / Supermartingale)** | $\mathbb{E}[S_n] < S_0$. Price trajectory slopes downward on average. |

---

### 3. Variance & Standard Deviation

The variance of single step increment $X_t$:
$$\text{Var}(X_t) = \mathbb{E}[X_t^2] - (\mathbb{E}[X_t])^2 = d^2 - [d(2p - 1)]^2 = 4d^2 p(1 - p)$$

Because individual steps $X_1, X_2, \dots, X_n$ are mutually independent:
$$\text{Var}(S_n) = \sum_{t=1}^{n} \text{Var}(X_t) = 4 n d^2 p(1 - p)$$

$$\sigma(S_n) = \sqrt{\text{Var}(S_n)} = 2 d \sqrt{n p (1 - p)}$$

---

### 4. Central Limit Theorem (CLT) Convergence

By the **Lindeberg-Lévy Central Limit Theorem**, as the number of independent steps $n \to \infty$:
$$\frac{S_n - \mathbb{E}[S_n]}{\sqrt{\text{Var}(S_n)}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Consequently:
$$S_n \sim \mathcal{N}\left(S_0 + n \cdot d(2p - 1), \; 4 n d^2 p(1 - p)\right)$$

---

## 🏗️ System Architecture

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

## 📂 Repository Structure

```bash
BDA/
├── README.md                      # Main project documentation & overview
└── Stock-Random-Walk/
    ├── app.py                     # Streamlit application & simulation core
    ├── PROJECT_REPORT.md          # Comprehensive mathematical project report & viva sheet
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Module-level documentation
```

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
- **Python 3.8+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/RIYA5566/BDA_Prac.git
cd BDA_Prac
```

### 3. Install Dependencies
```bash
pip install -r Stock-Random-Walk/requirements.txt
```

### 4. Launch the Application
```bash
streamlit run Stock-Random-Walk/app.py
```
The interactive web dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 📑 Application Modules

The application interface is organized into six specialized tabs:
1. **📈 Trajectories & Fan Chart**: Multi-path visualization with 10th-90th and IQR percentile bands, peak, trough, and mean trajectories.
2. **📊 Distribution & CLT Normal Fit**: Normalized histogram of terminal prices with overlaid theoretical Gaussian curve $\mathcal{N}(\mu, \sigma^2)$ and profit/loss color shading.
3. **⚖️ Theoretical vs Empirical (LLN)**: Mathematical verification table comparing theoretical formulas vs. empirical results and computing convergence error rates.
4. **🛡️ Risk & Profitability Metrics**: Probability of profit/loss/break-even pie chart, 95% Value at Risk (VaR), and Maximum Drawdown (MDD) distribution.
5. **🧮 Mathematical Proofs & Theory**: Complete LaTeX formulas, step-by-step variance derivations, drift classifications, and CLT proofs.
6. **💾 Export Simulation Data**: Download generated simulation matrices and statistical summaries as CSV files.

---

## 🔬 Experimental Case Studies

### 🔹 Experiment 1: Symmetric Walk ($p = 0.5$)
- **Configuration**: $S_0 = 100$, $n = 100$, $M = 1000$, $p = 0.5$, $d = 1$
- **Expected Outcome**: $\mathbb{E}[S_n] = 100$.
- **Observed Result**: Final mean price converges closely around ₹100. The terminal price distribution is bell-shaped (Binomial approaching Gaussian).

### 🔹 Experiment 2: Upward Bias ($p = 0.8$) vs. Downward Bias ($p = 0.2$)
- **Configuration**: $S_0 = 100$, $n = 100$, $M = 1000$, $d = 1$
- **Observed Result**:
  - For $p = 0.8$, $\mathbb{E}[S_n] = 100 + 100 \times 1 \times (1.6 - 1) = 160$. Trajectories consistently drift upward.
  - For $p = 0.2$, $\mathbb{E}[S_n] = 100 + 100 \times 1 \times (0.4 - 1) = 40$. Trajectories trend downward.

### 🔹 Experiment 3: Verification of the Law of Large Numbers (LLN)
- **Configuration**: Compare $M = 10$ trials vs. $M = 3000$ trials.
- **Observed Result**: Small sample sizes ($M=10$) show high empirical error from theoretical expectation. Scaling to $M \ge 1000$ stabilizes the sample mean to within $\pm 0.5\%$ of theoretical $\mathbb{E}[S_n]$.

---

## 🎓 Viva & Interview Q&A

<details>
<summary><b>1. Is this application a real-world stock market price predictor?</b></summary>
<br>
<b>No.</b> It is a mathematical stochastic simulation modeling a 1D discrete random walk. It demonstrates the statistical behavior of random variables, drift, variance scaling, and probability distributions rather than modeling market microstructure or order book dynamics.
</details>

<details>
<summary><b>2. How does this project connect to Big Data Analytics (BDA)?</b></summary>
<br>
The project demonstrates core BDA fundamentals:
<ul>
  <li><b>Monte Carlo Methods:</b> Solving analytical problems by performing repeated stochastic trials.</li>
  <li><b>High-Throughput Vectorization:</b> Eliminating iterative procedural loops in favor of matrix-based SIMD computations.</li>
  <li><b>Asymptotic Convergence:</b> Demonstrating empirical adherence to the Law of Large Numbers (LLN) and Central Limit Theorem (CLT) across big data samples.</li>
</ul>
</details>

<details>
<summary><b>3. Why does the terminal price histogram look bell-shaped?</b></summary>
<br>
By the <b>Central Limit Theorem (CLT)</b>, the sum of a large number of independent and identically distributed (i.i.d.) random variables $X_t$ converges in distribution to a Normal Distribution $\mathcal{N}(\mu, \sigma^2)$ as $n \to \infty$, regardless of the underlying discrete binomial distribution of individual steps.
</details>

<details>
<summary><b>4. What is the time complexity of the simulation?</b></summary>
<br>
Both the 2D random matrix generation and the cumulative summation along the horizontal axis operate in $\mathcal{O}(M \times n)$ time complexity. Executed through C-backed NumPy vectorization, even 5,000 trials × 500 steps (2.5 million points) computes in under 30 milliseconds.
</details>

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Interactive Framework:** [Streamlit](https://streamlit.io/)
- **Numerical Computation:** [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/)
- **Data Processing & Export:** [Pandas](https://pandas.pydata.org/)
- **Statistical Modeling:** [SciPy](https://scipy.org/)

---

## 📄 License

This project is licensed under the MIT License.
