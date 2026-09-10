# 📈 One-Dimensional Stock Price Random Walk Simulation
### *Mathematical Foundation of Big Data Analytics (BDA) — Mini-Project*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Export-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Modeling-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Enhancements & Features](#-key-enhancements--features)
- [Mathematical Formulation](#-mathematical-formulation)
  - [1. Discrete Stochastic Walk](#1-discrete-stochastic-walk)
  - [2. Expected Value \& Drift Analysis](#2-expected-value--drift-analysis)
  - [3. Variance \& Standard Deviation](#3-variance--standard-deviation)
  - [4. Central Limit Theorem (CLT) Convergence](#4-central-limit-theorem-clt-convergence)
- [System Architecture](#-system-architecture)
- [Installation \& Quickstart](#-installation--quickstart)
- [Application Modules](#-application-modules)
- [Experimental Case Studies](#-experimental-case-studies)
- [Viva \& Interview Q\&A](#-viva--interview-qa)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview

This application simulates stock price movements using a **one-dimensional discrete random walk model** and **Geometric Brownian Motion (GBM)**, executing thousands of high-throughput Monte Carlo trials to analyze statistical convergence, price drift, risk metrics, and probability distributions.

The project demonstrates foundational Big Data Analytics (BDA) principles:
1. **Stochastic Processes & Monte Carlo Methods**: Solving analytical expectations via vectorized simulation.
2. **The Law of Large Numbers (LLN)**: Verifying empirical convergence of sample means and variances to theoretical limits.
3. **The Central Limit Theorem (CLT)**: Demonstrating how sums of discrete random steps asymptotically form a continuous Gaussian bell curve.
4. **Vectorized Matrix Computation**: SIMD-accelerated array broadcasting in NumPy eliminating procedural bottlenecks.

---

## ✨ Key Enhancements & Features

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

## 🚀 Installation & Quickstart

### 1. Prerequisites
- **Python 3.8+**

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 📑 Application Modules

The application interface is structured across six dedicated analytical modules:
1. **📈 Trajectories & Fan Chart**: Multi-path visualization with 10th-90th and IQR percentile bands, peak, trough, and mean trajectories.
2. **📊 Distribution & CLT Normal Fit**: Normalized histogram of terminal prices with overlaid theoretical Gaussian curve $\mathcal{N}(\mu, \sigma^2)$ and profit/loss color shading.
3. **⚖️ Theoretical vs Empirical (LLN)**: Mathematical verification table comparing theoretical formulas vs. empirical results and computing convergence error rates.
4. **🛡️ Risk & Profitability Metrics**: Probability of profit/loss/break-even pie chart, 95% Value at Risk (VaR), and Maximum Drawdown (MDD) distribution.
5. **🧮 Mathematical Proofs & Theory**: Complete LaTeX formulas, step-by-step variance derivations, drift classifications, and CLT proofs.
6. **💾 Export Simulation Data**: Download generated simulation matrices and statistical summaries as CSV files.

---

## 🔬 Experimental Case Studies

- **Experiment 1: Symmetric Walk ($p = 0.5$)**: Theoretical expectation $\mathbb{E}[S_n] = S_0$. Price distribution is symmetric and centered at $S_0$.
- **Experiment 2: Upward ($p = 0.8$) vs. Downward ($p = 0.2$) Bias**: Clear drift patterns matching theoretical $\mathbb{E}[S_n] = S_0 + n \cdot d(2p-1)$.
- **Experiment 3: Law of Large Numbers ($M = 10 \to 5000$)**: Sample mean converges tightly to mathematical expectation as trial count $M$ increases, with percentage error approaching $0\%$.

---

## 🎓 Viva & Interview Q&A

1. **Q: Is this application a real-world stock market price predictor?**  
   *A:* No. It is a mathematical stochastic simulation modeling a 1D discrete random walk to study probability theory, drift, variance scaling, and distributions.

2. **Q: How does this connect to Big Data Analytics (BDA)?**  
   *A:* It demonstrates Monte Carlo simulations, high-performance vectorized operations on large arrays, and asymptotic convergence theorems (LLN & CLT).

3. **Q: Why does the terminal price histogram look bell-shaped?**  
   *A:* By the Central Limit Theorem (CLT), the sum of independent random steps converges to a Normal distribution $\mathcal{N}(\mu, \sigma^2)$ as $n$ grows.

4. **Q: What is the computational time complexity?**  
   *A:* $\mathcal{O}(M \times n)$ for both random array generation and cumulative summation, computed in milliseconds via NumPy C-extensions.

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Interactive Framework:** [Streamlit](https://streamlit.io/)
- **Numerical Computation:** [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/)
- **Data Processing & Export:** [Pandas](https://pandas.pydata.org/)
- **Statistical Modeling:** [SciPy](https://scipy.org/)
