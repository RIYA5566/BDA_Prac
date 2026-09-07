# 📈 One-Dimensional Stock Price Random Walk Simulation
### *Mathematical Foundation of Big Data Analytics (BDA) — Mini-Project*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Mathematical Formulation](#-mathematical-formulation)
  - [1. Discrete Stochastic Walk](#1-discrete-stochastic-walk)
  - [2. Expected Value \& Drift Analysis](#2-expected-value--drift-analysis)
  - [3. Variance \& Standard Deviation](#3-variance--standard-deviation)
- [System Architecture](#-system-architecture)
- [Installation \& Quickstart](#-installation--quickstart)
- [Experimental Case Studies](#-experimental-case-studies)
- [Viva \& Interview Q\&A](#-viva--interview-qa)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview

This application simulates stock price movements using a **one-dimensional discrete random walk model** and performs multiple Monte Carlo trials to analyze statistical properties, price drift, and probability distributions.

The project demonstrates foundational Big Data Analytics (BDA) principles:
1. **Stochastic Processes & Monte Carlo Methods**
2. **The Law of Large Numbers (LLN)**
3. **The Central Limit Theorem (CLT)**
4. **Vectorized Matrix Computation**

---

## ✨ Key Features

- **⚡ Fully Vectorized Simulation Engine**: Leverages NumPy 2D array broadcasting and `np.cumsum` to simulate up to 5,000 independent price paths across hundreds of time steps in milliseconds without Python loop bottlenecks.
- **🎛️ Interactive Parameter Controls**:
  - Initial Stock Price ($S_0 \in [10, 1000]$ ₹)
  - Number of Trials / Monte Carlo Paths ($M \in [10, 5000]$)
  - Time Steps per Trial ($n \in [10, 500]$)
  - Price Change Increment ($d \in [1, 20]$ ₹)
  - Upward Probability ($p \in [0.0, 1.0]$ with automatic complement $1-p$)
- **📊 Real-time Visualizations**:
  - **Multi-Path Trajectory Chart**: All $M$ individual paths rendered with low alpha blending alongside the theoretical/empirical **Mean Path** and initial price baseline.
  - **Final Price Distribution Histogram**: Displays frequency distribution of terminal prices with mean indicators, illustrating binomial/Gaussian convergence.
- **📋 Statistical Summary Suite**: Real-time calculation of Mean, Median, Standard Deviation, Variance, Min, and Max prices.

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

| Probability Condition | Market Drift Regime | Behavior |
| :--- | :--- | :--- |
| **$p = 0.5$** | **Zero Drift (Martingale)** | $\mathbb{E}[S_n] = S_0$. Price fluctuates symmetrically around initial price. |
| **$p > 0.5$** | **Positive Drift (Bullish)** | $\mathbb{E}[S_n] > S_0$. Price trajectory slopes upward on average. |
| **$p < 0.5$** | **Negative Drift (Bearish)** | $\mathbb{E}[S_n] < S_0$. Price trajectory slopes downward on average. |

---

### 3. Variance & Standard Deviation

The variance of single step increment $X_t$:
$$\text{Var}(X_t) = \mathbb{E}[X_t^2] - (\mathbb{E}[X_t])^2 = d^2 - [d(2p - 1)]^2 = 4d^2 p(1 - p)$$

Because individual steps $X_1, X_2, \dots, X_n$ are mutually independent:
$$\text{Var}(S_n) = \sum_{t=1}^{n} \text{Var}(X_t) = 4 n d^2 p(1 - p)$$

$$\sigma(S_n) = \sqrt{\text{Var}(S_n)} = 2 d \sqrt{n p (1 - p)}$$

---

## 🏗️ System Architecture

```text
               +------------------------------------------+
               |        Interactive Streamlit UI          |
               | - Initial Price (S0) | Trials (M)        |
               | - Time Steps (N)     | Step Change (d)   |
               | - Probability of Increase (p)            |
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
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-------------------------------+             +-------------------------------+
| Visualizations (Matplotlib)   |             | Statistical Aggregation       |
| - M Paths Overlay (Alpha 0.03)|             | - Mean & Median Final Price   |
| - Empirical Mean Trajectory   |             | - Variance & Std Deviation    |
| - Terminal Price Histogram    |             | - Min & Max Realizations      |
+-------------------------------+             +-------------------------------+
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

## 🔬 Experimental Case Studies

- **Experiment 1: Symmetric Walk ($p = 0.5$)**: Theoretical expectation $E[S_n] = S_0$. Price distribution is symmetric and centered at $S_0$.
- **Experiment 2: Upward ($p = 0.8$) vs. Downward ($p = 0.2$) Bias**: Clear drift patterns matching theoretical $E[S_n] = S_0 + n \cdot d(2p-1)$.
- **Experiment 3: Law of Large Numbers ($M = 10 \to 5000$)**: Sample mean converges tightly to mathematical expectation as trial count $M$ increases.

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
