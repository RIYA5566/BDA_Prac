# 📈 One-Dimensional Stock Price Random Walk Simulation
### *Mathematical Foundation of Big Data Analytics (BDA) — Mini-Project*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Mathematical Formulation](#-mathematical-formulation)
  - [1. Discrete Stochastic Walk](#1-discrete-stochastic-walk)
  - [2. Expected Value \& Drift Analysis](#2-expected-value--drift-analysis)
  - [3. Variance \& Standard Deviation](#3-variance--standard-deviation)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation \& Quickstart](#-installation--quickstart)
- [Experimental Case Studies](#-experimental-case-studies)
- [Viva \& Interview Q\&A](#-viva--interview-qa)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview

This project implements a computational and mathematical simulation of stock price dynamics using a **one-dimensional discrete random walk model**. Powered by **Streamlit**, **NumPy**, and **Matplotlib**, the application performs high-throughput Monte Carlo simulations across thousands of stochastic trajectories to analyze price convergence, drift dynamics, and probability distributions.

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

## 📂 Repository Structure

```bash
BDA/
├── README.md                      # Main project documentation & guide
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
- **Configuration**: Compare $M = 10$ trials vs. $M = 2000$ trials.
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

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
