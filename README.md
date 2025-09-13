# Adaptive RL for Regime-Aware Trading (VAE + PPO)
Hybrid β-VAE + PPO for Tactical Asset Allocation
This repository contains the implementation of a sophisticated quantitative trading strategy that uses a hybrid machine learning model to perform tactical asset allocation. The core of the strategy is a combination of a β-Variational Autoencoder (β-VAE) for market regime detection and a Proximal Policy Optimization (PPO) agent for making trading decisions.

The model is designed to be causal and avoid any look-ahead bias, ensuring that all decisions are made using only information available at that time step (t → t+1). It is rigorously backtested on S&P 500 data and evaluated for robustness on out-of-distribution assets like cryptocurrencies and major ETFs.

Key Features
Causal Design: Strict t→t+1 alignment prevents look-ahead bias.

Hybrid Model: Combines a β-VAE for unsupervised feature learning with a PPO agent for reinforcement learning-based decision making.

Regime Detection: A Gaussian Mixture Model (GMM) is trained on the VAE's latent space to identify different market regimes (e.g., "calm" vs. "turbulent").

Multiple Strategies: Implements and compares several allocation strategies:

Baseline: A simple equal-weighted S&P 500 portfolio.

Hard Allocation: A rule-based strategy using EWMA-smoothed regime probabilities with hysteresis and a dwell time to reduce turnover.

Soft Allocation: A dynamic allocation based on regime-conditioned return forecasts.

PPO Agent: An adaptive RL agent trained to maximize risk-adjusted returns.

Robust Evaluation: The trained PPO agent is tested on both the S&P 500 test set and out-of-distribution (OOD) assets (BTC, ETH, SPY, QQQ) to assess its generalization capabilities.

Calendar Split: Data is split chronologically (Train < 2022, Validation = 2022, Test >= 2023) to simulate a real-world deployment scenario.

Methodology
The project follows a multi-stage pipeline:

Feature Engineering: Raw price and volume data for ~470 long-history S&P 500 stocks are used to engineer features like log returns, momentum, volatility, and volume spikes.

Dimensionality Reduction (β-VAE): The high-dimensional feature space is compressed into a low-dimensional latent space (LATENT_DIM=16) using a β-VAE. This captures the essential market dynamics.

Regime Modeling (GMM): A Gaussian Mixture Model identifies distinct market regimes from the latent space embeddings. The "calm" regime is selected as the one with the highest Sharpe ratio during training.

Strategy Execution & RL Training (PPO): The baseline, hard, and soft allocation models are executed on the test set. Concurrently, a PPO agent is trained on the VAE features to learn an optimal trading policy. The agent's goal is to maximize returns while accounting for transaction costs.

Getting Started
Prerequisites

Python 3.8+

A Google Drive account with the S&P 500 dataset (or a local copy).

Installation

Clone the repository:

git clone <your-repository-url>
cd <your-repository-directory>


Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required packages:
```
pip install -r requirements.txt

```
Prepare the data:

Download the S&P 500 dataset.

Update the DATA_CSV_PATH variable in main.py to point to the location of your S&P500_all_companies.csv file.

Usage

Run the main training and evaluation script:
This will perform feature engineering, train the VAE and PPO models, run all strategies, and save the final results table and plotting artifacts.
```
python main.py
```

Generate the plots:
After main.py has finished and created the run_artifacts.joblib file, you can generate all the analysis plots.
```
python plotter.py

```
The plots will be displayed on screen and saved in the images/ directory.

Performance & Results
Final Performance Metrics

The table below summarizes the performance of each strategy on the test set (2023 onwards). The PPO agent demonstrates strong performance, especially on out-of-distribution crypto assets, showcasing its adaptability.

| Strategy | CAGR | Volatility | Sharpe | Sortino | Calmar | Max Drawdown | Final Value |
| Baseline (Equal-Weighted) | 0.1518 | 0.1278 | 1.1882 | 1.8830 | 1.2562 | -0.1208 | 1.368234e+06 |
| Hard Allocation (Gross) | 0.0824 | 0.0969 | 0.8503 | 0.9476 | 0.6260 | -0.1316 | 1.193076e+06 |
| Hard Allocation (Net) | 0.0672 | 0.0969 | 0.6931 | 0.7759 | 0.4733 | -0.1420 | 1.156074e+06 |
| Soft Allocation (Gross) | 0.1676 | 0.1363 | 1.2298 | 1.9344 | 1.3057 | -0.1284 | 1.412842e+06 |
| Soft Allocation (Net) | 0.1676 | 0.1363 | 1.2298 | 1.9344 | 1.3057 | -0.1284 | 1.412842e+06 |
| Adaptive RL Agent (PPO, Net) - S&P | 0.1361 | 0.1240 | 1.0970 | 1.7507 | 1.0844 | -0.1255 | 1.345596e+06 |
| Adaptive RL Agent (PPO, Net) - Crypto (BTC-USD) | 0.7744 | 0.3898 | 1.9865 | 3.1060 | 3.8609 | -0.2006 | 3.601047e+06 |
| Adaptive RL Agent (PPO, Net) - Crypto (ETH-USD) | 0.5868 | 0.4383 | 1.3388 | 2.0980 | 2.0402 | -0.2876 | 2.827989e+06 |
| Adaptive RL Agent (PPO, Net) - Options (SPY) | 0.1945 | 0.1278 | 1.5222 | 2.2325 | 1.8969 | -0.1025 | 1.495621e+06 |
| Adaptive RL Agent (PPO, Net) - Options (QQQ) | 0.2910 | 0.1799 | 1.6175 | 2.3659 | 2.3526 | -0.1237 | 1.773266e+06 |

Visualizations

S&P 500 Test Performance

Equity curves for the different strategies on the S&P 500 test set.

Drawdown profiles for the S&P 500 strategies.

63-day rolling Sharpe ratios show the dynamic risk-adjusted performance over time.

PPO Agent Out-of-Distribution Performance

The PPO agent generalizes remarkably well to unseen assets like SPY and QQQ.

Drawdowns remain managed even on volatile, out-of-distribution assets.

The agent achieves the highest CAGR on cryptocurrencies, highlighting its ability to adapt to different market dynamics.

Sharpe ratios are strong across all tested assets, indicating efficient risk-taking.

Strategy Diagnostics

This plot shows the smoothed "calm" regime probability and the resulting binary position taken by the Hard Allocation strategy.

The distribution of the smoothed calm probability, showing how confident the model is in its regime classification.

Disclaimer
This project is for educational and research purposes only. The models and strategies presented here are not financial advice. Trading in financial markets involves substantial risk, and past performance is not indicative of future results.

