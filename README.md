# Transformer & DRL for Stock Prediction and Trading
This repository contains Jupyter notebooks demonstrating Transformer-based and Deep Reinforcement Learning (DRL) approaches for stock market forecasting and portfolio management for our 10-423/623 Generative AI Course Project.

## Contents

- **`transformer.ipynb`**: Implementation of a TFT Transformer forecasting model using the Darts library and a custom Multiplex Attention Transformer forescasting model. Includes data preprocessing, model training, evaluation (MSE, MAE, MAPE, R²), and result visualization.
- **`drl_implement.ipynb`**: Deep Reinforcement Learning implementation for portfolio management. Covers environment setup, agent design (e.g., DQN, Policy Gradient), training loops, and performance analysis.

## Data Files & Notebooks

The repository includes both raw data folders and the notebooks that generate them. Run the listed notebooks in order to reproduce or update the CSVs.

- **`technical_data.ipynb`**  
  Pulls OHLCV price series for all DJIA-30 symbols and scrapes a suite of technical indicators (e.g., moving averages, RSI, Bollinger Bands). Outputs per-symbol CSVs under `initial_train_data/` (and the corresponding validation/test folders).

- **`sentiment_analysis_data.ipynb`**  
  Scrapes daily news headlines and computes three sentiment embeddings (FinBERT, VADER, Loughran–McDonald) for each stock. Writes the results to `initial_train_data/`, `initial_validation_data/`, and `initial_test_data/` according to your configured date ranges.

- **`macro_data.ipynb`**  
  Downloads key macroeconomic time series (federal funds rate, S&P 500 index, Dow Jones index, etc.) from public APIs. Outputs per-symbol CSVs under `initial_train_data/` (and the corresponding validation/test folders).

- **Raw data folders**  
  - `initial_train_data/`  
  - `initial_validation_data/`  
  - `initial_test_data/`  

- **Forecasts CSVs**    
  - `rolling_one_step_forecasts.csv` — the TFT Transformer's one‐step‐ahead predictions for all DJIA30 symbols.


- **Baseline folder**  
  - `baseline/` — baseline RL notebooks & data (`baseline_data (1).ipynb`, `readme.md`).


---


