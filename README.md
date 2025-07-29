# SPY 5-Day Forecasting with Random Forests – Version 2.0

This is Version 2.0 of a machine learning project that predicts the 5-day forward return and direction of the S&P 500 ETF (SPY) using technical and cross-asset indicators. This release adds improved modeling, cleaner design, and more consistent results using Random Forests for both regression and classification.

## Overview

The model generates:

- A numeric forecast of SPY's 5-day return (e.g., 0.012 = +1.2%)
- A directional classification of whether SPY will go up or down over the next 5 days

Price data is downloaded directly from Yahoo Finance and includes SPY, TLT, and GLD to inform technical features and correlations.

## What’s New in Version 2.0

- Switched from pure classification to dual modeling: regression + classification
- Cleaned and simplified code structure (human-style)
- Added cross-asset features (rolling correlations with TLT and GLD)
- Applied consistent train/test split using only out-of-sample 2024 data
- Removed synthetic or proxy data used in earlier versions

## Features Used

- Momentum (5-day, 20-day)
- Volatility (20-day)
- RSI (14-day)
- SMA ratios (vs. 50-day and 100-day averages)
- MACD and MACD histogram
- Bollinger Band z-score
- Rolling 20-day correlations with TLT and GLD

## Target Variables

- `target_return`: actual 5-day future return (regression)
- `target_direction`: 1 if the return is positive, 0 if not (classification)

## Model Design

- Regression model: `RandomForestRegressor`
- Classification model: `RandomForestClassifier`
- StandardScaler is used within the regression pipeline

## Evaluation

- Training window: 2010–2023
- Test window: 2024 only (true out-of-sample)
- Example output:
```
5-day return prediction RMSE: 0.0142
Directional accuracy: 62.38%
```

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pandas
numpy
yfinance
scikit-learn
```

## Project Structure

```
.
├── spy_forecast.py           # Main script
├── README.md                 # Documentation
├── requirements.txt          # Dependencies
```
