# HedgeFund AI

A machine learning pipeline for predicting short-term equity movements using macroeconomic indicators, ETFs, and technical data. Built for research and potential deployment in quantitative hedge fund environments.

## Project Overview

This repository contains a modular training pipeline for building and evaluating ML-based models that predict binary price movements (up/down) for major equities. It supports:

- Automated data downloading via `yfinance`
- Feature engineering from price returns, momentum, volatility, and macro instruments (GLD, TLT)
- Model training with `XGBoost`, `LogisticRegression`, `RandomForest`, etc.
- Cross-validation performance reporting
- Scalable training over multiple tickers

## Key Features

- Data Sources: Historical adjusted prices from Yahoo Finance for equities and macro ETFs
- Models: Supports ensemble methods and interpretable classifiers
- CV Accuracy Reporting: Tracks model performance via cross-validation
- Modular Design: Easy to plug in new tickers, models, or features
- Command-line Ready: Train all models with a single command

## Tech Stack

| Tool              | Purpose                         |
|-------------------|----------------------------------|
| Python            | Core programming language        |
| yfinance          | Financial data acquisition       |
| pandas / numpy    | Data manipulation                |
| scikit-learn      | Model training and evaluation    |
| xgboost           | Gradient boosting classifier     |
| joblib            | Model persistence                |

## Example Output

```
Starting upgraded training pipeline: 2025-07-29
Training model for SPY...
Trained Logistic model with CV accuracy: 0.5456
Training model for AAPL...
Trained Logistic model with CV accuracy: 0.5266
...
```

## Repository Structure

```
hedgefund_ai/
├── models/                   # Saved models per ticker
├── train_models.py          # Main training script
├── feature_engineering.py   # Custom feature generation logic
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
└── README.md                # You're reading it
```

## How to Run

```bash
# Clone the repo
git clone https://github.com/Snake6678/risk-parity-ml-v2.git
cd hedgefund_ai

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train_models.py
```

## Current Model Status

- Models trained on: SPY, AAPL, MSFT, TSLA, GOOGL, NVDA, AMZN, META, JPM, XOM, V
- Target variable: 1-day forward binary return (up/down)
- Average CV accuracy range: 0.51 – 0.56

## Roadmap

- [x] Add CV reporting and logging
- [x] Add fallback model selection
- [ ] Integrate feature importance analysis
- [ ] Add backtesting of predictions
- [ ] Deploy prediction API (FastAPI)

## Author

**Bryan Wierdak**  
