# Changelog

## [v2.0] – 2025-07-29

### Added
- Logistic Regression classifier to predict the 5-day forward direction of SPY
- Feature engineering for SPY using technical indicators:
  - Momentum (5-day, 20-day)
  - Volatility (20-day rolling std)
  - RSI (14-day)
  - SMA ratios (50-day, 100-day)
  - MACD and MACD histogram
  - Bollinger Band z-score
  - Cross-asset correlations with TLT and GLD
- Out-of-sample test set covering January–December 2024
- Hyperparameter tuning with `GridSearchCV` (penalty and regularization strength)
- Evaluation metrics including directional accuracy, F1 score, and classification report

### Changed
- Removed synthetic asset generation (BondProxy and GoldProxy)
- Replaced regression targets (5-day forward returns) with a binary classification target (up/down)
- Focused on single-asset (SPY) forecasting rather than portfolio optimization
- Switched model from `RandomForestRegressor` to `LogisticRegression`

---

## [v1.0] – 2025-07-29

### Added
- End-to-end pipeline for building a multi-asset, risk-parity portfolio using synthetic data
- `DataLoader`: reads SPY CSV data and creates synthetic proxies for bonds and gold
- `FeatureEngineer`: calculates returns, momentum, volatility, and RSI for each asset
- `ModelTrainer`: trains `RandomForestRegressor` models per asset to forecast 5-day forward returns
- `PortfolioOptimizer`: uses risk parity optimization via `scipy.optimize.minimize`
- Demonstration script: trains models, predicts returns, and computes portfolio weights

