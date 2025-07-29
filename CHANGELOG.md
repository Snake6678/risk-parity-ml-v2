# Changelog

## [v2.0] – 2025-07-29

### Changed
- Replaced synthetic multi-asset system with a single-asset model focused only on SPY
- Removed custom class structure (`DataLoader`, `FeatureEngineer`, `ModelTrainer`, `PortfolioOptimizer`)
- Eliminated synthetic proxies for BondProxy and GoldProxy
- Replaced portfolio optimization with direct return and direction forecasting
- Simplified the code into a single, flat, human-readable script (`spy_forecast.py`)

### Added
- Daily feature engineering using SPY technical indicators:
  - Momentum, volatility, RSI, SMA ratios, MACD, Bollinger Bands
  - Cross-asset correlations with TLT and GLD
- Dual model structure:
  - RandomForestRegressor for 5-day forward return prediction
  - RandomForestClassifier for 5-day directional (up/down) classification
- Chronological train/test split (train through 2023, test on 2024 only)
- Evaluation metrics: RMSE and directional accuracy

### Removed
- Risk parity optimization logic
- Multi-asset regressors
- Need for local CSV file (uses Yahoo Finance API instead)

---

## [v1.0] – 2025-07-29

### Added
- Modular pipeline for multi-asset prediction and portfolio optimization
- `DataLoader`: loads SPY CSV and generates synthetic returns for bond/gold proxies
- `FeatureEngineer`: computes basic indicators (momentum, volatility, RSI)
- `ModelTrainer`: trains one `RandomForestRegressor` per asset
- `PortfolioOptimizer`: solves for risk parity weights using `scipy.optimize.minimize`
- Fully self-contained demonstration using only historical data and no live API calls
