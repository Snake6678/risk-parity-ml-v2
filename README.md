# Risk-Parity Multi-Factor Portfolio with Machine Learning – Version 2.0

This repository presents an updated version of a machine learning-based investment strategy focused on predicting the directional movement of the S&P 500 ETF (SPY). Version 2.0 introduces improved predictive accuracy, a broader feature set, and a more rigorous evaluation framework using recent market data.

## Overview of Improvements in Version 2.0

- Incorporates updated data through December 2024
- Expanded feature set, including both technical indicators and cross-asset correlations
- Chronological train/test split to ensure out-of-sample integrity
- Enhanced model tuning using `GridSearchCV` with logistic regression and both L1 and L2 penalties
- Full classification evaluation with precision, recall, and F1-score metrics

## Objective

The model aims to predict the **direction (up or down)** of SPY's return over a five-day horizon using a set of engineered financial indicators.

## Feature Set

The model includes the following features:

| Feature           | Description                                          |
|-------------------|------------------------------------------------------|
| `ret_spy`         | Daily return of SPY                                  |
| `mom5_spy`        | 5-day momentum                                       |
| `mom20_spy`       | 20-day momentum                                      |
| `vol20_spy`       | 20-day rolling volatility                            |
| `rsi14_spy`       | 14-day Relative Strength Index                       |
| `sma50_ratio`     | Percentage deviation from 50-day simple moving avg   |
| `sma100_ratio`    | Percentage deviation from 100-day simple moving avg  |
| `macd`            | MACD line (EMA12 - EMA26)                            |
| `macd_hist`       | MACD histogram (MACD - signal line)                  |
| `bollinger_z`     | Z-score of distance from 20-day mean                 |
| `corr_tlt_spy`    | 20-day rolling correlation between SPY and TLT       |
| `corr_gld_spy`    | 20-day rolling correlation between SPY and GLD       |

## Methodology

- **Model**: Logistic Regression
- **Pipeline**: StandardScaler + Logistic Regression with hyperparameter tuning
- **Hyperparameters**: Regularization strength (`C`) and penalty (`l1`, `l2`)
- **Tuning**: Performed using 3-fold cross-validation and F1-score as the optimization metric
- **Target**: Binary indicator of whether the SPY return over the next 5 trading days is positive

### Data Splitting

- **Training set**: January 2010 to December 2023
- **Testing set**: January 2024 onward
- This approach avoids look-ahead bias by using strictly historical data for training.

## Evaluation Results (Sample Output)

```
Best parameters: {'clf__C': 1, 'clf__penalty': 'l2'}
Directional accuracy: 62.34%
F1 score: 65.29%
              precision    recall  f1-score   support

        Down       0.60      0.63      0.61        54
          Up       0.65      0.63      0.64        64

    accuracy                           0.62       118
   macro avg       0.62      0.62      0.62       118
weighted avg       0.63      0.62      0.62       118
```

## Installation

Required packages:

- `pandas`
- `numpy`
- `yfinance`
- `scikit-learn`

Install dependencies using:

```bash
pip install -r requirements.txt
```

## File Structure

```
.
├── risk_parity_ml_v2.py       # Main script
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
└── LICENSE                    # (Optional) project license
```

## Previous Version

For the original implementation, refer to [Version 1.0](https://github.com/Snake6678/Risk-Parity-Multi-Factor-Portfolio-with-Machine-Learning).

## License

This project is licensed under the MIT License.

