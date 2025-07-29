import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------------
# 1. Download data (handles multi-index and yfinance adjustments)
# ---------------------------------------------------------------
tickers    = ["SPY", "TLT", "GLD"]
start_date = "2010-01-01"
end_date   = "2024-12-31"
raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    progress=False,
    auto_adjust=False
)

# extract Adjusted Close (fallback to Close if necessary)
if isinstance(raw.columns, pd.MultiIndex):
    try:
        prices = raw["Adj Close"]
    except KeyError:
        prices = raw["Close"]
else:
    prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

# Focus on the SPY series for most indicators
spy_price  = prices["SPY"]
spy_ret    = spy_price.pct_change()

# ---------------------------------------------------------------
# 2. Feature engineering: SPY indicators + a few cross‑asset vars
# ---------------------------------------------------------------
mom5  = spy_price / spy_price.shift(5) - 1
mom20 = spy_price / spy_price.shift(20) - 1
vol20 = spy_ret.rolling(20).std()

delta = spy_price.diff()
gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
rsi14 = 100 - (100/(1 + gain/loss))

sma50  = spy_price.rolling(50).mean()
sma100 = spy_price.rolling(100).mean()
sma50_ratio  = spy_price / sma50 - 1
sma100_ratio = spy_price / sma100 - 1

ema12  = spy_price.ewm(span=12, adjust=False).mean()
ema26  = spy_price.ewm(span=26, adjust=False).mean()
macd   = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
macd_hist = macd - signal

rolling_mean20 = spy_price.rolling(20).mean()
rolling_std20  = spy_price.rolling(20).std()
bollinger_z = (spy_price - rolling_mean20) / (rolling_std20 * 2)

corr_tlt = prices["TLT"].pct_change().rolling(20).corr(spy_ret)
corr_gld = prices["GLD"].pct_change().rolling(20).corr(spy_ret)

feature_df = pd.DataFrame({
    "ret_spy":        spy_ret,
    "mom5_spy":       mom5,
    "mom20_spy":      mom20,
    "vol20_spy":      vol20,
    "rsi14_spy":      rsi14,
    "sma50_ratio":    sma50_ratio,
    "sma100_ratio":   sma100_ratio,
    "macd":           macd,
    "macd_hist":      macd_hist,
    "bollinger_z":    bollinger_z,
    "corr_tlt_spy":   corr_tlt,
    "corr_gld_spy":   corr_gld
})

# ---------------------------------------------------------------
# 3. Target: direction of SPY’s return over the next 5 trading days
# ---------------------------------------------------------------
horizon = 5
future_ret = spy_price.pct_change(horizon).shift(-horizon)
target = (future_ret > 0).astype(int)

dataset = pd.concat([feature_df, target.rename("target")], axis=1).dropna()
X = dataset.drop("target", axis=1)
y = dataset["target"]

train_end = pd.Timestamp("2023-12-31")
mask      = dataset.index <= train_end
X_train, X_test = X.loc[mask], X.loc[~mask]
y_train, y_test = y.loc[mask], y.loc[~mask]

# ---------------------------------------------------------------
# 4. Model: Logistic Regression with hyper‑parameter tuning
# ---------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
])

param_grid = {
    "clf__C":      [0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2"]
}
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
probability = best_model.predict_proba(X_test)[:, 1]
predicted   = (probability > 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted)
f1       = f1_score(y_test, predicted)
print("Best parameters:", grid.best_params_)
print(f"Directional accuracy: {accuracy * 100:.2f}%")
print(f"F1 score: {f1 * 100:.2f}%")
print(classification_report(y_test, predicted, target_names=["Down", "Up"]))
