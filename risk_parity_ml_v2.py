import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

tickers = ["SPY", "TLT", "GLD"]
start_date = "2010-01-01"
end_date = "2024-12-31"

raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

if isinstance(raw_data.columns, pd.MultiIndex):
    prices = raw_data.get("Adj Close", raw_data.get("Close"))
else:
    prices = raw_data["Adj Close"] if "Adj Close" in raw_data else raw_data["Close"]

spy_price = prices["SPY"]
spy_return = spy_price.pct_change()

momentum_5 = spy_price.pct_change(5)
momentum_20 = spy_price.pct_change(20)
volatility_20 = spy_return.rolling(20).std()

delta = spy_price.diff()
gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
rsi_14 = 100 - (100 / (1 + gain / loss))

sma_50 = spy_price.rolling(50).mean()
sma_100 = spy_price.rolling(100).mean()
sma_50_ratio = spy_price / sma_50 - 1
sma_100_ratio = spy_price / sma_100 - 1

ema_12 = spy_price.ewm(span=12, adjust=False).mean()
ema_26 = spy_price.ewm(span=26, adjust=False).mean()
macd_line = ema_12 - ema_26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_hist = macd_line - signal_line

rolling_mean = spy_price.rolling(20).mean()
rolling_std = spy_price.rolling(20).std()
bollinger_z = (spy_price - rolling_mean) / (2 * rolling_std)

corr_with_tlt = prices["TLT"].pct_change().rolling(20).corr(spy_return)
corr_with_gld = prices["GLD"].pct_change().rolling(20).corr(spy_return)

features = pd.DataFrame({
    "spy_return": spy_return,
    "momentum_5": momentum_5,
    "momentum_20": momentum_20,
    "volatility_20": volatility_20,
    "rsi_14": rsi_14,
    "sma_50_ratio": sma_50_ratio,
    "sma_100_ratio": sma_100_ratio,
    "macd_line": macd_line,
    "macd_hist": macd_hist,
    "bollinger_z": bollinger_z,
    "corr_with_tlt": corr_with_tlt,
    "corr_with_gld": corr_with_gld
})

forecast_horizon = 5
future_return = spy_price.pct_change(forecast_horizon).shift(-forecast_horizon)
future_direction = (future_return > 0).astype(int)

data = pd.concat([
    features,
    future_return.rename("target_return"),
    future_direction.rename("target_direction")
], axis=1).dropna()

X = data[features.columns]
y_reg = data["target_return"]
y_cls = data["target_direction"]

train_cutoff = pd.Timestamp("2023-12-31")
train_mask = data.index <= train_cutoff

X_train = X.loc[train_mask]
X_test = X.loc[~train_mask]
y_train_reg = y_reg.loc[train_mask]
y_test_reg = y_reg.loc[~train_mask]
y_train_cls = y_cls.loc[train_mask]
y_test_cls = y_cls.loc[~train_mask]

regression_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=42
    ))
])

regression_model.fit(X_train, y_train_reg)
reg_predictions = regression_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_reg, reg_predictions))
print(f"5-day return prediction RMSE: {rmse:.4f}")

classification_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=10,
    random_state=42
)

classification_model.fit(X_train, y_train_cls)
cls_predictions = classification_model.predict(X_test)
accuracy = accuracy_score(y_test_cls, cls_predictions)
print(f"Directional accuracy: {accuracy * 100:.2f}%")
