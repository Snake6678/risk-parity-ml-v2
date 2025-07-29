# train_models.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from datetime import datetime

# Configurable
TICKERS = ["SPY", "AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "AMZN", "META", "JPM", "XOM", "V"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def download_data(ticker):
    symbols = [ticker, "TLT", "GLD"]
    end_date = datetime.today().strftime("%Y-%m-%d")
    raw = yf.download(symbols, start="2010-01-01", end=end_date, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            return raw["Adj Close"].dropna()
        elif "Close" in raw.columns.levels[0]:
            return raw["Close"].dropna()
    return raw.dropna()


def engineer_features(prices, target):
    series = prices[target]
    ret = series.pct_change()
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26

    feat = pd.DataFrame(index=series.index)
    feat["ret"] = ret
    feat["mom5"] = series.pct_change(5)
    feat["mom20"] = series.pct_change(20)
    feat["vol20"] = ret.rolling(20).std()
    feat["rsi14"] = 100 - (100 / (1 + gain / loss))
    feat["sma50r"] = series / series.rolling(50).mean() - 1
    feat["sma100r"] = series / series.rolling(100).mean() - 1
    feat["macd"] = macd
    feat["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    feat["bollinger_z"] = (series - series.rolling(20).mean()) / (2 * series.rolling(20).std())
    feat["corr_tlt"] = prices["TLT"].pct_change().rolling(20).corr(ret)
    feat["corr_gld"] = prices["GLD"].pct_change().rolling(20).corr(ret)

    target_return = series.pct_change(5).shift(-5)
    target_class = (target_return > 0).astype(int)

    return pd.concat([feat, target_class.rename("target")], axis=1).dropna()


def train_model(ticker):
    prices = download_data(ticker)
    data = engineer_features(prices, ticker)
    X = data.drop(columns=["target"])
    y = data["target"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42), threshold="median")
    X_sel = selector.fit_transform(X_scaled, y)

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X_sel):
        model.fit(X_sel[train_idx], y.iloc[train_idx])
        preds = model.predict(X_sel[test_idx])
        acc = accuracy_score(y.iloc[test_idx], preds)
        scores.append(acc)

    model.fit(X_sel, y)
    joblib.dump(model, f"{MODEL_DIR}/rf_{ticker}.pkl")
    print(f"[{ticker}] Model trained. CV Accuracy: {np.mean(scores):.2%}")


if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            train_model(ticker)
        except Exception as e:
            print(f"[{ticker}] Failed: {e}")

