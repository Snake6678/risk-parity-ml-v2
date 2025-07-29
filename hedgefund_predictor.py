# hedgefund_predictor.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier


def download_data():
    tickers = ["SPY", "TLT", "GLD"]
    raw = yf.download(tickers, start="2010-01-01", end="2024-12-31", progress=False)
    return raw["Adj Close"] if "Adj Close" in raw else raw["Close"]


def engineer_features(prices):
    spy = prices["SPY"]
    ret = spy.pct_change()
    delta = spy.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    ema12 = spy.ewm(span=12, adjust=False).mean()
    ema26 = spy.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26

    feat = pd.DataFrame(index=spy.index)
    feat["ret"] = ret
    feat["mom5"] = spy.pct_change(5)
    feat["mom20"] = spy.pct_change(20)
    feat["vol20"] = ret.rolling(20).std()
    feat["rsi14"] = 100 - (100 / (1 + gain / loss))
    feat["sma50r"] = spy / spy.rolling(50).mean() - 1
    feat["sma100r"] = spy / spy.rolling(100).mean() - 1
    feat["macd"] = macd
    feat["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    feat["bollinger_z"] = (spy - spy.rolling(20).mean()) / (2 * spy.rolling(20).std())
    feat["corr_tlt"] = prices["TLT"].pct_change().rolling(20).corr(ret)
    feat["corr_gld"] = prices["GLD"].pct_change().rolling(20).corr(ret)

    fwd_ret = spy.pct_change(5).shift(-5)
    target = (fwd_ret > 0).astype(int)
    return pd.concat([feat, fwd_ret.rename("target_return"), target.rename("target_class")], axis=1).dropna()


def build_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42),
        "Logistic": LogisticRegression(max_iter=1000, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }


def train_and_evaluate(data):
    X = data.drop(columns=["target_return", "target_class"])
    y = data["target_class"]

    cutoff = "2023-12-31"
    train_mask = data.index <= cutoff
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42), threshold="median")
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    results = {}
    for name, model in build_models().items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        pipe.fit(X_train_sel, y_train)
        preds = pipe.predict(X_test_sel)
        results[name] = accuracy_score(y_test, preds)

    best_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42)
    cv = TimeSeriesSplit(n_splits=5)
    scores = [
        accuracy_score(y_train[val], best_model.fit(X_train_sel[tr], y_train.iloc[tr]).predict(X_train_sel[val]))
        for tr, val in cv.split(X_train_sel)
    ]

    best_model.fit(X_train_sel, y_train)
    final_preds = best_model.predict(X_test_sel)
    acc = accuracy_score(y_test, final_preds)
    conf = confusion_matrix(y_test, final_preds)

    feat_names = X.columns[selector.get_support()]
    imp = pd.DataFrame({"Feature": feat_names, "Importance": best_model.feature_importances_})
    imp = imp.sort_values(by="Importance", ascending=False)

    print("\nModel Accuracy Results:")
    for m, v in results.items():
        print(f"{m}: {v:.2%}")
    print(f"\nSelected Model CV Avg: {np.mean(scores):.2%}")
    print(f"Final Test Accuracy: {acc:.2%}\n")
    print("Confusion Matrix:")
    print(conf)
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds))

    plt.figure(figsize=(10, 6))
    plt.barh(imp["Feature"].head(10), imp["Importance"].head(10))
    plt.xlabel("Feature Importance")
    plt.title("Top Predictive Signals")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Save models in 'models' directory with ticker-specific names for compatibility
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/xgb_SPY.pkl")
    joblib.dump(selector, "models/selector_SPY.pkl")
    joblib.dump(scaler, "models/scaler_SPY.pkl")


def main():
    data = engineer_features(download_data())
    train_and_evaluate(data)


if __name__ == "__main__":
    main()

