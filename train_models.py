"""
train_models.py
-----------------

This script trains machine‑learning models for a set of stock tickers to
predict the direction of the next move.  It fetches historical price data
from Yahoo! Finance, engineers a suite of technical indicators for each
instrument, evaluates several candidate XGBoost hyperparameter settings
using a time‑series cross‑validation scheme and finally persists the best
trained model along with its preprocessing steps (feature scaler and
selector) to disk.  Those artifacts can later be loaded by the
``realtime_api.py`` service to generate real‑time forecasts.

The feature set draws on common technical analysis tools.  Indicators such
as the Relative Strength Index (RSI), Bollinger Bands and the Moving
Average Convergence Divergence (MACD) are highlighted as some of the most
popular tools for trend and momentum analysis【320195700379674†L59-L65】.
We also include simple return‑based statistics (momentum and volatility)
and rolling correlations with bonds (TLT) and gold (GLD) to capture
shifting risk regimes.  Gold returns in particular are historically
weakly correlated with both equity and bond indices, making gold a
useful diversifier【549609825918978†L54-L94】; a changing correlation
between a stock and gold or bonds can therefore signal a transition from
risk‑on to risk‑off environments.

The script produces, for each ticker, three files inside a ``models/``
folder:

* ``xgb_<ticker>.pkl`` – the trained XGBoost classifier (with tuned
  hyperparameters)
* ``selector_<ticker>.pkl`` – a feature selector fitted on the training set
* ``scaler_<ticker>.pkl`` – a standard scaler used to normalize the raw
  features

These files can be consumed by downstream applications to compute
out‑of‑sample predictions without recomputing the entire training pipeline.

NOTE: This script requires the ``yfinance`` package to be installed.
Install it with ``pip install yfinance`` if it is not already available.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

# List of tickers to train models for.  You can modify this list to include
# additional equities.  The realtime API will look for models named
# xgb_<ticker>.pkl in the ``models`` folder.
TICKERS: List[str] = [
    "SPY", "AAPL", "MSFT", "TSLA", "GOOGL", "NVDA",
    "AMZN", "META", "JPM", "XOM", "V"
]

# Directory where the trained models and preprocessing objects will be saved.
MODELS_DIR: str = "models"

# Number of trading days ahead to use when defining the target.  Setting
# this value to one day means we aim to predict the next day's direction.
FORECAST_HORIZON: int = 1

# Hyperparameter grid for XGBoost.  The training routine will evaluate
# each combination using cross‑validation and select the one with the highest
# average accuracy.
XGB_PARAM_GRID: List[Dict[str, int | float]] = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03},
]

# Additional hyperparameter lists for alternative algorithms.  Random
# forests are evaluated with different numbers of trees; logistic
# regression uses a single configuration but is included for comparison.
RF_PARAM_LIST: List[Dict[str, int]] = [
    {"n_estimators": 200},
    {"n_estimators": 300},
]
LR_PARAM_LIST: List[Dict[str, int]] = [
    {}
]


def download_prices(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted or raw closing prices for a list of symbols.

    Parameters
    ----------
    symbols : List[str]
        The tickers to download.  Must include at least the primary equity and
        two reference assets used for correlation features (e.g. TLT and GLD).
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by trading date containing the closing prices
        of the requested symbols.  Columns are the ticker symbols.
    """
    # Fetch price history without auto adjustments.  yfinance will return a
    # DataFrame with columns such as 'Open', 'High', 'Low', 'Close',
    # 'Adj Close', 'Volume' when a single ticker is passed, and a multi‑index
    # DataFrame when multiple tickers are requested.
    # Explicitly set auto_adjust to False to avoid future warnings about
    # changing defaults.  When auto_adjust=True, yfinance returns a single
    # adjusted price series and omits the raw closing price, which can break
    # downstream selection logic.
    raw = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False)

    def extract_price(df: pd.DataFrame, field: str) -> pd.DataFrame:
        """Extract a price field ('Adj Close' or 'Close') from the yfinance output.

        When multiple tickers are requested the returned DataFrame has a
        multi‑index on its columns.  Pandas allows selecting a top‑level field
        directly with ``df[field]`` which returns a DataFrame of tickers as
        columns.  When a single ticker is requested, ``df[field]`` returns a
        Series; we convert it to a DataFrame with the ticker name as the
        column label.
        """
        try:
            extracted = df[field]
        except KeyError:
            raise
        # If a Series is returned (single ticker), wrap it into a DataFrame
        if isinstance(extracted, pd.Series):
            extracted = extracted.to_frame(name=symbols[0])
        # Flatten multi‑index columns to a single level if present
        if isinstance(extracted.columns, pd.MultiIndex):
            extracted.columns = extracted.columns.get_level_values(0)
        return extracted

    try:
        prices = extract_price(raw, "Adj Close")
    except KeyError:
        try:
            prices = extract_price(raw, "Close")
        except KeyError:
            raise ValueError(
                "Unable to locate 'Adj Close' or 'Close' prices in downloaded data."
            )
    return prices.dropna()


def engineer_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create a feature matrix and classification target for a given ticker.

    This function computes a suite of technical indicators for the specified
    equity and merges them into a single DataFrame.  It also constructs the
    classification target based on the sign of the forward return over
    ``FORECAST_HORIZON`` days.  The realtime API expects the returned
    DataFrame to contain a column named ``target`` alongside the engineered
    features.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices for the selected ticker and its reference instruments
        (must include columns for ``ticker``, 'TLT' and 'GLD').
    ticker : str
        The primary equity whose direction we want to predict.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date where each row contains feature values and
        the corresponding binary target (1 for an up move, 0 for a down move).
    """
    # Ensure required columns are present
    required_cols = {ticker, "TLT", "GLD"}
    missing = required_cols.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in price data")

    # Isolate the price series for the primary ticker and compute simple returns.
    # Use ``squeeze()`` to ensure we get a 1‑D Series even if the underlying
    # DataFrame returns a 1‑column frame (which can happen if a MultiIndex is
    # present).  Without squeezing, operations like rolling correlation can
    # inadvertently create two‑dimensional arrays, leading to shape errors.
    price = prices[ticker].squeeze()
    ret = price.pct_change()

    # Price differences to compute RSI components
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()

    # Exponential moving averages for MACD
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    # Rolling statistics
    mom5 = price.pct_change(5)
    mom20 = price.pct_change(20)
    vol20 = ret.rolling(20).std()
    rsi14 = 100 - (100 / (1 + gain / loss))
    sma50r = price / price.rolling(50).mean() - 1
    sma100r = price / price.rolling(100).mean() - 1
    macd_hist = macd_line - macd_signal
    rolling_mean = price.rolling(20).mean()
    rolling_std = price.rolling(20).std()
    bollinger_z = (price - rolling_mean) / (2 * rolling_std)

    # Correlation of the stock's returns with bond and gold returns over a 20‑day window
    # Correlations require 1‑D Series as the second argument.  Use squeeze to
    # prevent pandas from returning a DataFrame when selecting a single column.
    ret_tlt = prices["TLT"].pct_change().squeeze()
    ret_gld = prices["GLD"].pct_change().squeeze()
    corr_tlt = ret.rolling(20).corr(ret_tlt)
    corr_gld = ret.rolling(20).corr(ret_gld)

    # ---------------------------------------------------------------------------
    # Optional volume and range‑based indicators
    #
    # Some technical indicators require high, low and volume series.  We
    # retrieve this additional information separately via yfinance.  If the
    # request fails or the columns are absent, the corresponding features will
    # be populated with NaNs and subsequently dropped when we call dropna().
    #
    # On‑Balance Volume (OBV) accumulates volume when the price closes higher
    # and subtracts volume when it closes lower.  The Average True Range (ATR)
    # measures recent trading range volatility.  Including these features can
    # help the model capture volume and range dynamics.
    try:
        # Determine date range from the price data to avoid unnecessary
        # downloading.  Use the existing index if possible.
        start_date = prices.index.min().strftime("%Y-%m-%d")
        end_date = prices.index.max().strftime("%Y-%m-%d")
        extra = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        # Extract high, low, close and volume as 1‑D Series.  If the returned
        # objects are DataFrames (e.g. due to multi‑index columns), we call
        # ``squeeze()`` to convert them to Series to avoid accidental
        # broadcasting later.
        high = extra.get("High")
        if high is not None and isinstance(high, pd.DataFrame):
            high = high.squeeze()
        low = extra.get("Low")
        if low is not None and isinstance(low, pd.DataFrame):
            low = low.squeeze()
        close_extra = extra.get("Close")
        if close_extra is not None and isinstance(close_extra, pd.DataFrame):
            close_extra = close_extra.squeeze()
        volume = extra.get("Volume")
        if volume is not None and isinstance(volume, pd.DataFrame):
            volume = volume.squeeze()

        # True Range components for ATR.  We build each component as a
        # Series and collect them in a list.  Later we take the row‑wise
        # maximum across these components.  Using Series ensures that
        # ``pd.concat`` does not create a 2‑D array inadvertently.
        tr_components: List[pd.Series] = []
        if high is not None and low is not None:
            tr_components.append((high - low))
        if high is not None and close_extra is not None:
            tr_components.append((high - close_extra.shift(1)).abs())
        if low is not None and close_extra is not None:
            tr_components.append((low - close_extra.shift(1)).abs())
        if tr_components:
            tr_df = pd.concat(tr_components, axis=1)
            tr_series = tr_df.max(axis=1)
            atr14 = tr_series.rolling(14).mean()
        else:
            atr14 = pd.Series(index=prices.index, dtype=float)
        # On‑Balance Volume
        if volume is not None:
            # Align volume index with price index
            vol_aligned = volume.reindex(prices.index).fillna(0)
            # Compute directional sign of price change as a Series.  Using
            # ``pd.Series`` ensures the multiplication below yields a Series
            # rather than a higher‑dimensional object.
            sign_ret = pd.Series(np.sign(price.diff().fillna(0)), index=prices.index)
            obv = (vol_aligned * sign_ret).cumsum()
        else:
            obv = pd.Series(index=prices.index, dtype=float)
    except Exception:
        # If download fails, fall back to NaNs
        atr14 = pd.Series(index=prices.index, dtype=float)
        obv = pd.Series(index=prices.index, dtype=float)

    # Assemble the feature DataFrame
    features = pd.DataFrame({
        "ret": ret,
        "mom5": mom5,
        "mom20": mom20,
        "vol20": vol20,
        "rsi14": rsi14,
        "sma50r": sma50r,
        "sma100r": sma100r,
        "macd_line": macd_line,
        "macd_hist": macd_hist,
        "bollinger_z": bollinger_z,
        "corr_tlt": corr_tlt,
        "corr_gld": corr_gld,
        "atr14": atr14,
        "obv": obv
    })

    # Forward return and binary target
    fwd_ret = price.pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)
    target = (fwd_ret > 0).astype(int)

    # Combine features with target and drop rows containing NaNs
    data = pd.concat([features, target.rename("target")], axis=1).dropna()
    return data


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    scaler: StandardScaler,
    selector: SelectFromModel,
    algorithm: str,
    params: Dict[str, int | float]
) -> float:
    """Compute cross‑validated accuracy for a given algorithm and parameter set.

    This helper supports XGBoost, RandomForest and LogisticRegression.  It
    performs a time‑series split, fits the scaler and selector on each
    training fold, trains the specified model with the provided
    hyperparameters and returns the average accuracy across all folds.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    scaler : StandardScaler
        A StandardScaler instance used to scale features.
    selector : SelectFromModel
        A feature selector used to reduce dimensionality.
    algorithm : str
        Which algorithm to use ('xgb', 'rf' or 'lr').
    params : Dict[str, int | float]
        Hyperparameters to configure the model.

    Returns
    -------
    float
        Mean accuracy across all cross‑validation folds.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    scores: List[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        selector.fit(X_train_scaled, y_train)
        X_train_sel = selector.transform(X_train_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        if algorithm == "xgb":
            model = XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        elif algorithm == "rf":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                random_state=42
            )
        elif algorithm == "lr":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        model.fit(X_train_sel, y_train)
        preds = model.predict(X_test_sel)
        scores.append(accuracy_score(y_test, preds))
    return float(np.mean(scores)) if scores else float("nan")


def train_single_ticker(ticker: str, end_date: str) -> None:
    """Train and save the best model for a single ticker.

    This helper downloads the required price history, engineers features,
    performs hyperparameter tuning over a small grid of XGBoost settings using
    time‑series cross‑validation, and finally saves the best model along with
    its scaler and feature selector.  If there are insufficient data points
    to fit the model, the function logs a warning instead of raising an
    exception.

    Parameters
    ----------
    ticker : str
        The ticker symbol for which to train the model.
    end_date : str
        Final date for the training data in 'YYYY-MM-DD' format.
    """
    print(f"\nTraining model for {ticker} up to {end_date}...")

    # Download price data for the target ticker and reference assets
    try:
        prices = download_prices([ticker, "TLT", "GLD"], start="2010-01-01", end=end_date)
    except Exception as e:
        print(f"  Failed to download data for {ticker}: {e}")
        return

    # Engineer features and target
    try:
        data = engineer_features(prices, ticker)
    except Exception as e:
        print(f"  Feature engineering failed for {ticker}: {e}")
        return

    X = data.drop(columns=["target"])
    y = data["target"]

    if len(data) < 100:
        print(f"  Not enough data ({len(data)} rows) to train {ticker}; skipping.")
        return

    # Initialize scaler and feature selector
    scaler = StandardScaler()
    selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42), threshold="median")

    # Hyperparameter tuning across multiple algorithms
    best_overall_score = -1.0
    best_model_spec: Tuple[str, Dict[str, int | float]] | None = None

    # Evaluate XGBoost parameter grid
    for params in XGB_PARAM_GRID:
        score = evaluate_model(X, y, scaler, selector, "xgb", params)
        print(f"  Evaluated XGB params {params}: CV accuracy {score:.4f}")
        if score > best_overall_score:
            best_overall_score = score
            best_model_spec = ("xgb", params.copy())

    # Evaluate RandomForest parameter list
    for params in RF_PARAM_LIST:
        score = evaluate_model(X, y, scaler, selector, "rf", params)
        print(f"  Evaluated RF params {params}: CV accuracy {score:.4f}")
        if score > best_overall_score:
            best_overall_score = score
            best_model_spec = ("rf", params.copy())

    # Evaluate LogisticRegression (no hyperparameters)
    for params in LR_PARAM_LIST:
        score = evaluate_model(X, y, scaler, selector, "lr", params)
        print(f"  Evaluated LR: CV accuracy {score:.4f}")
        if score > best_overall_score:
            best_overall_score = score
            best_model_spec = ("lr", params.copy())

    if best_model_spec is None:
        print(f"  No valid model configuration found for {ticker}.")
        return
    best_alg, best_params = best_model_spec
    print(f"  Best model for {ticker}: {best_alg} with params {best_params} and CV accuracy {best_overall_score:.4f}")

    # Fit the scaler and selector on the full dataset
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    selector.fit(X_scaled, y)
    X_selected = selector.transform(X_scaled)

    # Instantiate and train the final model
    if best_alg == "xgb":
        final_model = XGBClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    elif best_alg == "rf":
        final_model = RandomForestClassifier(
            n_estimators=best_params.get("n_estimators", 200),
            random_state=42
        )
    else:
        from sklearn.linear_model import LogisticRegression
        final_model = LogisticRegression(max_iter=1000, random_state=42)

    final_model.fit(X_selected, y)

    # Persist model, scaler and selector.  For compatibility with the realtime
    # API we always use the ``xgb_<ticker>.pkl`` naming convention, even if
    # the underlying model is not an XGBoost classifier.  The API loads the
    # model and calls ``predict`` and ``predict_proba`` methods which are
    # available on RandomForest and LogisticRegression as well.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"xgb_{ticker}.pkl")
    selector_path = os.path.join(MODELS_DIR, f"selector_{ticker}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{ticker}.pkl")
    joblib.dump(final_model, model_path)
    joblib.dump(selector, selector_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved model to {model_path}\n")


def train_models(symbols: List[str], end_date: str) -> None:
    """Train models for all specified symbols up to a given date.

    Iterates over the provided tickers and invokes ``train_single_ticker`` for
    each.  The ``end_date`` parameter allows the caller to retrain models
    through any desired cutoff.  Models for which training data could not be
    obtained or were insufficient will be skipped gracefully.

    Parameters
    ----------
    symbols : List[str]
        Ticker symbols for which to build models.
    end_date : str
        Last date for the training period, in 'YYYY-MM-DD' format.
    """
    for sym in symbols:
        try:
            train_single_ticker(sym, end_date)
        except Exception as ex:
            print(f"Exception encountered while training {sym}: {ex}")


if __name__ == "__main__":
    # Determine today's date in ISO format.  The user runs this script
    # interactively, so we respect the system clock.  The realtime API will
    # similarly call the training routine when invoked.
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"Starting training pipeline as of {today}...")
    train_models(TICKERS, today)
