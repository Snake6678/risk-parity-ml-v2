# realtime_api.py

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from train_models import engineer_features
from sklearn.preprocessing import StandardScaler

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
    <head>
        <title>HedgeFund AI</title>
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
        <style>
            body { font-family: sans-serif; background: #111; color: #eee; padding: 2rem; }
            h1 { color: #00ffcc; }
            input, button { padding: 0.5rem; font-size: 1rem; }
        </style>
    </head>
    <body>
        <h1>📈 HedgeFund AI Predictor</h1>
        <form onsubmit=\"event.preventDefault(); fetchPrediction();\">
            <input id=\"ticker\" placeholder=\"Enter Ticker (e.g. AAPL)\" />
            <button>Predict</button>
        </form>
        <div id=\"output\" style=\"margin-top:1rem;\"></div>
        <script>
            async function fetchPrediction() {
                const t = document.getElementById("ticker").value;
                const res = await fetch(`/predict?ticker=${t}`);
                const data = await res.json();
                document.getElementById("output").innerHTML = 
                    `<b>${data.ticker}</b>: ${data.prediction} (Confidence: ${data.confidence || 'N/A'})`;
            }
        </script>
    </body>
    </html>
    """

@app.get("/predict")
def predict(ticker: str = Query(...)):
    try:
        model_path = f"models/rf_{ticker.upper()}.pkl"
        if not os.path.exists(model_path):
            return {"prediction": "ERROR: Model not found", "ticker": ticker.upper(), "timestamp": datetime.now()}

        model = joblib.load(model_path)

        raw = yf.download([ticker, "TLT", "GLD"], period="6mo")
        if "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].dropna()
        else:
            prices = raw["Close"].dropna()

        data = engineer_features(prices, ticker.upper())
        latest = data.drop(columns=["target"]).iloc[-1:]

        scaler = StandardScaler().fit(data.drop(columns=["target"]))
        X = scaler.transform(latest)
        pred = model.predict(X)[0]
        conf = model.predict_proba(X)[0][1]  # probability of UP class

        return {
            "prediction": "UP" if pred == 1 else "DOWN",
            "confidence": round(float(conf), 4),
            "ticker": ticker.upper(),
            "timestamp": datetime.now()
        }

    except Exception as e:
        return {"prediction": f"ERROR: {str(e)}", "ticker": ticker.upper(), "timestamp": datetime.now()}

