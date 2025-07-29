import time
import requests
from datetime import datetime

TICKERS = ["SPY", "AAPL", "TSLA"]
API_URL = "http://127.0.0.1:8000/predict?ticker={ticker}"

print("🔄 Starting real-time prediction loop (every 5 seconds)...")

while True:
    for ticker in TICKERS:
        try:
            response = requests.get(API_URL.format(ticker=ticker))
            if response.status_code == 200:
                data = response.json()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}: {data['prediction']}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}: ERROR {response.status_code}")
        except Exception as e:
            print(f"{ticker}: Exception: {e}")
    time.sleep(5)

