import yfinance as yf

# Data Download

import yfinance as yf

TICKER = "SPY"

datasets = {
    "spy_daily.csv":  {"start": "2018-01-01", "end": "2026-04-30", "interval": "1d"},
    "spy_hourly.csv": {"period": "730d",  "interval": "1h"},  # max ~2 years
    "spy_5min.csv":   {"period": "60d",   "interval": "5m"},  # max 60 days
}

for filename, params in datasets.items():
    df = yf.download(TICKER, **params)
    df.to_csv(filename)
    print(f"{filename}: {len(df)} rows | {df.index[0]} to {df.index[-1]}")

