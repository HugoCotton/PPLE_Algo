import pandas as pd
import yfinance as yf

# 1. Get S&P 500 tickers
url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
sp500 = pd.read_csv(url)

tickers = sp500["Symbol"].tolist()
tickers = [t.replace(".", "-") for t in tickers]

# 2. Download 3 months of data for all S&P 500 tickers
data = yf.download(
    tickers,
    period="3mo",
    group_by="ticker",
    auto_adjust=False,
    progress=False
)

# 3. Compute average daily volume for each stock
volume_dict = {}

for ticker in tickers:
    try:
        avg_volume = data[ticker]["Volume"].dropna().mean()
        if pd.notna(avg_volume):
            volume_dict[ticker] = avg_volume
    except Exception:
        continue

# 4. Rank stocks by average daily volume and keep top 100
sorted_stocks = sorted(volume_dict.items(), key=lambda x: x[1], reverse=True)
top_100 = sorted_stocks[:100]
top_tickers = [t[0] for t in top_100]

# 5. Save ranking file
top_100_df = pd.DataFrame(top_100, columns=["Ticker", "AvgVolume"])
top_100_df["AvgVolume"] = top_100_df["AvgVolume"].round(0).astype(int)
top_100_df.to_csv("top_100_volume.csv", index=False)

print(top_100_df.head(10))
print("Done: saved top_100_volume.csv")

# 6. Download full OHLCV data for top 100 stocks
top_data = yf.download(
    top_tickers,
    period="3mo",
    group_by="ticker",
    auto_adjust=False,
    progress=False
)

# Fix column order if yfinance returns (Price, Ticker) instead of (Ticker, Price)
if top_data.columns.nlevels == 2 and top_data.columns[0][0] in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    top_data = top_data.swaplevel(axis=1)

top_data = top_data.sort_index(axis=1)

# 7. Convert to clean long format
rows = []

for ticker in top_tickers:
    if ticker in top_data.columns.get_level_values(0):
        df = top_data[ticker].copy()
        df = df.reset_index()
        df["Ticker"] = ticker
        df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        rows.append(df)

if len(rows) == 0:
    print("No ticker data was found in top_data.")
else:
    clean_ohlcv = pd.concat(rows, ignore_index=True)
    clean_ohlcv = clean_ohlcv.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    clean_ohlcv.to_csv("top_100_clean_ohlcv.csv", index=False)

    print("Done: saved top_100_clean_ohlcv.csv")
    print(clean_ohlcv.head())
    import time
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, date


    def run_pipeline():
        today = date.today().strftime("%Y-%m-%d")
        print(f"\n{'=' * 50}")
        print(f"Running pipeline for {today}")
        print(f"{'=' * 50}")

        # 1. Get S&P 500 tickers
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        sp500 = pd.read_csv(url)

        tickers = sp500["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]

        # 2. Download 3 months of data for all S&P 500 tickers
        data = yf.download(
            tickers,
            period="3mo",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        # 3. Compute average daily volume for each stock
        volume_dict = {}

        for ticker in tickers:
            try:
                avg_volume = data[ticker]["Volume"].dropna().mean()
                if pd.notna(avg_volume):
                    volume_dict[ticker] = avg_volume
            except Exception:
                continue

        # 4. Rank stocks by average daily volume and keep top 100
        sorted_stocks = sorted(volume_dict.items(), key=lambda x: x[1], reverse=True)
        top_100 = sorted_stocks[:100]
        top_tickers = [t[0] for t in top_100]

        # 5. Save ranking file — dated
        top_100_df = pd.DataFrame(top_100, columns=["Ticker", "AvgVolume"])
        top_100_df["AvgVolume"] = top_100_df["AvgVolume"].round(0).astype(int)
        top_100_df.to_csv(f"top_100_volume_{today}.csv", index=False)

        print(top_100_df.head(10))
        print(f"Saved: top_100_volume_{today}.csv")

        # 6. Download full OHLCV data for top 100 stocks
        top_data = yf.download(
            top_tickers,
            period="3mo",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        # Fix column order if yfinance returns (Price, Ticker) instead of (Ticker, Price)
        if top_data.columns.nlevels == 2 and top_data.columns[0][0] in ["Open", "High", "Low", "Close", "Adj Close",
                                                                        "Volume"]:
            top_data = top_data.swaplevel(axis=1)

        top_data = top_data.sort_index(axis=1)

        # 7. Convert to clean long format
        rows = []

        for ticker in top_tickers:
            if ticker in top_data.columns.get_level_values(0):
                df = top_data[ticker].copy()
                df = df.reset_index()
                df["Ticker"] = ticker
                df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                rows.append(df)

        if len(rows) == 0:
            print("No ticker data was found in top_data.")
        else:
            clean_ohlcv = pd.concat(rows, ignore_index=True)
            clean_ohlcv = clean_ohlcv.sort_values(["Ticker", "Date"]).reset_index(drop=True)
            clean_ohlcv.to_csv(f"top_100_clean_ohlcv_{today}.csv", index=False)

            print(f"Saved: top_100_clean_ohlcv_{today}.csv")
            print(clean_ohlcv.head())


    def wait_until_next_run(run_hour=7, run_minute=0):
        """Sleep until the next occurrence of run_hour:run_minute."""
        now = datetime.now()
        next_run = now.replace(hour=run_hour, minute=run_minute, second=0, microsecond=0)

        # If that time has already passed today, schedule for tomorrow
        if now >= next_run:
            next_run = next_run.replace(day=now.day + 1)

        wait_seconds = (next_run - now).total_seconds()
        print(f"\nNext run scheduled at {next_run.strftime('%Y-%m-%d %H:%M:%S')} "
              f"({wait_seconds / 3600:.1f} hours from now)")
        time.sleep(wait_seconds)


    # ── Run loop ──────────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        while True:
            run_pipeline()
            wait_until_next_run(run_hour=7, run_minute=0)  # change time here if needed