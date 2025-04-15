import requests
import numpy as np
import pandas as pd
import datetime as dt
import torch

# A browser User-Agent header to help bypass simple scraping protections.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/90.0.4430.93 Safari/537.36")
}


class YahooDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.scaler = MinMaxScaler()

    def get_intraday_data(ticker, start_date, end_date, interval = "1h"):
        base_url = "https://query1.finance.yahoo.com"
        url = f"{base_url}/v8/finance/chart/{ticker}"
        period1 = int(pd.Timestamp(start_date).timestamp())
        period2 = int(pd.Timestamp(end_date).timestamp())
        params = {
            'interval': interval,
            'period1': period1,
            'period2': period2,
            'includePrePost': 'true'
        }
        response = requests.get(url, params=params, headers=HEADERS)
         # Debug: Check status code and content type/length
        print(f"URL: {response.url}")
        print("HTTP Status Code:", response.status_code)
        if response.status_code != 200:
            print("Error: Non-200 HTTP response")
            print(response.text)
            return pd.DataFrame()

        if not response.text:
            print("Error: Response is empty.")
            return pd.DataFrame()

        try:
            data = response.json()
        except Exception as e:
            print("Error decoding JSON:", e)
            print("Response content:", response.text)
            return pd.DataFrame()

        try:
            result = data['chart']['result'][0]
        except (KeyError, IndexError) as e:
            print("JSON structure error:", e)
            return pd.DataFrame()

        timestamps = result.get('timestamp')
        quote = result.get('indicators', {}).get('quote', [{}])[0]

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s') if timestamps else [],
            'open': quote.get('open', []),
            'high': quote.get('high', []),
            'low': quote.get('low', []),
            'close': quote.get('close', []),
            'volume': quote.get('volume', []),
        })

        return df

    def compute_intraday_indicators(df):
        """
        Compute technical indicators from intraday data:
        - MA20: 20-period moving average of 'close'
        - EMA12, EMA26: Exponential moving averages used for MACD
        - MACD: Difference between EMA12 and EMA26
        - Signal_Line: 9-period EMA of MACD
        - volume is retained as-is.

        Returns a DataFrame with columns: MA20, MACD, Signal_Line, volume.
        """
        df = df.copy()
        # Moving average (20-period)
        df['MA20'] = df['close'].rolling(window=20).mean()

        # Compute exponential moving averages for MACD calculation
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Select desired columns and drop rows with NA values (which occur during initial periods)
        indicators = df[['MA20', 'MACD', 'Signal_Line', 'volume']].dropna()
        return indicators

    def fetch_data(self, proxy = None, auto_adjust = False, interval = '1h') -> pd.DataFrame:
        fulldata_df = pd.DataFrame()
        for ticker in self.ticker_list:
            print(f"Fetching 1h data for {ticker} ...")
            stock_df = self.get_intraday_data(ticker, self.start_date, self.end_date,
                                   interval = interval)
            if stock_df.empty:
                print(f"No intraday data for {ticker}.")
                continue
            indicators = self.compute_intraday_indicators(stock_df)
            if indicators.empty:
                print(f"Not enough data to compute indicators for {ticker}.")
                continue
            norm_values = self.scaler.fit_transform(indicators.values)
            stock_df = pd.dataFrame(norm_values,
                                    columns = ['MA20', 'MACD', 'Signal_Line', 'Volume'])
            stock_df.insert(0, 'Ticker', ticker)
            fulldata_df = pd.concat([fulldata_df, stock_df])
        return fulldata_df
