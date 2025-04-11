import requests
import datetime as dt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
from transformers import pipeline
import random
from dateutil import parser as date_parser
from datetime import timedelta

# -----------------------------
# CONFIGURATION & PARAMETERS
# -----------------------------

# List of 20 stock tickers (sample tickers – adjust as needed)
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'INTC',
    'ORCL', 'IBM', 'CRM', 'PYPL', 'QCOM', 'TXN', 'AVGO', 'AMD', 'SBUX', 'UBER'
]
random.shuffle(stock_tickers)
selected_stocks = stock_tickers[:20]

# Define the date range for the technical data (using Unix timestamps for the API).
# WARNING: Yahoo Finance free API for intraday data often allows only ~60 days of data.
start_date_str = '2023-09-01'
end_date_str = '2025-03-31'  # adjust as needed within allowed limits for 1h data

# For news, we use a period (e.g., 2024-05-01 to 2025-03-31)
news_start_date = dt.datetime.strptime('2024-05-01', '%Y-%m-%d').date()
news_end_date = dt.datetime.strptime('2025-03-31', '%Y-%m-%d').date()

# BERT sentiment pipeline (using DistilBERT fine-tuned on SST-2)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# A browser User-Agent header to help bypass simple scraping protections.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/90.0.4430.93 Safari/537.36")
}

# -----------------------------
# HELPER FUNCTIONS: INTRADAY DATA FROM YAHOO FINANCE
# -----------------------------

def get_intraday_data(ticker, start_date, end_date, interval='1h'):
    base_url = 'https://query1.finance.yahoo.com'
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

# -----------------------------
# HELPER FUNCTIONS: INVESTING.COM NEWS SCRAPING & SENTIMENT
# -----------------------------

def scrape_investing_articles(stock, from_date, to_date):
    """
    Scrapes news articles for the provided stock from Investing.com.
    Uses the search page and heuristically follows links with '/news/' in the URL.
    
    Returns:
      A list of tuples (published_date, article_text).
    """
    articles = []
    search_url = f"https://www.investing.com/search/?q={stock}"
    
    try:
        r = requests.get(search_url, headers=HEADERS, timeout=10)
    except Exception as e:
        print(f"Error fetching search results for {stock}: {e}")
        return articles

    soup = BeautifulSoup(r.content, 'html.parser')
    
    # Heuristically locate <a> tags with '/news/' in their href.
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        if "/news/" in href:
            full_url = "https://www.investing.com" + href
            try:
                art_resp = requests.get(full_url, headers=HEADERS, timeout=10)
                art_soup = BeautifulSoup(art_resp.content, 'html.parser')
                
                # Extract publication date (adjust the selector as needed)
                date_elem = art_soup.find("span", class_="date")
                if not date_elem:
                    continue
                date_str = date_elem.get_text(strip=True)
                try:
                    pub_date = date_parser.parse(date_str, fuzzy=True).date()
                except Exception as e:
                    print(f"Date parsing error for {full_url}: {e}")
                    continue
                
                # Filter by requested date range
                if not (from_date <= pub_date <= to_date):
                    continue
                
                # Extract article content (adjust the selector as needed)
                content_div = art_soup.find("div", class_="WYSIWYG articlePage")
                if not content_div:
                    content_div = art_soup.find("div", id="article-content")
                article_text = content_div.get_text(" ", strip=True) if content_div else ""
                if article_text:
                    articles.append((pub_date, article_text))
            except Exception as e:
                print(f"Error processing article {full_url}: {e}")
                continue
    return articles

def aggregate_investing_news(stock, from_date, to_date):
    """
    Aggregates Investing.com articles per day for the given stock.
    Returns a dictionary mapping day (YYYY-MM-DD) to concatenated article texts.
    """
    scraped_articles = scrape_investing_articles(stock, from_date, to_date)
    daily_news = {}
    for art_date, art_text in scraped_articles:
        day_str = art_date.strftime("%Y-%m-%d")
        if day_str in daily_news:
            daily_news[day_str] += " " + art_text
        else:
            daily_news[day_str] = art_text
    return daily_news

def get_continuous_sentiment(text):
    """
    Uses the pretrained BERT sentiment pipeline to evaluate text.
    Returns a continuous score: positive score for "POSITIVE" and negative for "NEGATIVE".
    """
    if not text.strip():
        return 0.0
    result = sentiment_pipeline(text)
    score = result[0]['score']
    label = result[0]['label']
    return score if label.upper() == "POSITIVE" else -score

def create_date_range(start_date, end_date):
    """
    Returns a list of dates (date objects) from start_date to end_date inclusive.
    """
    num_days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(num_days)]

# -----------------------------
# STEP 1: FETCH 1h INTRADAY DATA & COMPUTE TECHNICAL INDICATORS
# -----------------------------

# Instead of a list, store each stock's technical indicator tensor in a dictionary.
tech_tensors = {}
scaler = MinMaxScaler()

for symbol in selected_stocks:
    print(f"Fetching 1h data for {symbol} ...")
    df = get_intraday_data(symbol, start_date_str, end_date_str, interval='1h')
    if df.empty:
        print(f"No intraday data for {symbol}.")
        continue
    indicators = compute_intraday_indicators(df)
    if indicators.empty:
        print(f"Not enough data to compute indicators for {symbol}.")
        continue
    # Normalize indicator values (each row corresponds to a timestamp and columns are technical indicators)
    norm_values = scaler.fit_transform(indicators.values)
    tensor_data = torch.tensor(norm_values, dtype=torch.float32)
    # Transpose so that rows represent technical indicators and columns are the 1-hour time stamps.
    tensor_data = tensor_data.transpose(0, 1)  
    tech_tensors[symbol] = tensor_data

# -----------------------------
# STEP 2: SCRAPE INVESTING.COM NEWS & CALCULATE DAILY SENTIMENT
# -----------------------------

# Create a date range for news aggregation (daily, from news_start_date to news_end_date)
date_list = create_date_range(news_start_date, news_end_date)
date_str_list = [d.strftime("%Y-%m-%d") for d in date_list]

sentiment_results = {}

for symbol in selected_stocks:
    print(f"Scraping Investing.com news and calculating sentiment for {symbol} ...")
    daily_news = aggregate_investing_news(symbol, news_start_date, news_end_date)
    daily_sentiments = {}
    for day_str in date_str_list:
        text = daily_news.get(day_str, "")
        score = get_continuous_sentiment(text)
        daily_sentiments[day_str] = score
    sentiment_results[symbol] = daily_sentiments

# Instead of stacking into one tensor, store each stock's sentiment tensor in a dictionary.
sentiment_tensors = {}
for symbol in selected_stocks:
    scores = [sentiment_results[symbol][day] for day in date_str_list]
    # Create a tensor with one column (each row is a daily sentiment score)
    stock_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)  
    sentiment_tensors[symbol] = stock_tensor

# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("\nTechnical Indicator Tensors (each tensor shape: [n_indicators, n_time_stamps]):")
for symbol, tensor in tech_tensors.items():
    print(f"{symbol}: {tensor.shape}")

print("\nSentiment Tensors (each tensor shape: [n_days, 1]):")
for symbol, tensor in sentiment_tensors.items():
    print(f"{symbol}: {tensor.shape}")


print("Technical Tensor for AAPL:")
print(tech_tensors.get("AAPL"))

print("\nSentiment Tensor for AAPL:")
print(sentiment_tensors.get("AAPL"))
