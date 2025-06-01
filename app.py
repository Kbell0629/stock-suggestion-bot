from flask import Flask, render_template
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import os
import time
from ratelimit import limits, sleep_and_retry
import psutil
import pickle
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your NewsAPI key
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "c715f7725a3147dfbfa89e8d51ecd49b")

# Portfolio settings
DAILY_INVESTMENT = 2500
STOCKS_PER_DAY = 5
PER_STOCK_INVESTMENT = DAILY_INVESTMENT / STOCKS_PER_DAY
PORTFOLIO_SIZE = 9000
MAX_ENTRY_PRICE = 100
MIN_ENTRY_PRICE = 5
PENNY_PRICE = 30

# Cache settings
CACHE_FILE = "stock_data_cache.pkl"
CACHE_DURATION = timedelta(hours=1)

# Fallback ticker list
FALLBACK_TICKERS = [
    'SIRI', 'F', 'NIO', 'AMD', 'BA', 'DIS', 'CSCO', 'INTC', 'PFE', 'T',
    'VZ', 'CMCSA', 'QCOM', 'AMAT', 'MU', 'GILD', 'OKTA', 'ZS', 'DDOG', 'CRWD',
    'PYPL', 'SQ', 'UBER', 'ZM', 'RBLX', 'PINS', 'SNAP', 'PLTR', 'SOFI', 'LCID'
]

# Hardcoded DJIA tickers
DJIA_FALLBACK = [
    'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'KO', 'DIS', 'GS', 'HD'
]

# Fetch tickers
def get_all_tickers():
    logger.info("Fetching all tickers...")
    tickers = set()
    try:
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables_sp500 = pd.read_html(url_sp500)
        sp500_table = tables_sp500[0]
        sp500_tickers = None
        for col in ['Symbol', 'Ticker', 'Stock Symbol', 'Ticker symbol']:
            if col in sp500_table.columns:
                sp500_tickers = sp500_table[col].tolist()
                break
        if sp500_tickers:
            tickers.update([ticker.replace('.', '-') for ticker in sp500_tickers])
            logger.info(f"Added {len(sp500_tickers)} S&P 500 tickers")
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")

    try:
        url_djia = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tables_djia = pd.read_html(url_djia, match="Constituents")
        djia_table = tables_djia[0]
        djia_tickers = None
        for col in ['Symbol', 'Ticker', 'Stock Symbol', 'Ticker symbol']:
            if col in djia_table.columns:
                djia_tickers = djia_table[col].tolist()
                break
        if djia_tickers is None:
            logger.info("Using DJIA fallback tickers")
            djia_tickers = DJIA_FALLBACK
        tickers.update([ticker.replace('.', '-') for ticker in djia_tickers])
        logger.info(f"Added {len(djia_tickers)} DJIA tickers")
    except Exception as e:
        logger.error(f"Error fetching DJIA tickers: {e}")
        tickers.update([ticker.replace('.', '-') for ticker in DJIA_FALLBACK])

    try:
        url_nasdaq = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables_nasdaq = pd.read_html(url_nasdaq, match="Company")
        nasdaq_table = tables_nasdaq[0]
        nasdaq_tickers = None
        for col in ['Ticker', 'Symbol', 'Stock Symbol']:
            if col in nasdaq_table.columns:
                nasdaq_tickers = nasdaq_table[col].tolist()
                break
        if nasdaq_tickers:
            tickers.update([ticker.replace('.', '-') for ticker in nasdaq_tickers])
            logger.info(f"Added {len(nasdaq_tickers)} Nasdaq-100 tickers")
    except Exception as e:
        logger.error(f"Error fetching Nasdaq-100 tickers: {e}")

    if not tickers:
        logger.info("No tickers retrieved, using fallback tickers")
        return FALLBACK_TICKERS

    tickers = list(tickers)
    logger.info(f"Retrieved {len(tickers)} unique tickers")
    return tickers[:30]  # Reduced to 30

STOCKS = get_all_tickers()

def load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            if cache['timestamp'] > datetime.now() - CACHE_DURATION:
                logger.info("Loaded valid cache")
                return cache['data']
        return {}
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return {}

def save_cache(data):
    try:
        cache = {'timestamp': datetime.now(), 'data': data}
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        logger.info("Saved cache")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

@sleep_and_retry
@limits(calls=1, period=1)
def get_stock_data(ticker):
    logger.info(f"Processing {ticker}...")
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    cache = load_cache()
    if ticker in cache:
        logger.info(f"Using cached data for {ticker}")
        return cache[ticker]

    for attempt in range(4):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo", interval="1d")

            if hist.empty or len(hist) < 20:
                logger.warning(f"No data for {ticker}")
                return None

            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            exp1 = hist['Close'].ewm(span=5, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=13, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=5, adjust=False).mean()

            macd_values = macd[-3:]
            macd_momentum = all(macd_values[i] < macd_values[i+1] for i in range(len(macd_values)-1))

            sma = hist['Close'].rolling(window=20).mean()
            std = hist['Close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            returns = hist['Close'].pct_change()
            volatility = returns.std() * (252 ** 0.5)

            avg_volume = hist['Volume'].rolling(window=20).mean()[-1] if not hist['Volume'].empty else 0

            pe_ratio = stock.info.get('trailingPE', 0)

            current_price = hist['Close'][-1]

            high_20d = hist['High'].rolling(window=20).max()[-1]

            price_5d_ago = hist['Close'][-6] if len(hist) >= 6 else current_price
            price_momentum = (current_price - price_5d_ago) / price_5d_ago

            entry_price = sma[-1] * 1.01

            if not (entry_price <= current_price <= entry_price * 1.05):
                logger.info(f"{ticker} current price {current_price:.2f} not within entry range {entry_price:.2f}–{entry_price*1.05:.2f}")
                return None

            if volatility <= 0.6:
                logger.info(f"{ticker} volatility {volatility:.2f} too low")
                return None

            if avg_volume < 10000:
                logger.info(f"{ticker} volume {avg_volume:.0f} too low")
                return None

            target_price = entry_price * 1.10
            stop_loss = entry_price * 0.95

            shares = min(PER_STOCK_INVESTMENT // current_price, PORTFOLIO_SIZE // (current_price * STOCKS_PER_DAY))
            if shares < 1:
                logger.info(f"Cannot afford {ticker} at current price {current_price:.2f}")
                return None

            result = {
                'rsi': rsi[-1] if not pd.isna(rsi[-1]) else 50,
                'macd': macd[-1] - signal[-1] if not pd.isna(macd[-1]) else 0,
                'macd_momentum': macd_momentum,
                'price': current_price,
                'volatility': volatility,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'shares': int(shares),
                'upper_band': upper_band[-1],
                'lower_band': lower_band[-1],
                'pe_ratio': pe_ratio,
                'avg_volume': avg_volume,
                'high_20d': high_20d,
                'price_momentum': price_momentum
            }

            del hist, delta, gain, loss, rs, rsi, exp1, exp2, macd, signal, sma, std, upper_band, lower_band, returns
            mem_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage for {ticker}: {mem_after - mem_before:.2f} MB")

            cache[ticker] = result
            save_cache(cache)
            return result

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            if "Too Many Requests" in str(e) and attempt < 3:
                wait_time = 4 ** (attempt + 1)  # 4, 16, 64 seconds
                logger.info(f"Retrying {ticker} after {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return None

@sleep_and_retry
@limits(calls=5, period=60)
def get_analyst_sentiment(ticker):
    logger.info(f"Fetching analyst sentiment for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        recommendation = stock.info.get('recommendationMean', 3.0)
        sentiment = (3.0 - recommendation) / 2.0
        logger.info(f"Analyst sentiment for {ticker}: {sentiment:.2f}")
        return sentiment
    except Exception as e:
        logger.error(f"Analyst sentiment error for {ticker}: {e}")
        return 0

def get_news_sentiment(ticker):
    logger.info(f"Fetching news for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', '').split()[0] or ticker
        query = f"{ticker} OR {company_name}"
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.error(f"NewsAPI error for {ticker}: Status {response.status_code}")
            return 0
        data = response.json()
        articles = data.get('articles', [])
        if not articles:
            logger.info(f"No articles found for {ticker}")
            return 0
        sentiment_score = 0
        count = 0
        for article in articles[:5]:
            text = article.get('title', '') + ' ' + article.get('description', '')
            if text.strip():
                analysis = TextBlob(text)
                sentiment_score += analysis.sentiment.polarity
                count += 1
        if count == 0:
            logger.info(f"No valid articles for {ticker}")
            return 0
        avg_sentiment = sentiment_score / count
        logger.info(f"News sentiment for {ticker}: {avg_sentiment:.2f}")
        return avg_sentiment
    except Exception as e:
        logger.error(f"News error for {ticker}: {e}")
        return 0

def select_stocks():
    logger.info("Selecting stocks...")
    stock_scores = []
    for ticker in STOCKS:
        try:
            data = get_stock_data(ticker)
            if not data:
                logger.info(f"Skipping {ticker} due to missing data")
                continue
            sentiment = get_analyst_sentiment(ticker)

            score = 0
            if 10 < data['rsi'] < 70:
                score += 30
            if data['macd'] > 0:
                score += 20
            if data['macd_momentum']:
                score += 10
            if data['price'] < data['upper_band']:
                score += 20
            if data['volatility'] > 0.6:
                score += 30
            if data['pe_ratio'] != 0 and data['pe_ratio'] < 20:
                score += 10
            if data['avg_volume'] > 10000:
                score += 20
            if data['entry_price'] <= PENNY_PRICE:
                score += 20
            if abs(data['price'] - data['high_20d']) / data['high_20d'] < 0.10:
                score += 30
            if data['price_momentum'] > 0.0:
                score += 30
            score += sentiment * 30

            stock_data = {
                'ticker': ticker,
                'score': score,
                'price': data['price'],
                'rsi': data['rsi'],
                'macd': data['macd'],
                'volatility': data['volatility'],
                'sentiment': sentiment,
                'entry_price': data['entry_price'],
                'target_price': data['target_price'],
                'stop_loss': data['stop_loss'],
                'shares': data['shares'],
                'pe_ratio': data.get('pe_ratio', 0),
                'avg_volume': data.get('avg_volume', 0)
            }
            stock_scores.append(stock_data)
            logger.info(f"Scored {ticker}: Score {score}")
        except Exception as e:
            logger.error(f"Error in scoring {ticker}: {e}")
            continue

    if len(stock_scores) < 5:
        logger.info(f"Only {len(stock_scores)} stocks scored. Using fallback scoring.")
        for ticker in STOCKS[:20]:
            try:
                data = get_stock_data(ticker)
                if not data:
                    continue
                sentiment = get_analyst_sentiment(ticker)

                score = 0
                if data['entry_price'] <= MAX_ENTRY_PRICE:
                    score += 50
                if data['volatility'] >= 0.5:
                    score += 30
                score += sentiment * 20

                stock_data = {
                    'ticker': ticker,
                    'score': score,
                    'price': data['price'],
                    'rsi': data['rsi'],
                    'macd': data['macd'],
                    'volatility': data['volatility'],
                    'sentiment': sentiment,
                    'entry_price': data['entry_price'],
                    'target_price': data['target_price'],
                    'stop_loss': data['stop_loss'],
                    'shares': data['shares'],
                    'pe_ratio': data.get('pe_ratio', 0),
                    'avg_volume': data.get('avg_volume', 0)
                }
                stock_scores.append(stock_data)
                logger.info(f"Fallback scored {ticker}: Score {score}")
            except Exception as e:
                logger.error(f"Error in fallback scoring {ticker}: {e}")
                continue

    if not stock_scores:
        logger.warning("No stocks scored. Using static fallback.")
        return [
            {'ticker': 'SIRI', 'score': 100, 'price': 5.60, 'rsi': 50, 'macd': 0, 'volatility': 0.9, 'sentiment': 0.5, 'entry_price': 5.50, 'target_price': 6.05, 'stop_loss': 5.23, 'shares': 89, 'pe_ratio': 9.5, 'avg_volume': 4000000},
            {'ticker': 'F', 'score': 95, 'price': 10.30, 'rsi': 50, 'macd': 0, 'volatility': 0.85, 'sentiment': 0.5, 'entry_price': 10.20, 'target_price': 11.22, 'stop_loss': 9.69, 'shares': 48, 'pe_ratio': 8.5, 'avg_volume': 3000000},
            {'ticker': 'NIO', 'score': 90, 'price': 18.20, 'rsi': 50, 'macd': 0, 'volatility': 0.8, 'sentiment': 0.5, 'entry_price': 18.00, 'target_price': 19.80, 'stop_loss': 17.10, 'shares': 27, 'pe_ratio': 0, 'avg_volume': 2000000},
            {'ticker': 'PLTR', 'score': 85, 'price': 25.50, 'rsi': 50, 'macd': 0, 'volatility': 0.95, 'sentiment': 0.5, 'entry_price': 25.00, 'target_price': 27.50, 'stop_loss': 23.75, 'shares': 19, 'pe_ratio': 0, 'avg_volume': 1500000},
            {'ticker': 'ROKU', 'score': 80, 'price': 29.80, 'rsi': 50, 'macd': 0, 'volatility': 1.0, 'sentiment': 0.5, 'entry_price': 29.50, 'target_price': 32.45, 'stop_loss': 28.03, 'shares': 16, 'pe_ratio': 0, 'avg_volume': 1000000}
        ]

    sorted_stocks = sorted(stock_scores, key=lambda x: x['score'], reverse=True)[:5]
    logger.info(f"Selected {len(sorted_stocks)} stocks")
    return sorted_stocks

@app.route('/')
def home():
    logger.info("Rendering homepage...")
    try:
        stocks = select_stocks()
        return render_template('index.html', stocks=stocks, date=datetime.now().strftime("%Y-%m-%d"))
    except Exception as e:
        logger.error(f"Error rendering homepage: {e}")
        return render_template('index.html', stocks=[
            {'ticker': 'SIRI', 'score': 100, 'price': 5.60, 'rsi': 50, 'macd': 0, 'volatility': 0.9, 'sentiment': 0.5, 'entry_price': 5.50, 'target_price': 6.05, 'stop_loss': 5.23, 'shares': 89, 'pe_ratio': 9.5, 'avg_volume': 4000000},
            {'ticker': 'F', 'score': 95, 'price': 10.30, 'rsi': 50, 'macd': 0, 'volatility': 0.85, 'sentiment': 0.5, 'entry_price': 10.20, 'target_price': 11.22, 'stop_loss': 9.69, 'shares': 48, 'pe_ratio': 8.5, 'avg_volume': 3000000},
            {'ticker': 'NIO', 'score': 90, 'price': 18.20, 'rsi': 50, 'macd': 0, 'volatility': 0.8, 'sentiment': 0.5, 'entry_price': 18.00, 'target_price': 19.80, 'stop_loss': 17.10, 'shares': 27, 'pe_ratio': 0, 'avg_volume': 2000000},
            {'ticker': 'PLTR', 'score': 85, 'price': 25.50, 'rsi': 50, 'macd': 0, 'volatility': 0.95, 'sentiment': 0.5, 'entry_price': 25.00, 'target_price': 27.50, 'stop_loss': 23.75, 'shares': 19, 'pe_ratio': 0, 'avg_volume': 1500000},
            {'ticker': 'ROKU', 'score': 80, 'price': 29.80, 'rsi': 50, 'macd': 0, 'volatility': 1.0, 'sentiment': 0.5, 'entry_price': 29.50, 'target_price': 32.45, 'stop_loss': 28.03, 'shares': 16, 'pe_ratio': 0, 'avg_volume': 1000000}
        ], date=datetime.now().strftime("%Y-%m-%d"), error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)