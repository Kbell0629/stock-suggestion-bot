from flask import Flask, render_template
import pandas as pd
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import os
import time
from ratelimit import limits, sleep_and_retry
import pickle
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "ZKIM7MWR7PHXHYWU")
FINLIGHT_KEY = os.getenv("FINLIGHT_KEY", "sk_be774f798075218cf67071d562868776e91d910f1166ee36254e52fa94bc1601")

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

# Fetch tickers (simplified to fallback for speed)
def get_all_tickers():
    logger.info("Using fallback tickers for simplicity")
    return FALLBACK_TICKERS[:30]  # Limit to 30

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
@limits(calls=5, period=60)  # Alpha Vantage free tier: 5 calls/minute
def get_stock_data(ticker):
    logger.info(f"Processing {ticker}...")
    cache = load_cache()
    if ticker in cache:
        logger.info(f"Using cached data for {ticker}")
        return cache[ticker]

    for attempt in range(4):
        try:
            # Fetch daily adjusted prices
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Alpha Vantage error for {ticker}: Status {response.status_code}")
                return None
            data = response.json()
            if "Time Series (Daily)" not in data:
                logger.warning(f"No data for {ticker}")
                return None

            hist = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index').astype(float)
            hist.index = pd.to_datetime(hist.index)
            hist = hist.sort_index()

            if len(hist) < 20:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            delta = hist['4. close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            exp1 = hist['4. close'].ewm(span=5, adjust=False).mean()
            exp2 = hist['4. close'].ewm(span=13, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=5, adjust=False).mean()

            macd_values = macd[-3:]
            macd_momentum = all(macd_values[i] < macd_values[i+1] for i in range(len(macd_values)-1))

            sma = hist['4. close'].rolling(window=20).mean()
            std = hist['4. close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            returns = hist['4. close'].pct_change()
            volatility = returns.std() * (252 ** 0.5)

            avg_volume = hist['6. volume'].rolling(window=20).mean()[-1]

            # Fetch fundamentals
            url_fund = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
            response_fund = requests.get(url_fund, timeout=10)
            fund_data = response_fund.json()
            pe_ratio = float(fund_data.get('PERatio', 0)) if fund_data.get('PERatio') != 'None' else 0

            current_price = hist['4. close'][-1]
            high_20d = hist['2. high'].rolling(window=20).max()[-1]
            price_5d_ago = hist['4. close'][-6] if len(hist) >= 6 else current_price
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
            cache[ticker] = result
            save_cache(cache)
            return result

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            if "Too Many Requests" in str(e) and attempt < 3:
                wait_time = 4 ** (attempt + 1)
                logger.info(f"Retrying {ticker} after {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return None

def get_news_sentiment(ticker):
    logger.info(f"Fetching news for {ticker}...")
    try:
        url = f"https://api.finlight.me/v1/news?ticker={ticker}&api_key={FINLIGHT_KEY}"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.error(f"Finlight error for {ticker}: Status {response.status_code}")
            return 0
        data = response.json()
        articles = data.get('articles', [])
        if not articles:
            logger.info(f"No articles found for {ticker}")
            return 0
        sentiment_score = 0
        count = 0
        for article in articles[:5]:
            text = article.get('title', '') + ' ' + article.get('summary', '')
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
            sentiment = get_news_sentiment(ticker)

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
        for ticker in STOCKS[:10]:
            try:
                data = get_stock_data(ticker)
                if not data:
                    continue
                sentiment = get_news_sentiment(ticker)

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