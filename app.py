from flask import Flask, render_template
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import requests
from datetime import datetime
import os
import time
from ratelimit import limits, sleep_and_retry

app = Flask(__name__)

# Your NewsAPI key (replace with your actual key)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "c715f7725a3147dfbfa89e8d51ecd49b")

# Portfolio settings
DAILY_INVESTMENT = 2500  # Total $2,500 per day
STOCKS_PER_DAY = 5       # Spread across 5 stocks
PER_STOCK_INVESTMENT = DAILY_INVESTMENT / STOCKS_PER_DAY  # $500 per stock
PORTFOLIO_SIZE = 9000    # Total portfolio $9,000
MAX_ENTRY_PRICE = 100    # Max entry price
MIN_ENTRY_PRICE = 5      # Min entry price for penny-like stocks
PENNY_PRICE = 30         # Target penny-like stocks ≤ $30

# Expanded fallback ticker list
FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'WMT', 'V',
    'SIRI', 'F', 'NIO', 'AMD', 'BA', 'DIS', 'NFLX', 'CSCO', 'INTC', 'PFE',
    'T', 'VZ', 'CMCSA', 'ADBE', 'CRM', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'MU',
    'GILD', 'REGN', 'BIIB', 'OKTA', 'ZS', 'DDOG', 'CRWD', 'PANW', 'SPLK', 'NOW',
    'PYPL', 'SQ', 'SHOP', 'UBER', 'LYFT', 'ZM', 'DOCU', 'RBLX', 'PINS', 'SNAP',
    'X', 'PLTR', 'SOFI', 'LCID', 'RIVN', 'CHPT', 'NKLA', 'BB', 'PLUG', 'FCEL',
    'SPCE', 'WKHS', 'MARA', 'RIOT', 'CLOV', 'TLRY', 'AMC', 'GME', 'BBBY', 'KOSS',
    'AAL', 'UAL', 'DAL', 'LUV', 'CCL', 'NCLH', 'RCL', 'MGM', 'WYNN', 'CZR',
    'DKNG', 'PENN', 'BYND', 'PLNT', 'ROKU', 'TDOC', 'ETSY', 'CHWY', 'W', 'RVLV',
    'MMM', 'AXP', 'CAT', 'CVX', 'GS', 'HD', 'IBM', 'JNJ', 'KO', 'MCD', 'NKE', 'PG',
    'AMGN', 'DOW', 'HON', 'MRK', 'TRV', 'UNH', 'WBA', 'XOM', 'CVS', 'LOW', 'UPS',
    'TGT', 'FDX', 'BKNG', 'MDT', 'GPN', 'SYK', 'ISRG', 'EL', 'ZTS', 'CL', 'KMB',
    'SHW', 'ECL', 'ROP', 'SPGI', 'MCO', 'CTAS', 'AOS', 'TT', 'ITW', 'PH',
    'APD', 'EOG', 'SLB', 'HAL', 'BKR', 'VLO', 'PSX', 'MPC', 'HES', 'OXY', 'FANG',
    'APA', 'DVN', 'PXD', 'MRO', 'COP', 'NUE', 'STLD', 'FCX', 'NEM', 'GOLD'
]

# Hardcoded DJIA tickers as fallback
DJIA_FALLBACK = [
    'MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
    'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK',
    'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT'
]

# Fetch tickers for S&P 500, DJIA, and Nasdaq-100
def get_all_tickers():
    """Get tickers from S&P 500, DJIA, and Nasdaq-100."""
    print("Fetching all tickers...")
    tickers = set()
    try:
        # S&P 500
        try:
            url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables_sp500 = pd.read_html(url_sp500)
            sp500_table = tables_sp500[0]
            sp500_tickers = None
            for col in ['Symbol', 'Ticker', 'Stock Symbol', 'Ticker symbol']:
                if col in sp500_table.columns:
                    sp500_tickers = sp500_table[col].tolist()
                    break
            if sp500_tickers is None:
                print(f"S&P 500 table columns: {sp500_table.columns.tolist()}")
                raise KeyError("No Symbol, Ticker, or Stock Symbol column in S&P 500 table")
            tickers.update([ticker.replace('.', '-') for ticker in sp500_tickers])
            print(f"Added {len(sp500_tickers)} S&P 500 tickers")
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")

        # DJIA
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
                print(f"DJIA table columns: {djia_table.columns.tolist()}")
                print("Using DJIA fallback tickers")
                djia_tickers = DJIA_FALLBACK
            tickers.update([ticker.replace('.', '-') for ticker in djia_tickers])
            print(f"Added {len(djia_tickers)} DJIA tickers")
        except Exception as e:
            print(f"Error fetching DJIA tickers: {e}")
            print("Using DJIA fallback tickers")
            tickers.update([ticker.replace('.', '-') for ticker in DJIA_FALLBACK])
            print(f"Added {len(DJIA_FALLBACK)} DJIA fallback tickers")

        # Nasdaq-100
        try:
            url_nasdaq = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables_nasdaq = pd.read_html(url_nasdaq, match="Company")
            nasdaq_table = tables_nasdaq[0]
            nasdaq_tickers = None
            for col in ['Ticker', 'Symbol', 'Stock Symbol']:
                if col in nasdaq_table.columns:
                    nasdaq_tickers = nasdaq_table[col].tolist()
                    break
            if nasdaq_tickers is None:
                print(f"Nasdaq-100 table columns: {nasdaq_table.columns.tolist()}")
                raise KeyError("No Ticker, Symbol, or Stock Symbol column in Nasdaq-100 table")
            tickers.update([ticker.replace('.', '-') for ticker in nasdaq_tickers])
            print(f"Added {len(nasdaq_tickers)} Nasdaq-100 tickers")
        except Exception as e:
            print(f"Error fetching Nasdaq-100 tickers: {e}")

        if not tickers:
            print("No tickers retrieved, using fallback tickers")
            return FALLBACK_TICKERS

        tickers = list(tickers)
        print(f"Retrieved {len(tickers)} unique tickers")
        return tickers[:300]  # Limit to 300 for performance
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        print("Using fallback tickers")
        return FALLBACK_TICKERS

# List of stocks to analyze
STOCKS = get_all_tickers()

# Rate limit yfinance requests: 2 calls per second
@sleep_and_retry
@limits(calls=2, period=1)
def get_stock_data(ticker):
    """Fetch stock data and calculate indicators."""
    print(f"Processing {ticker}...")
    for attempt in range(3):  # Retry up to 3 times
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo", interval="1d")

            if hist.empty or len(hist) < 20:
                print(f"No data for {ticker}")
                return None

            # Calculate 5-day RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Calculate fast MACD (5, 13, 5)
            exp1 = hist['Close'].ewm(span=5, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=13, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=5, adjust=False).mean()

            # MACD momentum (increasing over 2 days)
            macd_values = macd[-3:]
            macd_momentum = all(macd_values[i] < macd_values[i+1] for i in range(len(macd_values)-1))

            # Calculate Bollinger Bands and SMA
            sma = hist['Close'].rolling(window=20).mean()
            std = hist['Close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            # Volatility
            returns = hist['Close'].pct_change()
            volatility = returns.std() * (252 ** 0.5)

            # Volume (20-day average)
            avg_volume = hist['Volume'].rolling(window=20).mean()[-1] if not hist['Volume'].empty else 0

            # Fundamentals (P/E Ratio)
            pe_ratio = stock.info.get('trailingPE', 0)

            # Current price
            current_price = hist['Close'][-1]

            # 20-day high for breakout detection
            high_20d = hist['High'].rolling(window=20).max()[-1]

            # 5-day price momentum
            price_5d_ago = hist['Close'][-6] if len(hist) >= 6 else current_price
            price_momentum = (current_price - price_5d_ago) / price_5d_ago

            # Entry price (20-day SMA + 1%)
            entry_price = sma[-1] * 1.01

            # Filter for stocks where current price has just passed entry price
            if not (entry_price <= current_price <= entry_price * 1.05):
                print(f"{ticker} current price {current_price:.2f} not within entry range {entry_price:.2f}–{entry_price*1.05:.2f}")
                return None

            # Filter for high volatility
            if volatility <= 0.6:
                print(f"{ticker} volatility {volatility:.2f} too low")
                return None

            # Filter for minimum volume
            if avg_volume < 10000:
                print(f"{ticker} volume {avg_volume:.0f} too low")
                return None

            # Exit prices based on entry price
            target_price = entry_price * 1.10  # 10% above entry
            stop_loss = entry_price * 0.95    # 5% below entry

            # Shares to buy based on current price
            shares = min(PER_STOCK_INVESTMENT // current_price, PORTFOLIO_SIZE // (current_price * STOCKS_PER_DAY))
            if shares < 1:
                print(f"Cannot afford {ticker} at current price {current_price:.2f}")
                return None

            return {
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
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            if "Too Many Requests" in str(e) and attempt < 2:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Retrying {ticker} after {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return None

@sleep_and_retry
@limits(calls=5, period=60)  # Limit NewsAPI calls to 5 per minute
def get_analyst_sentiment(ticker):
    """Fetch analyst sentiment from yfinance."""
    print(f"Fetching analyst sentiment for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        recommendation = stock.info.get('recommendationMean', 3.0)
        sentiment = (3.0 - recommendation) / 2.0
        print(f"Analyst sentiment for {ticker}: {sentiment:.2f} (Recommendation: {recommendation})")
        return sentiment
    except Exception as e:
        print(f"Analyst sentiment error for {ticker}: {e}")
        return 0

def get_news_sentiment(ticker):
    """Fetch news sentiment from NewsAPI as fallback."""
    print(f"Fetching news for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', '').split()[0] or ticker
        query = f"{ticker} OR {company_name}"
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"NewsAPI error for {ticker}: Status {response.status_code}, Response: {response.text}")
            return 0
        data = response.json()
        articles = data.get('articles', [])
        if not articles:
            print(f"No articles found for {ticker}")
            return 0
        sentiment_score = 0
        count = 0
        for article in articles[:5]:
            text = article.get('title', '') + ' ' + article.get('description', '')
            if text.strip():
                analysis = TextBlob(text)
                sentiment_score += analysis.sentiment.polarity
                count += 1
                print(f"Article for {ticker}: Polarity {analysis.sentiment.polarity:.2f}, Text: {text[:100]}...")
        if count == 0:
            print(f"No valid articles with text for {ticker}")
            return 0
        avg_sentiment = sentiment_score / count
        print(f"News sentiment for {ticker}: {avg_sentiment:.2f} based on {count} articles")
        return avg_sentiment
    except Exception as e:
        print(f"News error for {ticker}: {e}")
        return 0

def select_stocks():
    """Select top 5 stocks based on indicators and sentiment."""
    print("Selecting stocks...")
    stock_scores = []
    for ticker in STOCKS:
        try:
            data = get_stock_data(ticker)
            if not data:
                print(f"Skipping {ticker} due to missing data")
                continue
            sentiment = get_analyst_sentiment(ticker)
            if sentiment == 0:
                sentiment = get_news_sentiment(ticker)

            # Score based on indicators
            score = 0
            if 10 < data['rsi'] < 70:  # Relaxed momentum
                score += 30
            if data['macd'] > 0:  # Bullish MACD
                score += 20
            if data['macd_momentum']:  # MACD momentum
                score += 10
            if data['price'] < data['upper_band']:
                score += 20
            if data['volatility'] > 0.6:  # Relaxed volatility
                score += 30
            if data['pe_ratio'] != 0 and data['pe_ratio'] < 20:
                score += 10
            if data['avg_volume'] > 10000:  # Minimal liquidity
                score += 20
            if data['entry_price'] <= PENNY_PRICE:
                score += 20
            if abs(data['price'] - data['high_20d']) / data['high_20d'] < 0.10:  # Relaxed breakout
                score += 30
            if data['price_momentum'] > 0.0:  # Relaxed momentum
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
            print(f"Scored {ticker}: Score {score}")
        except Exception as e:
            print(f"Error in scoring {ticker}: {e}")
            continue

    # Sort and select top 5
    if len(stock_scores) < 5:
        print(f"Only {len(stock_scores)} stocks scored. Using fallback scoring to ensure 5 picks.")
        # Retry with minimal criteria
        for ticker in STOCKS[:100]:  # Increase to 100 for more candidates
            try:
                data = get_stock_data(ticker)
                if not data:
                    continue
                sentiment = get_analyst_sentiment(ticker)
                if sentiment == 0:
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
                print(f"Fallback scored {ticker}: Score {score}")
            except Exception as e:
                print(f"Error in fallback scoring {ticker}: {e}")
                continue

    if not stock_scores:
        print("No stocks scored even with fallback. Returning empty list.")
        return []

    sorted_stocks = sorted(stock_scores, key=lambda x: x['score'], reverse=True)[:5]
    print(f"Selected {len(sorted_stocks)} stocks")
    return sorted_stocks

@app.route('/')
def home():
    """Render the main page with stock picks."""
    print("Rendering homepage...")
    try:
        stocks = select_stocks()
        if not stocks:
            print("No stocks to display")
            return render_template('index.html', stocks=[], date=datetime.now().strftime("%Y-%m-%d"), error="No stocks available. Try again later.")
        return render_template('index.html', stocks=stocks, date=datetime.now().strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"Error rendering homepage: {e}")
        return render_template('index.html', stocks=[], date=datetime.now().strftime("%Y-%m-%d"), error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)