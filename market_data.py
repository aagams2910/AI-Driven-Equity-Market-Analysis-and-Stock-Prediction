import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Load API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSDATA_IO_API_KEY = os.getenv("NEWSDATA_IO_API_KEY")

def get_market_data(symbols: List[str], timeframe: str = "1d") -> Dict[str, Any]:
    """
    Fetch market data for specified symbols
    
    Args:
        symbols: List of stock symbols
        timeframe: Time period for historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with market data for each symbol
    """
    result = {}
    
    try:
        print(f"Fetching market data for symbols: {symbols}, timeframe: {timeframe}")
        # Convert string to list if needed
        if isinstance(symbols, str):
            symbols = [symbol.strip() for symbol in symbols.split(',')]
            
        for symbol in symbols:
            try:
                print(f"Processing symbol: {symbol}")
                # Get stock data using yfinance
                stock = yf.Ticker(symbol)
                
                # Map timeframe to yfinance format if needed
                yf_timeframe = timeframe
                if timeframe == '1m':
                    yf_timeframe = '1mo'
                elif timeframe == '3m':
                    yf_timeframe = '3mo'
                elif timeframe == '6m':
                    yf_timeframe = '6mo'
                
                # Get historical data
                hist = stock.history(period=yf_timeframe)
                print(f"Historical data for {symbol}: {len(hist)} records")
                
                if hist.empty:
                    print(f"No historical data available for {symbol}")
                    result[symbol] = {
                        "error": f"No data available for {symbol}"
                    }
                    continue
                
                # Get company info
                info = stock.info
                
                # Calculate basic technical indicators
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['RSI'] = calculate_rsi(hist['Close'])
                
                # Format the data
                historical_data = []
                print(f"Formatting historical data for {symbol}")
                for date, row in hist.iterrows():
                    historical_data.append({
                        "Date": date.strftime('%Y-%m-%d'),  # Ensure consistent capitalization with frontend
                        "Open": round(row['Open'], 2) if not pd.isna(row['Open']) else None,
                        "High": round(row['High'], 2) if not pd.isna(row['High']) else None,
                        "Low": round(row['Low'], 2) if not pd.isna(row['Low']) else None,
                        "Close": round(row['Close'], 2) if not pd.isna(row['Close']) else None,
                        "Volume": int(row['Volume']) if not pd.isna(row['Volume']) else None,
                        "SMA_20": round(row['SMA_20'], 2) if not pd.isna(row['SMA_20']) else None,
                        "SMA_50": round(row['SMA_50'], 2) if not pd.isna(row['SMA_50']) else None,
                        "RSI": round(row['RSI'], 2) if not pd.isna(row['RSI']) else None
                    })
                print(f"Formatted {len(historical_data)} records for {symbol}")
                print(f"Sample historical data for {symbol}: {historical_data[:1]}")
                
                # Get current price and calculate change
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[0]
                price_change = current_price - prev_price
                price_change_percent = (price_change / prev_price) * 100 if prev_price > 0 else 0
                
                # Compile the result
                result[symbol] = {
                    "symbol": symbol,
                    "company_name": info.get('shortName', symbol),
                    "current_price": round(current_price, 2),
                    "price_change": round(price_change, 2),
                    "price_change_percent": round(price_change_percent, 2),
                    "market_cap": info.get('marketCap'),
                    "pe_ratio": info.get('trailingPE'),
                    "dividend_yield": info.get('dividendYield'),
                    "volume": info.get('volume'),
                    "avg_volume": info.get('averageVolume'),
                    "high_52week": info.get('fiftyTwoWeekHigh'),
                    "low_52week": info.get('fiftyTwoWeekLow'),
                    "historical_data": historical_data
                }
                
            except Exception as e:
                result[symbol] = {
                    "error": f"Error fetching data for {symbol}: {str(e)}"
                }
                
    except Exception as e:
        return {
            "error": f"Error processing market data: {str(e)}"
        }
        
    # Return the result directly without nesting it in a 'data' key
    return result

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_financial_news(symbols: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch financial news using NewsData.io API
    
    Args:
        symbols: Stock symbols to get news for (comma-separated string)
        limit: Maximum number of news items to return
        
    Returns:
        List of news articles
    """
    try:
        # Base URL for NewsData.io API
        base_url = "https://newsdata.io/api/1/news"
        
        # Prepare query parameters
        params = {
            "apikey": NEWSDATA_IO_API_KEY,
            "language": "en",
            "category": "business",
            "size": limit
        }
        
        # Add symbols to query if provided
        if symbols:
            # Convert string to list if needed
            if isinstance(symbols, str):
                symbol_list = [symbol.strip() for symbol in symbols.split(',')]
            else:
                symbol_list = symbols
                
            # Get company names for better search results
            company_names = []
            for symbol in symbol_list:
                try:
                    stock = yf.Ticker(symbol)
                    company_name = stock.info.get('shortName')
                    if company_name:
                        company_names.append(company_name)
                except:
                    # If we can't get the company name, use the symbol
                    company_names.append(symbol)
            
            # Join company names for the query
            if company_names:
                params["q"] = " OR ".join(company_names)
        
        # Make the API request
        response = requests.get(base_url, params=params)
        data = response.json()
        
        # Process the results
        news_articles = []
        if data.get("status") == "success" and "results" in data:
            for article in data["results"]:
                # Format the article data
                news_item = {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "url": article.get("link"),
                    "image_url": article.get("image_url"),
                    "source": article.get("source_id"),
                    "published_at": article.get("pubDate"),
                    "category": "business",
                    "sentiment": analyze_sentiment(article.get("title", "") + " " + article.get("description", ""))
                }
                news_articles.append(news_item)
                
        return {
            "data": news_articles,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Error fetching news: {str(e)}",
            "data": []
        }

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Simple sentiment analysis for news headlines
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # This is a simple implementation - in a real app, you'd use a more sophisticated model
    positive_words = [
        'gain', 'gains', 'up', 'rise', 'rises', 'rising', 'rose', 'positive', 'profit', 'profits',
        'growth', 'growing', 'grew', 'increase', 'increases', 'increasing', 'increased', 'higher',
        'surge', 'surges', 'surging', 'surged', 'rally', 'rallies', 'rallying', 'rallied',
        'strong', 'stronger', 'strongest', 'strength', 'opportunity', 'opportunities', 'success', 'successful'
    ]
    
    negative_words = [
        'loss', 'losses', 'down', 'fall', 'falls', 'falling', 'fell', 'negative', 'deficit',
        'decline', 'declines', 'declining', 'declined', 'decrease', 'decreases', 'decreasing', 'decreased',
        'lower', 'drop', 'drops', 'dropping', 'dropped', 'plunge', 'plunges', 'plunging', 'plunged',
        'weak', 'weaker', 'weakest', 'weakness', 'risk', 'risks', 'risky', 'danger', 'dangerous',
        'fail', 'fails', 'failing', 'failed', 'failure', 'concern', 'concerns', 'concerning', 'concerned'
    ]
    
    neutral_words = [
        'announce', 'announces', 'announcing', 'announced', 'announcement',
        'report', 'reports', 'reporting', 'reported', 'say', 'says', 'saying', 'said',
        'plan', 'plans', 'planning', 'planned', 'expect', 'expects', 'expecting', 'expected',
        'launch', 'launches', 'launching', 'launched', 'introduce', 'introduces', 'introducing', 'introduced'
    ]
    
    text = text.lower()
    
    # Count occurrences of sentiment words
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    neutral_count = sum(1 for word in neutral_words if word in text)
    
    # Calculate total and percentages
    total = positive_count + negative_count + neutral_count
    if total == 0:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    return {
        "positive": round(positive_count / total, 2),
        "negative": round(negative_count / total, 2),
        "neutral": round(neutral_count / total, 2)
    } 