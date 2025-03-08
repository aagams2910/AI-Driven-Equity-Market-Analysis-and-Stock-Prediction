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
        print(f"Fetching financial news for symbols: {symbols}, limit: {limit}")
        
        # If no symbols provided, use some default popular stocks
        if not symbols:
            symbols = "AAPL,MSFT,GOOGL,AMZN,TSLA"
            print(f"No symbols provided, using defaults: {symbols}")
        
        # Check if API key is available
        if not NEWSDATA_IO_API_KEY:
            print("ERROR: NEWSDATA_IO_API_KEY is not set in environment variables")
            return [{
                "title": "API Key Not Configured",
                "summary": "The NewsData.io API key is not configured. Please set the NEWSDATA_IO_API_KEY environment variable.",
                "content": "Please contact the administrator to set up the API key for news data.",
                "link": "https://newsdata.io",
                "publisher": "System",
                "published_date": datetime.now().isoformat(),
                "related_symbols": symbols.split(',') if isinstance(symbols, str) else symbols,
                "sentiment": {"positive": 0, "negative": 0, "neutral": 1}
            }]
            
        # Base URL for NewsData.io API
        base_url = "https://newsdata.io/api/1/news"
        
        params = {
            "apikey": NEWSDATA_IO_API_KEY,
            "language": "en",
            "category": "business",
            "size": min(limit * 2, 50)  # Request more articles but stay within API limits
        }
        
        # Add symbols to query if provided
        if symbols:
            # Convert string to list if needed
            if isinstance(symbols, str):
                symbol_list = [symbol.strip() for symbol in symbols.split(',')]
            else:
                symbol_list = symbols
                
            # Get company names and build search query
            search_terms = []
            company_info = {}
            
            for symbol in symbol_list:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    company_name = info.get('shortName') or info.get('longName')
                    if company_name:
                        # Store company info for later filtering
                        company_info[symbol] = {
                            'name': company_name,
                            'industry': info.get('industry'),
                            'sector': info.get('sector'),
                            'keywords': set([
                                symbol,
                                company_name,
                                info.get('industry', ''),
                                info.get('sector', '')
                            ])
                        }
                        # Add both symbol and company name to search terms
                        # Don't use quotes as they might cause API issues
                        search_terms.append(symbol)
                        search_terms.append(company_name)
                except Exception as e:
                    print(f"Error fetching company info for {symbol}: {str(e)}")
                    search_terms.append(symbol)
            
            # Join search terms with OR operator
            if search_terms:
                # Limit query length to avoid 422 errors
                query = " OR ".join(search_terms)
                if len(query) > 500:  # API might have limits on query length
                    # Simplify to just use symbols if query is too long
                    query = " OR ".join(symbol_list)
                params["q"] = query
                print(f"Using query: {params['q']}")
        
        # Make the API request
        print(f"Making request to {base_url} with params: {params}")
        response = requests.get(base_url, params=params)
        print(f"Response status code: {response.status_code}")
        
        # Check if response is valid
        if response.status_code != 200:
            print(f"Error response from NewsData.io API: {response.text}")
            
            # If we get a 422 error, try a simpler query with just the symbols
            if response.status_code == 422 and symbols:
                print("Retrying with simplified query...")
                if isinstance(symbols, str):
                    symbol_list = [symbol.strip() for symbol in symbols.split(',')]
                else:
                    symbol_list = symbols
                    
                # Use a simpler query with just the symbols
                params["q"] = " OR ".join(symbol_list)
                print(f"Using simplified query: {params['q']}")
                
                response = requests.get(base_url, params=params)
                print(f"Retry response status code: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"Retry also failed: {response.text}")
                    # Return fallback data
                    return get_fallback_news(symbol_list)
            else:
                # Return fallback data
                return get_fallback_news(symbol_list if 'symbol_list' in locals() else 
                                        symbols.split(',') if isinstance(symbols, str) else 
                                        symbols if symbols else ["AAPL", "MSFT", "GOOGL"])
            
        data = response.json()
        print(f"NewsData.io API response: {data.get('status')}, results count: {len(data.get('results', []))}")
        
        # Process and filter the results
        news_articles = []
        if data.get("status") == "success" and "results" in data:
            for article in data["results"]:
                # Skip articles without title or content
                if not article.get("title"):
                    continue
                
                # Calculate relevance score and determine related symbols
                article_text = (
                    (article.get("title") or "").lower() + " " +
                    (article.get("description") or "").lower() + " " +
                    (article.get("content") or "").lower()
                )
                
                related_symbols = []
                relevance_score = 0
                
                for symbol, info in company_info.items():
                    symbol_relevance = 0
                    for keyword in info['keywords']:
                        if not keyword:
                            continue
                        keyword = keyword.lower()
                        # Count occurrences of each keyword
                        symbol_relevance += article_text.count(keyword.lower()) * (
                            2 if keyword == symbol.lower() else  # Higher weight for exact symbol match
                            1.5 if keyword == info['name'].lower() else  # Medium weight for company name
                            0.5  # Lower weight for industry/sector
                        )
                    
                    if symbol_relevance > 0:
                        related_symbols.append(symbol)
                        relevance_score += symbol_relevance
                
                # If no specific relevance found but we have symbols, assign them all
                if relevance_score == 0 and symbol_list:
                    related_symbols = symbol_list
                    relevance_score = 0.1  # Low but non-zero score
                
                # Create the news item
                news_item = {
                    "title": article.get("title"),
                    "summary": article.get("description") or "No description available",
                    "content": article.get("content") or article.get("description") or "No content available",
                    "link": article.get("link"),
                    "thumbnail": article.get("image_url"),
                    "publisher": article.get("source_id"),
                    "published_date": article.get("pubDate"),
                    "category": "business",
                    "related_symbols": related_symbols,
                    "relevance_score": relevance_score,
                    "sentiment": analyze_sentiment(article.get("title", "") + " " + (article.get("description") or ""))
                }
                news_articles.append(news_item)
        
        # Sort by relevance score and limit results
        news_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        news_articles = news_articles[:limit]
        
        print(f"Processed {len(news_articles)} relevant news articles")
        
        # If no news found, return fallback data
        if not news_articles:
            print("No news articles found, returning fallback data")
            return get_fallback_news(symbol_list if 'symbol_list' in locals() else 
                                   symbols.split(',') if isinstance(symbols, str) else 
                                   symbols if symbols else ["AAPL", "MSFT", "GOOGL"])
        
        return news_articles
        
    except Exception as e:
        print(f"Error in get_financial_news: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return fallback data in case of error
        return get_fallback_news(symbols.split(',') if isinstance(symbols, str) else symbols if symbols else ["AAPL", "MSFT", "GOOGL"])

def get_fallback_news(symbols):
    """
    Generate fallback news data when the API fails
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        List of fallback news articles
    """
    print(f"Generating fallback news for symbols: {symbols}")
    
    fallback_news = []
    current_date = datetime.now()
    
    # News types to generate for variety
    news_types = [
        {
            "type": "earnings",
            "title_template": "{company} ({symbol}) Reports Quarterly Earnings",
            "summary_template": "Financial results and performance analysis for {company} in the {industry} sector.",
            "content_template": "{company} ({symbol}) has reported its quarterly earnings. This system-generated article provides a placeholder for actual earnings news. In real market conditions, this would contain details about revenue, EPS, growth metrics, and management commentary.",
            "image": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "sentiment": {"positive": 0.5, "negative": 0.2, "neutral": 0.3}
        },
        {
            "type": "product",
            "title_template": "{company} Announces New Product Developments",
            "summary_template": "Latest innovations and product launches from {company} that could impact its market position.",
            "content_template": "{company} ({symbol}) has announced new product developments that could significantly impact its market position. This system-generated article serves as a placeholder for actual product news. In real market conditions, this would contain details about product features, market potential, and competitive advantages.",
            "image": "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "sentiment": {"positive": 0.6, "negative": 0.1, "neutral": 0.3}
        },
        {
            "type": "market_analysis",
            "title_template": "Market Analysis: {company} ({symbol}) Recent Trends",
            "summary_template": "Analysis of recent market trends for {company} in the {industry} industry.",
            "content_template": "This market analysis examines recent trends for {company} ({symbol}), a company in the {sector} sector, specifically in the {industry} industry. This system-generated article serves as a placeholder for actual market analysis. In real market conditions, this would contain details about price movements, trading volumes, and technical indicators.",
            "image": "https://images.unsplash.com/photo-1535320903710-d993d3d77d29?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        },
        {
            "type": "industry",
            "title_template": "Industry Overview: Impact on {symbol} and Peers",
            "summary_template": "Overview of the {industry} industry and its impact on {company} and its competitors.",
            "content_template": "This industry overview examines the {industry} sector, focusing on {company} ({symbol}) and its peers. This system-generated article serves as a placeholder for actual industry news. In real market conditions, this would contain details about industry trends, regulatory changes, and competitive dynamics.",
            "image": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "sentiment": {"positive": 0.4, "negative": 0.2, "neutral": 0.4}
        },
        {
            "type": "analyst",
            "title_template": "Analyst Ratings Update: {symbol} Stock Outlook",
            "summary_template": "Recent analyst ratings and price targets for {company} stock.",
            "content_template": "Financial analysts have updated their ratings for {company} ({symbol}). This system-generated article serves as a placeholder for actual analyst coverage. In real market conditions, this would contain details about price targets, buy/sell recommendations, and analyst commentary on future prospects.",
            "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "sentiment": {"positive": 0.45, "negative": 0.25, "neutral": 0.3}
        },
        {
            "type": "partnership",
            "title_template": "{company} Forms Strategic Partnership to Expand Market Reach",
            "summary_template": "{company} announces a new strategic partnership that could strengthen its position in the {industry} market.",
            "content_template": "{company} ({symbol}) has formed a strategic partnership to expand its market reach. This system-generated article serves as a placeholder for actual partnership news. In real market conditions, this would contain details about the partnership terms, strategic benefits, and market implications.",
            "image": "https://images.unsplash.com/photo-1521791136064-7986c2920216?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "sentiment": {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
        }
    ]
    
    # Create a fallback news item for each symbol
    for symbol in symbols:
        try:
            # Try to get company info
            stock = yf.Ticker(symbol)
            company_name = stock.info.get('shortName') or stock.info.get('longName') or symbol
            sector = stock.info.get('sector') or "Technology"
            industry = stock.info.get('industry') or "General"
            
            # Generate different types of news for each symbol
            for i, news_type in enumerate(news_types):
                # Adjust the date to spread out the news
                news_date = current_date - timedelta(days=i % 3)
                
                # Format the templates with company info
                title = news_type["title_template"].format(
                    company=company_name, 
                    symbol=symbol,
                    sector=sector,
                    industry=industry
                )
                
                summary = news_type["summary_template"].format(
                    company=company_name, 
                    symbol=symbol,
                    sector=sector,
                    industry=industry
                )
                
                content = news_type["content_template"].format(
                    company=company_name, 
                    symbol=symbol,
                    sector=sector,
                    industry=industry
                )
                
                # Create a fallback news item
                fallback_news.append({
                    "title": title,
                    "summary": summary,
                    "content": content,
                    "link": f"https://finance.yahoo.com/quote/{symbol}/news",
                    "thumbnail": news_type["image"],
                    "publisher": "Financial Insights",
                    "published_date": news_date.isoformat(),
                    "category": "business",
                    "related_symbols": [symbol],
                    "relevance_score": 1.0 - (i * 0.1),  # Decrease relevance for variety
                    "sentiment": news_type["sentiment"],
                    "is_fallback": True  # Flag to indicate this is fallback content
                })
                
        except Exception as e:
            print(f"Error generating fallback news for {symbol}: {str(e)}")
            # Create a generic fallback item if we can't get company info
            fallback_news.append({
                "title": f"Market Update: {symbol} Stock Analysis",
                "summary": f"Latest market analysis for {symbol} stock.",
                "content": f"This is a system-generated market update for {symbol}. The actual news data could not be retrieved at this time. Please try again later or check other sources for the latest news on {symbol}.",
                "link": f"https://finance.yahoo.com/quote/{symbol}",
                "thumbnail": "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
                "publisher": "Financial Insights",
                "published_date": current_date.isoformat(),
                "category": "business",
                "related_symbols": [symbol],
                "relevance_score": 1.0,
                "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "is_fallback": True
            })
    
    # Shuffle the news to mix different types
    import random
    random.shuffle(fallback_news)
    
    # Limit the number of fallback items
    return fallback_news[:min(len(fallback_news), 15)]

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
