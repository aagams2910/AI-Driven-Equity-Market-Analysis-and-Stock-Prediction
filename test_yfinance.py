import yfinance as yf
import pandas as pd
from datetime import datetime

def test_yfinance():
    print("Testing yfinance library...")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    for symbol in symbols:
        try:
            print(f"\nTesting symbol: {symbol}")
            
            # Get stock data
            stock = yf.Ticker(symbol)
            
            # Get company info
            info = stock.info
            print(f"Company name: {info.get('shortName', 'N/A')}")
            
            # Get historical data
            hist = stock.history(period="1y")
            
            if hist.empty:
                print(f"No historical data available for {symbol}")
            else:
                print(f"Historical data available: {len(hist)} rows")
                print(f"Latest close price: {hist['Close'].iloc[-1]}")
                
                # Calculate a simple moving average
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                print(f"Latest 20-day moving average: {hist['MA20'].iloc[-1]}")
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

if __name__ == "__main__":
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    test_yfinance()
    print("\nTest completed.") 