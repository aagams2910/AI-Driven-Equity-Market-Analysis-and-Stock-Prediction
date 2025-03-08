import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import io
import base64
import google.generativeai as genai
from market_data import get_market_data, get_financial_news

# Load API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Define a simple neural network for time series prediction
class StockPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(StockPredictionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def prepare_stock_data(symbol: str, timeframe: str = "1y") -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare stock data for prediction models
    
    Args:
        symbol: Stock symbol
        timeframe: Time period for data
        
    Returns:
        Tuple of (processed_dataframe, feature_names)
    """
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=timeframe)
        
        if hist.empty:
            raise ValueError(f"No historical data available for {symbol}")
        
        # Calculate technical indicators
        # Moving averages
        hist['MA5'] = hist['Close'].rolling(window=5).mean()
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        hist['20d_std'] = hist['Close'].rolling(window=20).std()
        hist['Upper_Band'] = hist['MA20'] + (hist['20d_std'] * 2)
        hist['Lower_Band'] = hist['MA20'] - (hist['20d_std'] * 2)
        
        # Average True Range (ATR)
        hist['H-L'] = hist['High'] - hist['Low']
        hist['H-PC'] = abs(hist['High'] - hist['Close'].shift(1))
        hist['L-PC'] = abs(hist['Low'] - hist['Close'].shift(1))
        hist['TR'] = hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        hist['ATR'] = hist['TR'].rolling(window=14).mean()
        
        # Volume indicators
        hist['Volume_Change'] = hist['Volume'].pct_change()
        hist['Volume_MA10'] = hist['Volume'].rolling(window=10).mean()
        
        # Price momentum
        hist['Price_Change'] = hist['Close'].pct_change()
        hist['Price_Change_5d'] = hist['Close'].pct_change(periods=5)
        
        # Target variable: Next day's closing price
        hist['Target'] = hist['Close'].shift(-1)
        
        # Drop rows with NaN values
        hist = hist.dropna()
        
        # Select features for prediction
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line',
            'Upper_Band', 'Lower_Band', 'ATR',
            'Volume_Change', 'Volume_MA10',
            'Price_Change', 'Price_Change_5d'
        ]
        
        return hist[features + ['Target']], features
    except Exception as e:
        print(f"Error preparing data for {symbol}: {str(e)}")
        raise ValueError(f"Failed to prepare data for {symbol}: {str(e)}")

def train_prediction_models(data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """
    Train multiple prediction models on the prepared data
    
    Args:
        data: Prepared dataframe with features and target
        features: List of feature names
        
    Returns:
        Dict of trained models and their performance metrics
    """
    # Prepare data
    X = data[features].values
    y = data['Target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Neural Network
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    input_size = X_train_scaled.shape[1]
    nn_model = StockPredictionNN(input_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    
    # Train the model
    nn_model.train()
    epochs = 100
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Make predictions
    nn_model.eval()
    with torch.no_grad():
        nn_pred = nn_model(X_test_tensor).numpy().flatten()
    
    # Ensemble prediction (average of all models)
    ensemble_pred = (lr_pred + rf_pred + gb_pred + nn_pred) / 4
    
    # Calculate metrics
    models = {
        "linear_regression": {
            "model": lr_model,
            "predictions": lr_pred,
            "mse": mean_squared_error(y_test, lr_pred),
            "mae": mean_absolute_error(y_test, lr_pred),
            "r2": r2_score(y_test, lr_pred)
        },
        "random_forest": {
            "model": rf_model,
            "predictions": rf_pred,
            "mse": mean_squared_error(y_test, rf_pred),
            "mae": mean_absolute_error(y_test, rf_pred),
            "r2": r2_score(y_test, rf_pred)
        },
        "gradient_boosting": {
            "model": gb_model,
            "predictions": gb_pred,
            "mse": mean_squared_error(y_test, gb_pred),
            "mae": mean_absolute_error(y_test, gb_pred),
            "r2": r2_score(y_test, gb_pred)
        },
        "neural_network": {
            "model": nn_model,
            "predictions": nn_pred,
            "mse": mean_squared_error(y_test, nn_pred),
            "mae": mean_absolute_error(y_test, nn_pred),
            "r2": r2_score(y_test, nn_pred)
        },
        "ensemble": {
            "predictions": ensemble_pred,
            "mse": mean_squared_error(y_test, ensemble_pred),
            "mae": mean_absolute_error(y_test, ensemble_pred),
            "r2": r2_score(y_test, ensemble_pred)
        }
    }
    
    # Add test data for visualization
    models["test_data"] = {
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "features": features
    }
    
    return models

def generate_prediction_chart(symbol: str, models: Dict[str, Any], data: pd.DataFrame) -> str:
    """
    Generate a chart showing actual vs predicted prices
    
    Args:
        symbol: Stock symbol
        models: Trained models and predictions
        data: Original data used for training
        
    Returns:
        Base64 encoded image of the chart
    """
    # Get test data
    y_test = models["test_data"]["y_test"]
    
    # Get predictions from each model
    lr_pred = models["linear_regression"]["predictions"]
    rf_pred = models["random_forest"]["predictions"]
    gb_pred = models["gradient_boosting"]["predictions"]
    nn_pred = models["neural_network"]["predictions"]
    ensemble_pred = models["ensemble"]["predictions"]
    
    # Create a date range for the test data
    test_dates = data.index[-len(y_test):]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Actual', linewidth=2)
    plt.plot(test_dates, ensemble_pred, label='Ensemble Prediction', linewidth=2, linestyle='--')
    plt.plot(test_dates, lr_pred, label='Linear Regression', alpha=0.5)
    plt.plot(test_dates, rf_pred, label='Random Forest', alpha=0.5)
    plt.plot(test_dates, gb_pred, label='Gradient Boosting', alpha=0.5)
    plt.plot(test_dates, nn_pred, label='Neural Network', alpha=0.5)
    
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Convert plot to base64 encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64
    encoded_image = base64.b64encode(image_png).decode('utf-8')
    
    return encoded_image

def generate_stock_recommendations(symbols: List[str], timeframe: str = "1m") -> Dict[str, Any]:
    """
    Generate stock recommendations based on predictive models
    
    Args:
        symbols: List of stock symbols
        timeframe: Time period for analysis
        
    Returns:
        Dict containing recommendations for each symbol
    """
    recommendations = {}
    
    # Convert string to list if needed
    if isinstance(symbols, str):
        symbols = [symbol.strip() for symbol in symbols.split(',')]
    
    for symbol in symbols:
        try:
            print(f"Processing symbol: {symbol}")
            
            # Get stock data using yfinance
            stock = yf.Ticker(symbol)
            
            # Get historical data
            data = stock.history(period="1y")
            
            if data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Calculate technical indicators
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # Get the latest data point
            latest_data = data.iloc[-1]
            latest_close = latest_data['Close']
            
            # Simple prediction based on recent trend
            recent_trend = (latest_data['Close'] - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100
            
            # Determine recommendation based on recent trend
            if recent_trend > 5:
                recommendation = "Strong Buy"
                confidence = 85
            elif recent_trend > 2:
                recommendation = "Buy"
                confidence = 70
            elif recent_trend < -5:
                recommendation = "Strong Sell"
                confidence = 85
            elif recent_trend < -2:
                recommendation = "Sell"
                confidence = 70
            else:
                recommendation = "Hold"
                confidence = 60
            
            # Get company info
            info = stock.info
            company_name = info.get('shortName', symbol)
            
            # Compile recommendation
            recommendations[symbol] = {
                "symbol": symbol,
                "company_name": company_name,
                "current_price": latest_close,
                "predicted_change_percent": recent_trend,
                "recommendation": recommendation,
                "confidence": confidence,
                "market_data": {
                    "pe_ratio": info.get("trailingPE", None),
                    "market_cap": info.get("marketCap", None),
                    "sector": info.get("sector", None),
                    "industry": info.get("industry", None)
                },
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error generating recommendation for {symbol}: {str(e)}")
            recommendations[symbol] = {
                "symbol": symbol,
                "error": str(e),
                "recommendation": "Unable to generate recommendation",
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    return recommendations

def simulate_trade(symbol: str, action: str, amount: float, price: float, timeframe: str = "1m") -> Dict[str, Any]:
    """
    Simulate a trade and provide analysis of the potential outcome
    
    Args:
        symbol: Stock symbol
        action: 'buy' or 'sell'
        price: Price per share
        amount: Number of shares
        timeframe: Time period for simulation
        
    Returns:
        Dict containing trade simulation results
    """
    # Validate inputs
    if action.lower() not in ['buy', 'sell']:
        raise ValueError("Action must be either 'buy' or 'sell'")
    
    if amount <= 0 or price <= 0:
        raise ValueError("Amount and price must be positive")
    
    # Get stock data
    stock_data = get_market_data([symbol], timeframe)[symbol]
    
    # Get stock recommendations
    recommendations = generate_stock_recommendations([symbol], timeframe)[symbol]
    
    # Calculate trade value
    trade_value = amount * price
    
    # Get current market price
    current_price = stock_data.get("current_price", price)
    
    # Calculate market value
    market_value = amount * current_price
    
    # Calculate potential profit/loss based on predicted change percentage
    # The recommendations contain 'predicted_change_percent' but not 'predicted_price'
    predicted_change_percent = recommendations.get("predicted_change_percent", 0)
    predicted_price = current_price * (1 + predicted_change_percent / 100)
    
    if action.lower() == 'buy':
        potential_profit = amount * (predicted_price - price)
        roi = (predicted_price - price) / price * 100
    else:  # sell
        potential_profit = amount * (price - predicted_price)
        roi = (price - predicted_price) / price * 100
    
    # Determine if the trade aligns with recommendation
    recommendation = recommendations["recommendation"]
    
    if action.lower() == 'buy' and recommendation in ["Buy", "Strong Buy"]:
        alignment = "Aligned with recommendation"
        alignment_score = 1.0
    elif action.lower() == 'sell' and recommendation in ["Sell", "Strong Sell"]:
        alignment = "Aligned with recommendation"
        alignment_score = 1.0
    elif recommendation == "Hold":
        alignment = "Neutral - recommendation is Hold"
        alignment_score = 0.5
    else:
        alignment = "Contradicts recommendation"
        alignment_score = 0.0
    
    # Calculate risk score (0-100)
    # Higher score means higher risk
    volatility = np.std([d.get("Close", 0) for d in stock_data.get("historical_data", [])[-30:]])
    avg_price = np.mean([d.get("Close", 0) for d in stock_data.get("historical_data", [])[-30:]])
    volatility_pct = (volatility / avg_price) * 100
    
    risk_score = min(volatility_pct * 10, 100)
    
    # Adjust risk score based on trade size and alignment
    risk_score = risk_score * (1 - alignment_score * 0.3)
    
    # Generate simulation result
    result = {
        "symbol": symbol,
        "action": action,
        "amount": amount,
        "price": price,
        "trade_value": trade_value,
        "current_market_price": current_price,
        "current_market_value": market_value,
        "predicted_price": predicted_price,
        "potential_profit_loss": potential_profit,
        "roi_percent": roi,
        "recommendation": recommendation,
        "recommendation_alignment": alignment,
        "risk_score": risk_score,
        "confidence": recommendations["confidence"],
        "simulation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result 
