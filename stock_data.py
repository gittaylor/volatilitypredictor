import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_historical_data(symbol, period='2y', interval='1h'):
    """
    Fetch historical stock data using yfinance
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Time period to fetch (e.g., '2y' for 2 years)
        interval (str): Data interval (e.g., '1h' for hourly)
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_features(df):
    """
    Prepare features for the prediction model
    """
    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Calculate price volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Create features DataFrame
    features = df[['SMA_20', 'SMA_50', 'RSI', 'Volatility', 'Volume']].copy()
    
    # Forward fill any NaN values
    features = features.fillna(method='ffill')
    
    return features

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi 