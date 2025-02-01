from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class PriceRangePredictor:
    def __init__(self):
        self.high_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.low_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_target(self, df, forecast_period):
        """
        Prepare target variables (future high and low prices)
        """
        # Calculate future high and low prices for the forecast period
        df['Future_High'] = df['High'].rolling(window=forecast_period).max().shift(-forecast_period)
        df['Future_Low'] = df['Low'].rolling(window=forecast_period).min().shift(-forecast_period)
        
        return df
        
    def train(self, features, df, forecast_period):
        """
        Train the prediction models
        """
        # Prepare target variables
        df = self.prepare_target(df, forecast_period)
        
        # Remove rows with NaN values
        valid_idx = df['Future_High'].notna()
        X = features[valid_idx]
        y_high = df['Future_High'][valid_idx]
        y_low = df['Future_Low'][valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.high_model.fit(X_scaled, y_high)
        self.low_model.fit(X_scaled, y_low)
        
    def predict(self, features):
        """
        Predict future price range
        """
        X_scaled = self.scaler.transform(features)
        
        high_pred = self.high_model.predict(X_scaled)
        low_pred = self.low_model.predict(X_scaled)
        
        return high_pred[-1], low_pred[-1] 