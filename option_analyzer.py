import csv
from datetime import datetime
import pandas as pd

class OptionAnalyzer:
    def __init__(self):
        self.current_price = None
        self.predicted_max = None
        self.predicted_min = None
        
    def analyze_options(self, filename, current_price, predicted_max, predicted_min, prediction_interval):
        """
        Analyze options based on price predictions
        
        Args:
            filename (str): Path to options data file
            current_price (float): Current stock price
            predicted_max (float): Predicted maximum price
            predicted_min (float): Predicted minimum price
            prediction_interval (int): Number of days to analyze (from main.py)
        """
        self.current_price = current_price
        self.predicted_max = predicted_max
        self.predicted_min = predicted_min
        
        # Read CSV data
        df = pd.read_csv(filename)
        
        # Convert date columns to datetime
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
        
        today = datetime.now()
        
        # Filter for options within prediction interval from main.py
        df['days_to_expiry'] = (df['Expiration Date'] - pd.Timestamp(today)).dt.days
        df = df[df['days_to_expiry'] <= prediction_interval]
        
        # Analyze CALLS
        potential_calls = df[
            (df['Option Type'] == 'CALL') & 
            (df['Strike Price'] + 2 * df['Ask'] < predicted_max)
        ]
        
        # Analyze PUTS
        potential_puts = df[
            (df['Option Type'] == 'PUT') & 
            (df['Strike Price'] - 2 * df['Ask'] > predicted_min)
        ]
        
        # Combine results
        selected_options = pd.concat([potential_calls, potential_puts])
        
        # Add prediction metrics
        selected_options['Predicted Max'] = predicted_max
        selected_options['Predicted Min'] = predicted_min
        selected_options['Current Price'] = current_price
        selected_options['Analysis Period'] = prediction_interval
        
        # Save results with symbol in filename
        symbol = df['Symbol'].iloc[0] if not df.empty else 'UNKNOWN'
        output_filename = f'selected_options_{symbol}.csv'
        selected_options.to_csv(output_filename, index=False)
        
        return selected_options 