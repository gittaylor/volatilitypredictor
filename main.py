from stock_data import get_historical_data, prepare_features
from price_predictor import PriceRangePredictor
from option_analyzer import OptionAnalyzer
import pandas as pd
import os

def predict_for_symbol(symbol, forecast_period):
    """
    Make predictions for a single symbol and analyze options if available
    """
    # Get historical data
    df = get_historical_data(symbol)
    if df is None:
        return None
    
    # Prepare features
    features = prepare_features(df)
    
    # Create and train predictor
    predictor = PriceRangePredictor()
    predictor.train(features, df, forecast_period)
    
    # Make prediction
    high_pred, low_pred = predictor.predict(features)
    
    # Get current price
    current_price = df['Close'].iloc[-1]
    
    # Check for options data and analyze if available
    options_file = f'{symbol} Option Data-dEgQw.csv'
    options_analysis = None
    if os.path.exists(options_file):
        analyzer = OptionAnalyzer()
        options_analysis = analyzer.analyze_options(
            filename=options_file,
            current_price=current_price,
            predicted_max=high_pred,
            predicted_min=low_pred,
            prediction_interval=forecast_period
        )
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_high': high_pred,
        'predicted_low': low_pred,
        'high_change%': ((high_pred - current_price) / current_price) * 100,
        'low_change%': ((low_pred - current_price) / current_price) * 100,
        'options_analyzed': options_analysis is not None,
        'potential_options': len(options_analysis) if options_analysis is not None else 0
    }

def main():
    # Set parameters
    symbols = [
        'NVDA', 'MSFT', 'AAPL', 'AMZN', 'TSLA', 
        'META', 'GOOG', 'ORCL', 'SMH', 'AIQ', 
        'VOO', 'QQQM', 'SWPPX'
    ]
    forecast_period = 30  # Changed to 30 days to match options analysis
    
    # Store results
    results = []
    
    # Make predictions for each symbol
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        result = predict_for_symbol(symbol, forecast_period)
        if result:
            results.append(result)
    
    # Create DataFrame for better display
    df_results = pd.DataFrame(results)
    
    # Format the results
    for col in df_results.columns:
        if 'price' in col.lower() or 'predicted' in col.lower():
            df_results[col] = df_results[col].apply(lambda x: f'${x:,.2f}')
        elif '%' in col:
            df_results[col] = df_results[col].apply(lambda x: f'{x:.2f}%')
    
    print(f"\nPredictions for the next {forecast_period} days:")
    print("\n", df_results)
    
    # Save results to CSV
    df_results.to_csv('stock_predictions.csv', index=False)
    print("\nResults have been saved to 'stock_predictions.csv'")
    
    # Print options analysis summary for stocks with options data
    for symbol in symbols:
        options_file = f'selected_options_{symbol}.csv'
        if os.path.exists(options_file):
            print(f"\nPotential options trades for {symbol}:")
            options_df = pd.read_csv(options_file)
            print(f"Found {len(options_df)} potential options trades:")
            print("\nCALL Options:")
            calls = options_df[options_df['Option Type'] == 'CALL']
            if len(calls) > 0:
                print(calls[['Strike Price', 'Ask', 'Expiration Date']].to_string())
            else:
                print("No suitable CALL options found")
                
            print("\nPUT Options:")
            puts = options_df[options_df['Option Type'] == 'PUT']
            if len(puts) > 0:
                print(puts[['Strike Price', 'Ask', 'Expiration Date']].to_string())
            else:
                print("No suitable PUT options found")

if __name__ == "__main__":
    main() 