import talib as ta
import pandas as pd
import logging
import os


# Logging configuration
logging.basicConfig(filename=os.path.join('logs', 'technical_indicators.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Apply technical indicators using TA-Lib
def apply_indicators(df):
    # Copy of the DataFrame to avoid modifying the original one
    df = df.copy()

    if 'Date' not in df.columns:
        logging.error("The 'Date' column is missing from the DataFrame. Aborting.")
        return df
    
    # Ensure 'Date' column is of datetime type
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        logging.error(f"Error converting 'Date' column to datetime: {str(e)}. Aborting.")
        return df

    # Temporarily set 'Date' as the index for TA-Lib operations
    df.set_index('Date', inplace=True)
    
    # Moving Averages (SMA & EMA)
    df['MA_7'] = ta.SMA(df['Close'], timeperiod=7)
    df['MA_21'] = ta.SMA(df['Close'], timeperiod=21)
    df['SMA_50'] = ta.SMA(df['Close'], timeperiod=50)
    df['SMA_200'] = ta.SMA(df['Close'], timeperiod=200)
    df['EMA_12'] = ta.EMA(df['Close'], timeperiod=12)
    df['EMA_26'] = ta.EMA(df['Close'], timeperiod=26)

    # Relative Strength Index (RSI)
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)

    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Average True Range (ATR)
    df['ATR_14'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    # Stochastic Oscillator (STOCH)
    df['slowk'], df['slowd'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # Average Directional Index (ADX)
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Volume Weighted Average Price (VWAP) approximation
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # On-Balance Volume (OBV)
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])
    
    # Williams %R
    df['Williams_%R'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Feature engineering for trend-based analysis
    df['Price_Change'] = df['Close'].pct_change()  
    df['Volatility'] = df['ATR_14'] / df['Close'] 
    df['Momentum'] = df['Close'] - df['Close'].shift(14)  

    # Reset index to make 'Date' a column again
    df.reset_index(inplace=True)
    
    logging.info("All technical indicators and additional features applied successfully.")
    return df

# Save DataFrame with indicators to CSV
def save_with_indicators(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Data with technical indicators saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to CSV: {str(e)}")

# Example usage with actual stock data
if __name__ == "__main__":
    data = {
        'Date': ['2024-10-01', '2024-10-02', '2024-10-03'],
        'Close': [150.0, 152.0, 155.0],
        'High': [151.0, 153.0, 156.0],
        'Low': [148.0, 149.0, 150.0],
        'Volume': [1000000, 1100000, 1200000]
    }

    df = pd.DataFrame(data)

    # Apply indicators
    df_with_indicators = apply_indicators(df)

    # Save to CSV
    save_with_indicators(df_with_indicators, 'technical_indicators_nasdaq_tech.csv')
