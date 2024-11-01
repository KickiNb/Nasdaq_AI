# Dependencies
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import timedelta
from data_collection import load_historical_data, fetch_and_update_stock_data

# Logging configuration
logging.basicConfig(filename=os.path.join('logs', 'preprocess_data.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')



# 1. Ensure every DataFrame has a 'Date' column and is correctly formatted
def ensure_date_column(df, column_name='Date'):
    """Ensure the 'Date' column exists and is properly formatted in the DataFrame."""
    try:
        if df is None or df.empty:
            logging.error(f"DataFrame is None or empty. A valid '{column_name}' is required.")
            return None

        possible_date_columns = ['Date', 'transactionDate', 'Date Reported', 'lastTradeDate', 'expirationDate']
        found_date_column = None
        for possible_column in possible_date_columns:
            if possible_column in df.columns:
                found_date_column = possible_column
                break

        if found_date_column:
            df[column_name] = pd.to_datetime(df[found_date_column], errors='coerce')
        else:
            logging.error(f"No valid date column found in DataFrame. Available columns: {df.columns.tolist()}. A valid 'Date' is required.")
            return None

        df.dropna(subset=[column_name], inplace=True)
        df.sort_values(by=column_name, inplace=True)

        if df.index.name == column_name:
            df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:
        logging.error(f"Error in ensuring date column: {str(e)}")
        return None



# 2. Handle Missing Data (Forward-fill and Backward-fill)
def handle_missing_data(df):
    """Handle missing values by forward filling first, then backward filling if needed."""
    if df is not None:
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        logging.info("Missing data handled with forward and backward fill.")
    else:
        logging.warning("DataFrame is None. Skipping missing data handling.")
    return df


# 3. Drop all NaN columns or rows
def drop_all_nan_columns(df):
    df = df.dropna(axis=1, how='all')
    logging.info("Dropped columns with all NaN values.")
    return df


def drop_all_nan_rows(df):
    df = df.dropna(axis=0, how='all')
    logging.info("Dropped rows with all NaN values.")
    return df


def check_for_nans(df):
    nan_counts = df.isna().sum()
    logging.info(f"NaN count per column: {nan_counts}")
    total_nans = df.isna().sum().sum()
    logging.info(f"Total NaNs in DataFrame: {total_nans}")


def log_dataframe_info(df):
    logging.info(f"DataFrame info: {df.info()}")
    logging.info(f"DataFrame sample: \n{df.head()}")


# 4. Combine All Data
def combine_all_data(stock_data, sentiment_data, macroeconomic_data):
    """Combine all relevant data into one DataFrame for model training."""
    stock_data = ensure_date_column(stock_data)
    sentiment_data = ensure_date_column(sentiment_data) if sentiment_data is not None else None
    macroeconomic_data = ensure_date_column(macroeconomic_data) if macroeconomic_data is not None else None

    if stock_data is None:
        logging.error("Stock data is invalid after ensuring 'Date' column. Aborting combine process.")
        return None

    combined_data = stock_data
    if sentiment_data is not None and not sentiment_data.empty:
        combined_data = pd.merge(combined_data, sentiment_data, on=['Date', 'Symbol'], how='left').fillna(0)
    
    if macroeconomic_data is not None and not macroeconomic_data.empty:
        combined_data = pd.merge(combined_data, macroeconomic_data, on='Date', how='left').fillna(0)

    logging.info("All data combined successfully.")
    return combined_data


# 5. Feature Engineering
def add_feature_engineering(df):
    """Add additional features like percentage changes, moving averages, and ratios."""
    # Ensure numeric columns are converted properly
    numeric_columns = ['Close', 'High', 'Low', 'Volume', 'Open']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN after type conversion
    df.dropna(subset=numeric_columns, inplace=True)

    # Calculate percentage change for each feature
    for column in numeric_columns:
        try:
            df[f'{column}_pct_change'] = df[column].pct_change()
        except Exception as e:
            logging.error(f"Error calculating percentage change for {column}: {str(e)}")

    # Calculate Moving Averages
    try:
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    except Exception as e:
        logging.error(f"Error calculating moving averages: {str(e)}")

    # Calculate Volatility
    if 'Close' in df.columns:
        try:
            df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")

    logging.info("Feature engineering completed.")
    return df


# 6. Scaling and Normalization
def scale_data(df):
    """Scale data using MinMaxScaler."""
    if df is None or df.empty:
        logging.error("DataFrame is None or empty. Scaling skipped.")
        return df

    # Only include numeric columns for scaling
    numeric_columns_to_scale = df.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()
    try:
        df[numeric_columns_to_scale] = scaler.fit_transform(df[numeric_columns_to_scale])
        logging.info("Data scaling and normalization completed.")
    except Exception as e:
        logging.error(f"Error during scaling: {str(e)}")

    return df


# 7. Preprocess all data
def preprocess_data(stock_data, sentiment_data=None, macroeconomic_data=None):
    """Preprocess the stock, sentiment, and macroeconomic data, including merging, feature engineering, and scaling."""
    # Ensure 'Date' is in datetime format
    stock_data = ensure_date_column(stock_data, column_name='Date')
    sentiment_data = ensure_date_column(sentiment_data) if sentiment_data is not None else None
    macroeconomic_data = ensure_date_column(macroeconomic_data) if macroeconomic_data is not None else None

    # Check if stock_data is still valid after ensuring Date format
    if stock_data is None or stock_data.empty:
        logging.error("Stock data is invalid after ensuring 'Date' column. Aborting preprocessing.")
        return None

    # Handle missing data
    stock_data = handle_missing_data(stock_data)

    # Drop rows and columns that are all NaN
    stock_data = drop_all_nan_columns(stock_data)
    stock_data = drop_all_nan_rows(stock_data)

    # Combine stock data with sentiment and macroeconomic data if provided
    combined_data = combine_all_data(stock_data, sentiment_data, macroeconomic_data)

    # Add feature engineering
    combined_data = add_feature_engineering(combined_data)
    if combined_data is None or combined_data.empty:
        logging.error("Feature engineering resulted in an empty dataset. Aborting preprocessing.")
        return None

    # Fill any remaining NaNs and replace infinity values before scaling
    combined_data.fillna(0, inplace=True)
    combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_data.dropna(inplace=True)

    # Scale data
    scaled_data = scale_data(combined_data)
    if scaled_data is None or scaled_data.empty:
        logging.error("Data scaling resulted in an empty dataset. Aborting preprocessing.")
        return None

    logging.info("Preprocessing completed successfully.")
    return scaled_data



# Example Usage
if __name__ == "__main__":
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    root_dir = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/daily'
    macroeconomic_data_path = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/macro_economic_data.csv'

    # Load Historical Data
    combined_stock_data = load_historical_data(root_dir)

    # Load Macroeconomic Data
    if os.path.exists(macroeconomic_data_path):
        macroeconomic_data = pd.read_csv(macroeconomic_data_path)
    else:
        macroeconomic_data = None
        logging.warning(f"Macroeconomic data file not found at {macroeconomic_data_path}.")

    # Step 2: Data Preprocessing
    preprocessed_data = preprocess_data(combined_stock_data, macroeconomic_data=macroeconomic_data)

    if preprocessed_data is not None:
        preprocessed_data.to_csv('C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/preprocessed_stock_data.csv', index=False)
        logging.info("Preprocessed data saved successfully.")