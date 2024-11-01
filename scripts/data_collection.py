# Dependencies 
import requests
import requests_cache
import pandas as pd
import os
import time
import gc
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import diskcache


# Logging configuration
logging.basicConfig(filename=os.path.join('logs', 'data_collection.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load environment variables from .env file
load_dotenv(os.path.join('.env'))

# API Keys
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')


# Cache configuration
cache = diskcache.Cache('./cache')

# Retry mechanism for API calls for failures or rate limits
session = requests.Session()
retry = Retry(connect=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)

# Enhanced retry function with exponential backoff and rate limit handling
def cached_retry_request(url, headers=None, max_attempts=5, initial_delay=10, timeout=10):
    # Check if the response is already cached
    cached_response = cache.get(url)
    if cached_response is not None:
        logging.info(f"Cache hit for URL: {url}")
        return cached_response

    delay = initial_delay
    for attempt in range(max_attempts):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            if response.text.strip():  # Ensure response is not empty
                cache.set(url, response, expire=86400)  # Cache the response for 24 hours
                logging.info(f"Fetched and cached response for URL: {url}")
                return response
            else:
                logging.error(f"Empty response for URL: {url}")
                return None
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 403:
                logging.error(f"403 Forbidden error for URL: {url}. Check API key permissions or subscription level.")
                return None
            elif response.status_code == 429:
                logging.warning(f"Rate limit exceeded for URL: {url}. Retrying after {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            elif response.status_code in [500, 502, 503, 504]:
                logging.warning(f"Attempt {attempt + 1}/{max_attempts} failed for URL: {url} with error: {str(http_err)}. Retrying after {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logging.error(f"HTTP error occurred for URL: {url}: {str(http_err)}")
                return None
        except Exception as err:
            logging.error(f"Unexpected error for URL: {url}: {str(err)}")
            return None

    logging.error(f"Max attempts reached for URL: {url}")
    return None


# 1. Load historical stock data
def load_and_clean_historical_data(file_path):
    """Load and clean the historical stock data to match the expected format."""
    try:
        # Load the data
        df = pd.read_csv(file_path, header=0)  


        # Check if columns are already cleaned or need to be processed
        if list(df.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
            logging.info(f"Data already cleaned for file: {file_path}")
            return df

        # If columns are raw data, then proceed with cleaning
        expected_columns = ['<TICKER>', '<PER>', '<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>', '<OPENINT>']
        if list(df.columns) != expected_columns:
            logging.error(f"Unexpected columns in file {file_path}. Expected: {expected_columns}, Found: {list(df.columns)}")
            return None

        # Drop unnecessary columns
        df = df[['<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]

        # Rename columns to match the expected format
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Convert 'Date' to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)  # Ensure only rows with valid dates are kept

        # Ensure numeric columns are of the correct type
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaN values in important columns
        df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # Sort data by 'Date'
        df.sort_values(by='Date', inplace=True)

        # Verify if the Date column exists after all transformations
        if 'Date' not in df.columns:
            logging.error(f"'Date' column is missing in the cleaned data for file: {file_path}")
            return None

        logging.info(f"Data successfully cleaned for file: {file_path}")
        return df

    except Exception as e:
        logging.error(f"Error loading and cleaning file {file_path}: {str(e)}")
        return None
    

# Load historical stock data in chunks
def load_historical_data(root_dir):
    stock_data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                logging.info(f"Loading historical data file: {file_path}")
                df = load_and_clean_historical_data(file_path)
                if df is not None:
                    stock_data.append(df)

    if not stock_data:
        logging.error("No historical stock files found.")
        return None

    combined_data = pd.concat(stock_data, ignore_index=True)
    combined_data.sort_values(by='Date', inplace=True)
    logging.info(f"Combined historical data shape: {combined_data.shape}")

    return combined_data


# 2. Fetch daily stock data from Polygon API
def fetch_and_update_stock_data(symbol, historical_data_path, api_start_date='2024-10-28'):
    try:
        # Load historical data from the cleaned file
        if os.path.exists(historical_data_path):
            historical_data = pd.read_csv(historical_data_path)
            logging.info(f"Loaded local historical data for {symbol}.")
        else:
            logging.error(f"Local historical data file not found for {symbol}.")
            return None

        # Ensure 'Date' column is present and in datetime format
        if 'Date' not in historical_data.columns:
            logging.error(f"'Date' column not found in historical data for {symbol}.")
            return None

        historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')
        historical_data.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates
        
        # Fetch only recent data from the Polygon API if necessary
        latest_date = pd.to_datetime(historical_data['Date']).max() + timedelta(days=1)
        start_date = max(latest_date, pd.to_datetime(api_start_date))
        end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date <= pd.to_datetime(end_date):
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date}?apiKey={POLYGON_API_KEY}"
            response = cached_retry_request(url)

            if response is None:
                logging.error(f"Failed to fetch daily data for {symbol}.")
                return historical_data  # Return existing data even if API fails

            data = response.json()
            if 'results' in data and isinstance(data['results'], list):
                df = pd.DataFrame(data['results'])
                required_columns = {'t', 'o', 'h', 'l', 'c', 'v'}
                if required_columns.issubset(df.columns):
                    df['Date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df[['Date', 'o', 'h', 'l', 'c', 'v']]
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'int'})

                    # Combine the existing historical data with the new data from API
                    combined_data = pd.concat([historical_data, df], ignore_index=True).drop_duplicates(subset=['Date']).sort_values(by='Date')
                    combined_data.to_csv(historical_data_path, index=False)
                    logging.info(f"Daily stock data updated successfully for {symbol}.")
                    return combined_data
                else:
                    logging.error(f"Unexpected data format for {symbol}.")
            else:
                logging.error(f"No results found for {symbol}.")
        else:
            logging.info(f"No new data to fetch for {symbol} after {start_date.strftime('%Y-%m-%d')}.")
            return historical_data

    except Exception as e:
        logging.error(f"Unexpected error fetching stock data for {symbol}: {str(e)}")
        return None  # Return None in case of error to indicate failure
    

# 3. Fetch FRED Macroeconomic data
def fetch_fred_macroeconomic(series_id, start_date='1980-01-01', end_date=None):
    """Fetch macroeconomic data from FRED API."""
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&observation_start={start_date}&observation_end={end_date}&api_key={FRED_API_KEY}&file_type=json"
    
    logging.info(f"Fetching data for {series_id} from {start_date} to {end_date}")
    response = cached_retry_request(url)
    if response is None:
        logging.error(f"Failed to fetch macroeconomic data for series_id: {series_id}")
        return None
    
    try:
        response.raise_for_status()
        data = response.json()
        if 'observations' in data and isinstance(data['observations'], list):
            df = pd.DataFrame(data['observations'])
            df['Date'] = pd.to_datetime(df['date'])
            df = df[['Date', 'value']].rename(columns={'value': series_id})
            return df
        else:
            logging.error(f"No data returned for series {series_id}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching macroeconomic data for {series_id}: {e}")
        return pd.DataFrame()
    

def fetch_all_macroeconomic_data(start_date='1980-01-01', end_date=None):
    """Fetch and combine multiple macroeconomic indicators."""
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logging.info(f"Fetching all macroeconomic data from {start_date} to {end_date}")

    indicators = {
        "CPI": "CPIAUCSL",  # Consumer Price Index
        "PPI": "PPIACO",  # Producer Price Index
        "UnemploymentRate": "UNRATE",  # Unemployment Rate
        "GDP": "GDP",  # Gross Domestic Product
        "2Y_Treasury": "DGS2",  # 2-Year Treasury Rate
        "10Y_Treasury": "DGS10",  # 10-Year Treasury Yield
        "GoldPrices": "GOLDAMGBD228NLBM",  # Gold Prices
        "CrudeOil": "DCOILWTICO",  # Crude Oil Prices
        "DollarIndex": "DTWEXBGS",  # U.S. Dollar Index
        "RetailSales": "RSXFS",  # Retail Sales
        "FedFundsRate": "FEDFUNDS",  # Federal Funds Rate
        "VIX": "VIXCLS",  # CBOE Volatility Index
        "HousingPermits": "PERMIT",  # Building Permits (Housing Market)
    }

    combined_data = pd.DataFrame()

    # ThreadPoolExecutor to fetch all macroeconomic indicators concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_indicator = {executor.submit(fetch_fred_macroeconomic, series_id, start_date, end_date): name for name, series_id in indicators.items()}
        for future in concurrent.futures.as_completed(future_to_indicator):
            name = future_to_indicator[future]
            try:
                macro_data = future.result()
                if macro_data is not None:
                    if combined_data.empty:
                        combined_data = macro_data
                    else:
                        combined_data = combined_data.merge(macro_data, on='Date', how='left')
            except Exception as e:
                logging.error(f"Error processing macroeconomic data for {name}: {str(e)}")

    # Handle missing or sparse data by forward-filling the missing values
    if not combined_data.empty:
        combined_data.ffill(inplace=True)
        logging.info("All macroeconomic data combined successfully.")
    else:
        logging.warning("No macroeconomic data was fetched successfully.")

    return combined_data



if __name__ == "__main__":
    root_dir = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/daily/us/nasdaq stocks'
    save_dir = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/'

    # Step 1: Load Historical Data
    combined_data = load_historical_data(root_dir)
    if combined_data is not None:
        save_path = os.path.join(save_dir, 'combined_historical_data.csv')
        combined_data.to_csv(save_path, index=False)
        logging.info(f"Combined historical data saved to {save_path}")

    # Step 2: Fetch and Update Daily Data using ThreadPoolExecutor for concurrency
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_and_update_stock_data, symbol, os.path.join(save_dir, f'historical_data_{symbol}.csv'), '2024-10-25') for symbol in stock_symbols]
        concurrent.futures.wait(futures)

    logging.info("Daily data fetching and updating completed.")
