# Dependencies
import pandas as pd
import logging
import os
import concurrent.futures
from datetime import datetime, timedelta
from data_collection import load_and_clean_historical_data, fetch_and_update_stock_data, fetch_all_macroeconomic_data, load_historical_data
from technical_ind import apply_indicators
from preprocess_data import preprocess_data
from news_sentiment import fetch_yahoo_news, fetch_reddit_news, fetch_twitter_news, analyze_sentiment

# Logging configuration
logging.basicConfig(filename='C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/logs/pipeline_integration.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Paths for saving intermediate and final results
data_folder = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/'
historical_data_folder = 'C:/Users/kicki/Data_Scientist_projects/Stock_prediction_AI/nasdaq_tech_AI/data/daily/us/nasdaq stocks'

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# Clear old historical data
def clear_old_data():
    """Deletes old historical data files to ensure fresh data collection."""
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    for symbol in stock_symbols:
        historical_data_path = os.path.join(data_folder, f'historical_data_{symbol}.csv')
        if os.path.exists(historical_data_path):
            os.remove(historical_data_path)
            logging.info(f"Deleted old historical data for {symbol}: {historical_data_path}")
        else:
            logging.info(f"No existing historical data found for {symbol} to delete.")
                         
                         
def main(full_refresh=False):
    try:
        # Option to clear old data
        if full_refresh:
            logging.info("Full refresh selected. Clearing old historical data.")
            clear_old_data()

        # Step 1: Data Collection
        logging.info("Step 1: Data Collection - Started")
        stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
        all_stock_data = []

        # Load full historical data from the 'daily/us/nasdaq stocks' folder for each stock symbol
        for symbol in stock_symbols:
            found_file = False
            historical_data = None
            for folder in ['1', '2', '3']:
                historical_data_path = os.path.join(historical_data_folder, folder, f'{symbol.lower()}.us.txt')
                if os.path.exists(historical_data_path):
                    # Load and clean historical data
                    historical_data = load_and_clean_historical_data(historical_data_path)
                    if historical_data is not None:
                        logging.info(f"Columns in historical data for {symbol}: {historical_data.columns.tolist()}")
                        if 'Date' not in historical_data.columns:
                            logging.error(f"'Date' column is missing after data cleaning for {symbol}.")
                        else:
                            historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')
                            historical_data['Symbol'] = symbol  # Add 'Symbol' column for merging later
                            found_file = True
                            break
                    else:
                        logging.error(f"Failed to clean data for file: {historical_data_path}")
    
            if not found_file or historical_data is None or 'Date' not in historical_data.columns:
                logging.warning(f"Historical data file for {symbol} not found or 'Date' column missing in any folder within 'daily/us/nasdaq stocks'.")
                continue

            # Update with recent data from Polygon API (if required)
            latest_date = pd.to_datetime(historical_data['Date']).max()
            start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            if start_date <= end_date:
                updated_data = fetch_and_update_stock_data(symbol, historical_data_path, start_date)
                if updated_data is not None and not updated_data.empty:
                    updated_data['Date'] = pd.to_datetime(updated_data['Date'])
                    updated_data['Symbol'] = symbol  # Ensure 'Symbol' column is present in updated data
                    # Combine the full historical data with updated data
                    combined_data = pd.concat([historical_data, updated_data], ignore_index=True).drop_duplicates(subset=['Date']).sort_values(by='Date')
                    logging.info(f"Historical data updated for {symbol}.")
                else:
                    combined_data = historical_data
                    logging.info(f"No new data to fetch for {symbol} after {latest_date.strftime('%Y-%m-%d')}.")
            else:
                combined_data = historical_data
                logging.info(f"No new data to fetch for {symbol} after {latest_date.strftime('%Y-%m-%d')}.")
    
            # Save combined data to the respective CSV file
            if 'Date' in combined_data.columns:
                combined_data.to_csv(os.path.join(data_folder, f"historical_data_{symbol}.csv"), index=False)
                all_stock_data.append(combined_data)
            else:
                logging.error(f"'Date' column missing in combined data for {symbol}. Skipping save.")

        # Check if any valid stock data was collected
        if not all_stock_data:
            logging.error("No valid stock data collected. Pipeline cannot proceed.")
            return

        # Combine all stock data into a single DataFrame
        combined_stock_data = pd.concat(all_stock_data, ignore_index=True)

        # Verify 'Date' and 'Symbol' column presence
        if 'Date' not in combined_stock_data.columns or 'Symbol' not in combined_stock_data.columns:
            logging.error("Combined stock data does not contain 'Date' or 'Symbol' column. Aborting pipeline.")
            return

        # Save the combined stock data
        combined_stock_data.to_csv(os.path.join(data_folder, 'combined_stock_data.csv'), index=False)
        logging.info("Step 1: Data Collection - Completed")

        # Step 2: Apply Technical Indicators
        logging.info("Step 2: Applying Technical Indicators - Started")
        if combined_stock_data is not None and not combined_stock_data.empty:
            try:
                stock_data_with_indicators = apply_indicators(combined_stock_data)

                # Save data with indicators
                stock_data_with_indicators.to_csv(os.path.join(data_folder, 'stock_data_with_indicators.csv'), index=False)
                logging.info("Step 2: Applying Technical Indicators - Completed")
            except Exception as e:
                logging.error(f"Error applying technical indicators: {str(e)}")
                return
        else:
            logging.error("Stock data is missing or invalid. Pipeline cannot proceed.")
            return  # Exit since stock data is essential

        # Step 3: News Sentiment Collection and Analysis
        logging.info("Step 3: News Sentiment Collection and Analysis - Started")
        all_sentiments = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(fetch_sentiments_for_symbol, symbol): symbol for symbol in stock_symbols
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    sentiment_df = future.result()
                    if sentiment_df is not None:
                        sentiment_df['Symbol'] = symbol  
                        all_sentiments.append(sentiment_df)
                except Exception as e:
                    logging.error(f"Error fetching sentiment for symbol {symbol}: {str(e)}")

        if not all_sentiments:
            logging.warning("No sentiment data collected.")
        else:
            combined_sentiment_data = pd.concat(all_sentiments, ignore_index=True)
            combined_sentiment_data.to_csv(os.path.join(data_folder, 'combined_sentiment_data.csv'), index=False)
            logging.info("Step 3: News Sentiment Collection and Analysis - Completed")

        # Step 4: Data Preprocessing
        logging.info("Step 4: Data Preprocessing - Started")
        try:
            # Preprocess the data (stock data with indicators + sentiment data)
            preprocessed_data = preprocess_data(
                stock_data_with_indicators,
                sentiment_data=combined_sentiment_data if 'combined_sentiment_data' in locals() else None
            )

            # Check if preprocessing was successful
            if preprocessed_data is not None and not preprocessed_data.empty:
                # Save preprocessed data
                preprocessed_data.to_csv(os.path.join(data_folder, 'preprocessed_stock_data.csv'), index=False)
                logging.info("Step 4: Data Preprocessing - Completed")
            else:
                logging.error("Preprocessed data is None or empty. Data was not saved.")
        except Exception as e:
            logging.error(f"Error during data preprocessing: {str(e)}")

        logging.info("Pipeline Integration Completed Successfully")

        # Step 5: Fetch Macroeconomic Data
        logging.info("Step 5: Fetching Macroeconomic Data - Started")
        try:
            macroeconomic_data = fetch_all_macroeconomic_data(start_date='2015-01-01', end_date=datetime.now().strftime('%Y-%m-%d'))
            if macroeconomic_data is not None and not macroeconomic_data.empty:
                macroeconomic_data.to_csv(os.path.join(data_folder, 'macroeconomic_data.csv'), index=False)
                logging.info("Step 5: Fetching Macroeconomic Data - Completed")
            else:
                logging.warning("No macroeconomic data fetched.")
        except Exception as e:
            logging.error(f"Error fetching macroeconomic data: {str(e)}")

        logging.info("Pipeline Integration Completed Successfully")

    except Exception as e:
        logging.error(f"An error occurred during pipeline integration: {str(e)}")


def fetch_sentiments_for_symbol(symbol):
    """Fetch and analyze sentiments for the given stock symbol."""
    yahoo_news = fetch_yahoo_news(symbol)
    reddit_news = fetch_reddit_news(symbol)
    twitter_news = fetch_twitter_news(symbol)

    # Analyze sentiment from news sources
    yahoo_sentiment = analyze_sentiment(yahoo_news)
    reddit_sentiment = analyze_sentiment(reddit_news)
    twitter_sentiment = analyze_sentiment(twitter_news)

    # Average sentiment calculation (equal weighting for simplicity)
    all_sentiments = yahoo_sentiment + reddit_sentiment + twitter_sentiment
    if all_sentiments:
        average_sentiment = sum(all_sentiments) / len(all_sentiments)
    else:
        average_sentiment = 0

    sentiment_df = pd.DataFrame([{
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Symbol': symbol,
        'Sentiment': average_sentiment
    }])
    return sentiment_df

if __name__ == "__main__":
    main()