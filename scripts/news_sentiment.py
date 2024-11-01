import yfinance as yf
import pandas as pd
import logging
import os
import time
import requests
import concurrent.futures
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry
from datetime import datetime, timedelta
from dotenv import load_dotenv
import torch
from fake_useragent import UserAgent

# Logging configuration
logging.basicConfig(filename=os.path.join('logs', 'news_sentiment.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load environment variables from .env file
load_dotenv(os.path.join('.env'))


# API Keys
REDDIT_API_KEY = os.getenv('REDDIT_API_KEY')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')

# Load FinBERT model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Ensure 'Date' column consistency across all data
def ensure_date_format(date_string):
    """Ensure the date string is in a consistent format."""
    return pd.to_datetime(date_string, format='%Y-%m-%d', errors='coerce')


# 1. Fetch Yahoo Finance news with BeautifulSoup
@sleep_and_retry
@limits(calls=5, period=60)  # Rate limiting to 5 calls per minute
def fetch_yahoo_news(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/news"
    headers = {"User-Agent": UserAgent().random}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('h3')
        news_titles = [article.get_text(strip=True) for article in articles]
        logging.info(f"Yahoo Finance news fetched for {symbol}.")
        return news_titles
    except requests.RequestException as e:
        logging.error(f"Error fetching Yahoo Finance news for {symbol}: {str(e)}")
        return []


# 2. Fetch Twitter news with start and end date
@sleep_and_retry
@limits(calls=15, period=15 * 60)  
def fetch_twitter_news(symbol):
    url = f"https://api.twitter.com/2/tweets/search/recent?query={symbol}"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tweets = response.json().get('data', [])
        logging.info(f"Twitter news fetched for {symbol}.")
        return [tweet['text'] for tweet in tweets]
    except requests.RequestException as e:
        logging.error(f"Error fetching Twitter news for {symbol}: {str(e)}")
        return []


# 3. Fetch Reddit news
@sleep_and_retry
@limits(calls=5, period=60)  # Rate limiting for Reddit API
def fetch_reddit_news(symbol):
    url = f"https://www.reddit.com/search.json?q={symbol}&sort=new&limit=100"
    headers = {"User-Agent": UserAgent().random}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        reddit_data = response.json().get('data', {}).get('children', [])
        logging.info(f"Reddit news fetched for {symbol}.")
        return [post['data']['title'] for post in reddit_data]
    except requests.RequestException as e:
        logging.error(f"Error fetching Reddit news for {symbol}: {str(e)}")
        return []
    


# 4. Analyze sentiment using FinBERT
def analyze_sentiment(news_list):
    sentiment_scores = []
    try:
        inputs = tokenizer(news_list, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            sentiments = torch.argmax(outputs.logits, dim=1).tolist()  # 0: negative, 1: neutral, 2: positive
            sentiment_scores.extend(sentiments)
        logging.info(f"Sentiment analysis completed with scores: {sentiment_scores}")
        return sentiment_scores
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {str(e)}")
        return []
    


# 5. Weighted Merge News Sentiment
def weighted_merge_news_sentiment(yahoo_sentiment, twitter_sentiment, reddit_sentiment):
    # Calculate average sentiment for each source if there are valid values
    yahoo_avg = sum(yahoo_sentiment) / len(yahoo_sentiment) if yahoo_sentiment else 0
    twitter_avg = sum(twitter_sentiment) / len(twitter_sentiment) if twitter_sentiment else 0
    reddit_avg = sum(reddit_sentiment) / len(reddit_sentiment) if reddit_sentiment else 0

    # Apply weights to each source's average sentiment
    weights = {'Yahoo': 0.5, 'Twitter': 0.3, 'Reddit': 0.2}
    weighted_avg_sentiment = (
        yahoo_avg * weights['Yahoo'] +
        twitter_avg * weights['Twitter'] +
        reddit_avg * weights['Reddit']
    )

    logging.info(f"Weighted average sentiment score: {weighted_avg_sentiment}")
    return weighted_avg_sentiment



# 6. Save Sentiment Data to CSV
def save_sentiment_data(symbol, sentiment_data, file_path='sentiment_data.csv'):
    df = pd.DataFrame(sentiment_data)
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
    logging.info(f"Sentiment data saved for {symbol}")


# Main function to fetch, analyze, and save sentiment data
if __name__ == "__main__":
    symbols_to_test = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    end_date = ensure_date_format(datetime.now().strftime('%Y-%m-%d'))

    for symbol in symbols_to_test:
        yahoo_news = fetch_yahoo_news(symbol)
        twitter_news = fetch_twitter_news(symbol)
        reddit_news = fetch_reddit_news(symbol)

        # Analyze sentiment
        yahoo_sentiment = analyze_sentiment(yahoo_news)
        twitter_sentiment = analyze_sentiment(twitter_news)
        reddit_sentiment = analyze_sentiment(reddit_news)

        # Merge sentiment with different weights
        average_sentiment = weighted_merge_news_sentiment(
            yahoo_sentiment, twitter_sentiment, reddit_sentiment
        )
        logging.info(f"Final sentiment for {symbol}: {average_sentiment}")

        # Save sentiment data
        save_sentiment_data(symbol, [{'Date': end_date, 'Symbol': symbol, 'Sentiment': average_sentiment}])
