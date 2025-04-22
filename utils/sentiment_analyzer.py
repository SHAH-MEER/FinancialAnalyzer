import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import requests
from datetime import datetime, timedelta
import streamlit as st

class SocialSentimentAnalyzer:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.sources = ['twitter', 'reddit', 'stocktwits']
        
    @st.cache_data(ttl=3600)
    def analyze_social_sentiment(self, ticker, days=7):
        """Analyze social media sentiment for a ticker"""
        results = {
            'twitter': self._analyze_twitter(ticker, days),
            'reddit': self._analyze_reddit(ticker, days),
            'stocktwits': self._analyze_stocktwits(ticker, days)
        }
        
        # Aggregate sentiment
        combined_sentiment = np.mean([
            result['sentiment_score'] 
            for result in results.values() 
            if result['sentiment_score'] is not None
        ])
        
        # Save to database
        for source, data in results.items():
            if data['sentiment_score'] is not None:
                self.db_manager.save_sentiment_data(
                    ticker=ticker,
                    source=source,
                    sentiment_score=data['sentiment_score'],
                    volume=data['volume'],
                    data=data['raw_data']
                )
        
        return results, combined_sentiment
    
    def _clean_text(self, text):
        """Clean text for sentiment analysis"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        return text.strip()
    
    def _analyze_text_sentiment(self, texts):
        """Analyze sentiment of text list"""
        if not texts:
            return None, 0
            
        sentiments = []
        for text in texts:
            clean_text = self._clean_text(text)
            blob = TextBlob(clean_text)
            sentiments.append(blob.sentiment.polarity)
        
        return np.mean(sentiments), len(texts)
    
    def _analyze_twitter(self, ticker, days):
        """Analyze Twitter sentiment using API"""
        # Implement with your preferred Twitter API
        # For demo, return dummy data
        return {
            'sentiment_score': 0.2,
            'volume': 100,
            'raw_data': {'source': 'twitter', 'count': 100}
        }
    
    def _analyze_reddit(self, ticker, days):
        """Analyze Reddit sentiment using PRAW"""
        # Implement with PRAW (Reddit API wrapper)
        # For demo, return dummy data
        return {
            'sentiment_score': 0.1,
            'volume': 50,
            'raw_data': {'source': 'reddit', 'count': 50}
        }
    
    def _analyze_stocktwits(self, ticker, days):
        """Analyze Stocktwits sentiment"""
        # Implement with Stocktwits API
        # For demo, return dummy data
        return {
            'sentiment_score': 0.3,
            'volume': 75,
            'raw_data': {'source': 'stocktwits', 'count': 75}
        }
