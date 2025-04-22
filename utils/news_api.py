import requests
import streamlit as st
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import List, Dict
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class NewsAPI:
    def __init__(self, api_key: str = os.getenv('NEWS_API_KEY')):
        if not api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        self.api_key = api_key
        self.base_url = 'https://newsapi.org/v2/everything'

    def get_stock_news(self, symbols: List[str], days: int = 7, category: str = None, sentiment: str = None) -> List[Dict]:
        """
        Fetch filtered news with enhanced parameters
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Build advanced query
        query_parts = []
        for symbol in symbols:
            company_query = f'("{symbol}" OR "${symbol}")'
            if category:
                category_queries = {
                    'earnings': 'AND (earnings OR revenue OR "financial results")',
                    'market': 'AND (trading OR "stock market" OR "market analysis")',
                    'company': 'AND (company OR corporate OR business)',
                    'analysis': 'AND (analysis OR forecast OR prediction)'
                }
                company_query += f' {category_queries.get(category, "")}'
            query_parts.append(company_query)
        
        query = ' OR '.join(query_parts)
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100  # Get more articles for better filtering
        }
        
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
            
        articles = response.json().get('articles', [])
        
        # Enhanced article processing
        processed_articles = []
        for article in articles:
            # Add sentiment analysis
            article['sentiment'] = analyze_sentiment(f"{article['title']} {article['description']}")
            
            # Filter by sentiment if specified
            if sentiment and article['sentiment']['category'] != sentiment.lower():
                continue
                
            # Add relevance score
            article['relevance_score'] = self._calculate_relevance(article, symbols)
            
            # Extract related stocks
            article['related_stocks'] = self._extract_related_stocks(article['content'], symbols)
            
            # Clean and validate article data
            if self._validate_article(article):
                processed_articles.append(article)
        
        # Sort by relevance score
        processed_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return processed_articles

    def _validate_article(self, article: Dict) -> bool:
        """Validate article has required fields and content"""
        required_fields = ['title', 'description', 'url', 'publishedAt', 'source']
        return all(field in article and article[field] for field in required_fields)

    def _calculate_relevance(self, article: Dict, symbols: List[str]) -> float:
        """Calculate article relevance score"""
        score = 0
        text = f"{article['title']} {article['description']} {article['content']}"
        for symbol in symbols:
            score += text.lower().count(symbol.lower()) * 2
        return min(score / len(symbols), 10)  # Normalize score to 0-10

    def _extract_related_stocks(self, content: str, watched_symbols: List[str]) -> List[str]:
        """Extract mentioned stock symbols"""
        if not content:
            return []
        return [symbol for symbol in watched_symbols if symbol.lower() in content.lower()]

    def get_market_news(self, days: int = 1) -> List[Dict]:
        """
        Fetch general market news for the last N days
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {
            'q': 'stock market OR financial markets',
            'from': from_date,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
            
        data = response.json()
        return data.get('articles', [])

@st.cache_data(ttl=900)  # Cache for 15 minutes
def analyze_sentiment(text):
    """Enhanced sentiment analysis"""
    try:
        analysis = TextBlob(text)
        # More granular sentiment categories
        if analysis.sentiment.polarity > 0.3:
            return {'category': 'very positive', 'score': analysis.sentiment.polarity}
        elif analysis.sentiment.polarity > 0:
            return {'category': 'positive', 'score': analysis.sentiment.polarity}
        elif analysis.sentiment.polarity < -0.3:
            return {'category': 'very negative', 'score': analysis.sentiment.polarity}
        elif analysis.sentiment.polarity < 0:
            return {'category': 'negative', 'score': analysis.sentiment.polarity}
        return {'category': 'neutral', 'score': 0}
    except:
        return {'category': 'neutral', 'score': 0}

@st.cache_data(ttl=900)
def fetch_stock_news(ticker, api_key, days_back=7):
    """Fetch and analyze news articles"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch news
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                'q': f'"{ticker}" AND (stock OR market)',
                'apiKey': api_key,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt'
            }
        )
        
        if response.status_code != 200:
            return []
            
        articles = response.json().get('articles', [])
        
        # Add sentiment analysis
        for article in articles:
            text = f"{article['title']} {article['description'] or ''}"
            article['sentiment'] = analyze_sentiment(text)
            
        return articles[:5]  # Return top 5 articles
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []