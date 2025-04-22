import yfinance as yf
import requests
from datetime import datetime

class YahooFinanceProvider:
    def get_stock_data(self, ticker: str, start_date: str = None, end_date: str = None):
        """Get stock data from Yahoo Finance"""
        try:
            return yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        except Exception as e:
            raise Exception(f"Error fetching data from Yahoo Finance: {str(e)}")

class NewsAPIProvider:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "YOUR_API_KEY"
        self.base_url = "https://newsapi.org/v2"

    def get_stock_news(self, ticker: str, days: int = 7):
        """Get news for a stock"""
        try:
            url = f"{self.base_url}/everything"
            params = {
                "q": ticker,
                "apiKey": self.api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get("articles", [])
        except Exception as e:
            raise Exception(f"Error fetching news: {str(e)}")

class CryptoProvider:
    def get_crypto_data(self, symbol: str, start_date: str = None, end_date: str = None):
        """Get cryptocurrency data (using Yahoo Finance as source)"""
        try:
            # Append -USD to symbol if not already present
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"
            return yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        except Exception as e:
            raise Exception(f"Error fetching crypto data: {str(e)}")
