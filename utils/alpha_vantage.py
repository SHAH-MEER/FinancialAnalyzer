import requests
import pandas as pd
from datetime import datetime
import streamlit as st
import os

class AlphaVantageAPI:
    def __init__(self, api_key=None):
        # Try different methods to get API key
        self.api_key = (
            api_key or 
            os.getenv('ALPHA_VANTAGE_API_KEY') or 
            st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
        )
        
        if not self.api_key:
            st.warning("Alpha Vantage API key not found. Some features may be limited.")
            
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = 5  # calls per minute
        self.calls = []
    
    def _check_rate_limit(self):
        """Check API rate limit"""
        now = datetime.now()
        self.calls = [call for call in self.calls if (now - call).seconds < 60]
        if len(self.calls) >= self.rate_limit:
            wait_time = 60 - (now - self.calls[0]).seconds
            raise Exception(f"Rate limit reached. Please wait {wait_time} seconds.")
        self.calls.append(now)
    
    def get_time_series(self, symbol, interval='daily', outputsize='full'):
        """Get time series data from Alpha Vantage"""
        try:
            self._check_rate_limit()
            
            if not self.api_key:
                return None
                
            params = {
                'function': f'TIME_SERIES_{interval.upper()}',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': outputsize
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            time_series_key = f"Time Series ({interval.capitalize()})"
            if time_series_key not in data:
                raise ValueError("No time series data found")
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.DatetimeIndex(df.index)
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            
            return df
        except Exception as e:
            st.error(f"API error: {str(e)}")
            return None
    
    def get_technical_indicators(self, symbol):
        """Get technical indicators from Alpha Vantage"""
        indicators = {}
        
        # Get RSI
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Technical Analysis: RSI' in data:
            rsi_data = pd.DataFrame.from_dict(data['Technical Analysis: RSI'], 
                                            orient='index')
            indicators['RSI'] = rsi_data
        
        # Add more indicators as needed
        
        return indicators
