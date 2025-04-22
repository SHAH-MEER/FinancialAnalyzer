import sqlite3
from datetime import datetime
import pandas as pd
import json

class DatabaseManager:
    def __init__(self, db_path='portfolio.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    date TEXT,
                    ticker TEXT,
                    shares REAL,
                    price REAL,
                    action TEXT,
                    PRIMARY KEY (date, ticker)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    date TEXT,
                    ticker TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (date, ticker)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    date TEXT,
                    ticker TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    volume INTEGER,
                    data JSON,
                    PRIMARY KEY (date, ticker, source)
                )
            ''')
    
    def save_portfolio_action(self, ticker, shares, price, action):
        """Save portfolio transaction"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO portfolio_history VALUES (?, ?, ?, ?, ?)',
                (datetime.now().isoformat(), ticker, shares, price, action)
            )
    
    def get_portfolio_history(self, ticker=None):
        """Get portfolio history for analysis"""
        query = 'SELECT * FROM portfolio_history'
        if ticker:
            query += f" WHERE ticker = '{ticker}'"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def save_price_data(self, ticker, data):
        """Save price history data"""
        with sqlite3.connect(self.db_path) as conn:
            for date, row in data.iterrows():
                conn.execute(
                    'INSERT OR REPLACE INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (date.isoformat(), ticker, row['Open'], row['High'], 
                     row['Low'], row['Close'], row['Volume'])
                )
    
    def save_sentiment_data(self, ticker, source, sentiment_score, volume, data):
        """Save sentiment analysis data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO sentiment_data VALUES (?, ?, ?, ?, ?, ?)',
                (datetime.now().isoformat(), ticker, source, 
                 sentiment_score, volume, json.dumps(data))
            )
