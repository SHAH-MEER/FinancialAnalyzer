import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import base64
from io import BytesIO
import sys
import os

# Add explicit import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Update imports
from utils.chart_themes import get_chart_layout, CHART_THEMES
from utils.theme import THEME
from docs.core_features import get_metric_tooltips, get_quick_start_guide, get_feature_help
from docs.page_guides import (  # Changed from doc to docs
    get_dashboard_guide, get_portfolio_management_guide,
    get_stock_analysis_guide, get_risk_assessment_guide,
    get_news_guide
)
from docs.ai_insights_guide import get_ai_insights_guide
from utils.portfolio_optimizer import optimize_portfolio
from utils.news_api import fetch_stock_news, NewsAPI, analyze_sentiment
from utils.ai_insights import PortfolioInsights
from components.news_card import render_news_card
from utils.time_series import TimeSeriesAnalyzer
from utils.anomaly_detection import AnomalyDetector
from utils.dashboard_config import DashboardConfig
from utils.cache_manager import CacheManager
from utils.data_import import import_portfolio_from_csv, export_portfolio_to_csv
from utils.factor_analysis import FactorAnalysis, RiskModel
from utils.pattern_recognition import PatternRecognition
from utils.database import DatabaseManager
from utils.sentiment_analyzer import SocialSentimentAnalyzer
from utils.validation import validate_ticker

def get_theme():
    """Get dark theme"""
    return THEME['dark']  # Always return dark theme

# Custom CSS with improved styling
def get_css():
    theme = get_theme()
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        .stApp {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{  /* Sidebar */
            background-color: {theme['surface']};
            border-right: 1px solid {theme['border']};
        }}
        
        /* Main content area */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }}
        
        /* Card styling */
        .metric-card {{
            background-color: {theme['surface']};
            border: 1px solid {theme['border']};
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}
        
        /* Table styling */
        .dataframe {{
            border: 1px solid {theme['border']};
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        
        .dataframe thead {{
            background-color: {theme['surface']};
            color: {theme['text']['primary']};
        }}
        
        .dataframe th, .dataframe td {{
            padding: 0.75rem 1rem !important;
            border-bottom: 1px solid {theme['border']} !important;
        }}
        
        /* Chart container */
        .chart-container {{
            background-color: {theme['surface']};
            border: 1px solid {theme['border']};
            border-radius: 0.75rem;
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        /* Button styling */
        .stButton>button {{
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            background-color: {theme['primary']};
            color: white;
            border: none;
            transition: all 0.2s;
        }}
        
        .stButton>button:hover {{
            background-color: {theme['primary']};
            opacity: 0.9;
            transform: translateY(-1px);
        }}
        
        /* Select box styling */
        .stSelectbox [data-baseweb="select"] {{
            border-radius: 0.5rem;
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {theme['text']['primary']};
            font-weight: 600;
            margin-bottom: 1.5rem;
        }}
        
        /* Metrics */
        .metric-container {{
            background-color: {theme['surface']};
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid {theme['border']};
        }}
        
        .metric-label {{
            color: {theme['text']['secondary']};
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }}
        
        .metric-value {{
            color: {theme['text']['primary']};
            font-size: 1.5rem;
            font-weight: 600;
        }}
    </style>
    """

def get_metric_tooltip(metric):
    """Get tooltip explanation for financial metrics"""
    tooltips = get_metric_tooltips()
    for category in tooltips.values():
        if metric in category:
            return category[metric]
    return ""

class PortfolioAnalyzer:
    def __init__(self):
        self.portfolio = {}
        self.data = None
        self.benchmark = None
        self.start_date = None
        self.end_date = None
        self.db = DatabaseManager()
        self.sentiment_analyzer = SocialSentimentAnalyzer(self.db)
    
    def add_stock(self, ticker, shares, purchase_date=None):
        """Add a stock to the portfolio with number of shares"""
        if ticker not in self.portfolio:
            self.portfolio[ticker] = {'shares': shares, 'purchase_date': purchase_date}
        else:
            self.portfolio[ticker]['shares'] += shares
        
        # Save to database
        self.db.save_portfolio_action(
            ticker=ticker,
            shares=shares,
            price=self.data['Close'][ticker][-1] if self.data is not None else 0,
            action='BUY'
        )
    
    def remove_stock(self, ticker, shares=None):
        """Remove a stock or reduce shares"""
        if ticker in self.portfolio:
            if shares is None or shares >= self.portfolio[ticker]['shares']:
                del self.portfolio[ticker]
            else:
                self.portfolio[ticker]['shares'] -= shares
        
        # Save to database
        self.db.save_portfolio_action(
            ticker=ticker,
            shares=shares or self.portfolio[ticker]['shares'],
            price=self.data['Close'][ticker][-1] if self.data is not None else 0,
            action='SELL'
        )
    
    def set_timeframe(self, start_date, end_date=None):
        """Set analysis timeframe"""
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
    
    def set_benchmark(self, ticker="^GSPC"):
        """Set benchmark for comparison (default: S&P 500)"""
        self.benchmark = ticker
    
    def fetch_data(self):
        """Fetch historical data for all portfolio stocks"""
        # Allow empty portfolio but show message
        if not self.portfolio:
            st.info("Portfolio is empty. Add stocks to see analysis.")
            return None
        
        if not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if not self.end_date:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get list of tickers
        tickers = list(self.portfolio.keys())
        if self.benchmark and self.benchmark not in tickers:
            tickers.append(self.benchmark)
        
        # Fetch data with auto_adjust=True
        try:
            self.data = yf.download(tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
            
            # Save price data to database
            if self.data is not None:
                for ticker in self.portfolio:
                    self.db.save_price_data(ticker, self.data[ticker])
            
            return self.data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_portfolio_value(self):
        """Calculate daily portfolio value"""
        if self.data is None:
            self.fetch_data()
        
        # Extract close prices (already adjusted when auto_adjust=True)
        close_prices = self.data['Close']
        
        # Handle single ticker case
        if isinstance(close_prices, pd.Series):
            ticker = list(self.portfolio.keys())[0]
            close_prices = pd.DataFrame({ticker: close_prices})
        
        # Calculate daily value for each position
        portfolio_value = pd.DataFrame(index=close_prices.index)
        portfolio_value['Total'] = 0
        
        for ticker, details in self.portfolio.items():
            shares = details['shares']
            if ticker in close_prices.columns:
                portfolio_value[ticker] = close_prices[ticker] * shares
                portfolio_value['Total'] += portfolio_value[ticker]
        
        return portfolio_value
    
    def calculate_returns(self):
        """Calculate daily returns for portfolio"""
        portfolio_value = self.calculate_portfolio_value()
        portfolio_returns = portfolio_value['Total'].pct_change().dropna()
        
        # If benchmark exists, calculate its returns too
        benchmark_returns = None
        if self.benchmark:
            if isinstance(self.data['Close'], pd.DataFrame) and self.benchmark in self.data['Close'].columns:
                benchmark_returns = self.data['Close'][self.benchmark].pct_change().dropna()
            elif isinstance(self.data['Close'], pd.Series):
                # Only one ticker and it's the benchmark
                if self.benchmark == list(self.portfolio.keys())[0]:
                    benchmark_returns = self.data['Close'].pct_change().dropna()
        
        return portfolio_returns, benchmark_returns
    
    def calculate_cumulative_returns(self):
        """Calculate cumulative returns over time"""
        returns, benchmark_returns = self.calculate_returns()
        
        cumulative_returns = (1 + returns).cumprod() - 1
        
        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod() - 1
            return cumulative_returns, cumulative_benchmark
        
        return cumulative_returns, None
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio (annualized)"""
        returns, _ = self.calculate_returns()
        
        # Annualized Sharpe Ratio
        annual_factor = 252  # Trading days in a year
        excess_returns = returns - (risk_free_rate / annual_factor)
        
        # Check for zero std to avoid division by zero
        std = returns.std()
        if std == 0:
            return 0
            
        sharpe_ratio = np.sqrt(annual_factor) * excess_returns.mean() / std
        
        return sharpe_ratio
    
    def calculate_volatility(self, annualized=True):
        """Calculate portfolio volatility"""
        returns, _ = self.calculate_returns()
        
        if annualized:
            return returns.std() * np.sqrt(252)  # Annualized volatility
        else:
            return returns.std()
    
    def calculate_drawdowns(self):
        """Calculate drawdowns"""
        portfolio_value = self.calculate_portfolio_value()['Total']
        
        # Calculate running maximum
        running_max = portfolio_value.cummax()
        
        # Calculate drawdown in percentage terms
        drawdowns = (portfolio_value / running_max) - 1
        
        return drawdowns
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        return self.calculate_drawdowns().min()
    
    def calculate_beta(self):
        """Calculate portfolio beta relative to benchmark"""
        if not self.benchmark:
            raise ValueError("Benchmark not set. Use set_benchmark() first.")
        
        returns, benchmark_returns = self.calculate_returns()
        
        if benchmark_returns is None:
            raise ValueError(f"Benchmark {self.benchmark} data not available")
        
        # Calculate beta using covariance and variance
        covariance = returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        
        # Check for zero variance to avoid division by zero
        if variance == 0:
            return 0
            
        return covariance / variance
    
    def calculate_alpha(self, risk_free_rate=0.02):
        """Calculate Jensen's Alpha"""
        if not self.benchmark:
            raise ValueError("Benchmark not set. Use set_benchmark() first.")
        
        returns, benchmark_returns = self.calculate_returns()
        
        if benchmark_returns is None:
            raise ValueError(f"Benchmark {self.benchmark} data not available")
        
        # Annualize returns
        annual_factor = 252  # Trading days in a year
        portfolio_return = returns.mean() * annual_factor
        market_return = benchmark_returns.mean() * annual_factor
        
        # Calculate beta
        beta = self.calculate_beta()
        
        # Calculate alpha
        alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
        
        return alpha
    
    def calculate_correlation_matrix(self):
        """Calculate correlation matrix between assets"""
        if self.data is None:
            self.fetch_data()
        
        # Get close prices (already adjusted when auto_adjust=True)
        close_prices = self.data['Close']
        
        # Handle single ticker case
        if isinstance(close_prices, pd.Series):
            return pd.DataFrame([[1]], index=[list(self.portfolio.keys())[0]], 
                               columns=[list(self.portfolio.keys())[0]])
        
        # Filter to only include portfolio stocks
        portfolio_tickers = [t for t in self.portfolio.keys() if t in close_prices.columns]
        
        # Calculate returns
        returns = close_prices[portfolio_tickers].pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap using Plotly"""
        corr_matrix = self.calculate_correlation_matrix()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            height=600,
            width=800
        )
        
        return fig
    
    def plot_portfolio_composition(self):
        """Plot portfolio composition using Plotly"""
        if self.data is None:
            self.fetch_data()
        
        # Get latest prices
        if isinstance(self.data['Close'], pd.DataFrame):
            latest_prices = self.data['Close'].iloc[-1]
        else:
            # Handle single ticker case
            ticker = list(self.portfolio.keys())[0]
            latest_prices = pd.Series({ticker: self.data['Close'].iloc[-1]})
        
        # Calculate value of each position
        values = {}
        for ticker, details in self.portfolio.items():
            if ticker in latest_prices.index:
                values[ticker] = latest_prices[ticker] * details['shares']
        
        if not values:
            return None
            
        # Create pie chart using Plotly
        fig = go.Figure(data=go.Pie(
            labels=list(values.keys()),
            values=list(values.values()),
            textinfo='label+percent',
            hoverinfo='label+value'
        ))
        
        fig.update_layout(
            title='Portfolio Composition',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_cumulative_returns(self):
        """Plot cumulative returns over time"""
        cum_returns, cum_benchmark = self.calculate_cumulative_returns()
        theme_colors = CHART_THEMES['dark']['colors']  # Always use dark theme colors
        
        fig = go.Figure()
        
        # Plot portfolio returns
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color=theme_colors['primary'], width=2)
        ))
        
        # Plot benchmark if available
        if cum_benchmark is not None:
            fig.add_trace(go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark * 100,
                mode='lines',
                name=f'Benchmark ({self.benchmark})',
                line=dict(color=theme_colors['neutral'], width=2)
            ))
        
        fig.update_layout(
            **get_chart_layout(
                theme='dark',
                title='Cumulative Returns (%)',
                height=500
            ),
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_drawdowns(self):
        """Plot drawdowns over time"""
        drawdowns = self.calculate_drawdowns()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns * 100,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1),
            name='Drawdown'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdowns (%)',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400,
            yaxis=dict(
                tickformat='.1f'
            )
        )
        
        return fig
    
    def plot_monthly_returns_heatmap(self):
        """Plot monthly returns heatmap using Plotly"""
        returns, _ = self.calculate_returns()
        
        # Convert to monthly returns using end of month frequency - Fix deprecated 'M'
        monthly_returns = returns.resample('ME').agg(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table using pandas pivot method correctly
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Returns': monthly_returns.values
        })
        # Use pivot method correctly
        pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='Returns')
        
        # Create heatmap with corrected properties
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values * 100,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot_table.index,
            text=np.round(pivot_table.values * 100, 2),
            texttemplate='%{text:.2f}%',
            colorscale='RdYlGn',
            zmid=0,  # This replaces 'center' - sets the midpoint of the color scale
            hoverongaps=False,
            showscale=True
        ))
        
        fig.update_layout(
            title='Monthly Returns (%)',
            height=500,
            xaxis_title='Month',
            yaxis_title='Year'
        )
        
        return fig
    
    def plot_candlestick(self, ticker):
        """Plot candlestick chart for a specific stock"""
        if self.data is None:
            self.fetch_data()
        
        if ticker not in self.portfolio:
            raise ValueError(f"Ticker {ticker} not in portfolio")
        
        # Extract OHLC data for the ticker
        if isinstance(self.data['Open'], pd.DataFrame):
            ohlc = pd.DataFrame({
                'Open': self.data['Open'][ticker],
                'High': self.data['High'][ticker],
                'Low': self.data['Low'][ticker],
                'Close': self.data['Close'][ticker],
                'Volume': self.data['Volume'][ticker]
            })
        else:
            # Single ticker case
            ohlc = pd.DataFrame({
                'Open': self.data['Open'],
                'High': self.data['High'],
                'Low': self.data['Low'],
                'Close': self.data['Close'],
                'Volume': self.data['Volume']
            })
        
        # Create candlestick chart using Plotly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['Open'], 
            high=ohlc['High'],
            low=ohlc['Low'], 
            close=ohlc['Close'],
            name=ticker
        ), row=1, col=1)
        
        # Add volume trace
        fig.add_trace(go.Bar(
            x=ohlc.index,
            y=ohlc['Volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.5)'
        ), row=2, col=1)
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=ohlc.index,
            y=ohlc['Close'].rolling(window=20).mean(),
            name='20 Day MA',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=ohlc.index,
            y=ohlc['Close'].rolling(window=50).mean(),
            name='50 Day MA',
            line=dict(color='green', width=1)
        ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        # Style candlesticks
        fig.update_traces(
            increasing_line_color='green',
            decreasing_line_color='red',
            selector=dict(type='candlestick')
        )
        
        return fig
    
    def get_stock_news(self, ticker, limit=5):
        """Get recent news for a stock (placeholder - would use an actual API in production)"""
        # In a real implementation, you would use NewsAPI or similar
        # For now, we'll return sample news
        sample_news = [
            {
                "title": f"Analysts upgrade {ticker} on strong earnings",
                "description": f"{ticker} beat expectations with quarterly results",
                "url": "#",
                "publishedAt": "2025-04-18T15:30:00Z",
                "sentiment": "positive"
            },
            {
                "title": f"{ticker} announces new product line",
                "description": "Innovation continues with new offerings",
                "url": "#",
                "publishedAt": "2025-04-16T12:15:00Z",
                "sentiment": "positive"
            },
            {
                "title": f"{ticker} faces supply chain challenges",
                "description": "Global logistics issues impact production",
                "url": "#",
                "publishedAt": "2025-04-14T09:45:00Z",
                "sentiment": "negative"
            }
        ]
        return sample_news[:limit]
    
    def calculate_value_at_risk(self, confidence_level=0.95):
        """Calculate Value at Risk (VaR)"""
        returns, _ = self.calculate_returns()
        
        # Historical VaR
        var = returns.quantile(1 - confidence_level)
        
        # Get latest portfolio value
        portfolio_value = self.calculate_portfolio_value()['Total'].iloc[-1]
        
        # Convert to dollar amount
        var_amount = portfolio_value * var
        
        return {
            'var_pct': var,
            'var_amount': var_amount,
            'confidence_level': confidence_level
        }
    
    def generate_summary_report(self):
        """Generate a summary report of portfolio performance"""
        if self.data is None:
            self.fetch_data()
        
        # Calculate metrics
        returns, benchmark_returns = self.calculate_returns()
        cum_returns, cum_benchmark = self.calculate_cumulative_returns()
        
        # Portfolio metrics
        total_return = cum_returns.iloc[-1] * 100
        
        # Check if we have enough data for annualized calculations
        if len(returns) > 1:
            annualized_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100
            sharpe = self.calculate_sharpe_ratio()
            volatility = self.calculate_volatility() * 100
            max_dd = self.calculate_max_drawdown() * 100
        else:
            annualized_return = total_return
            sharpe = 0
            volatility = 0
            max_dd = 0
        
        # Calculate portfolio composition
        # Handle both DataFrame and Series cases for multi vs single ticker portfolios
        if isinstance(self.data['Close'], pd.DataFrame):
            latest_prices = self.data['Close'].iloc[-1]
        else:
            ticker = list(self.portfolio.keys())[0]
            latest_prices = pd.Series({ticker: self.data['Close'].iloc[-1]})
        
        total_value = 0
        position_values = {}
        
        for ticker, details in self.portfolio.items():
            if ticker in latest_prices.index:
                position_values[ticker] = latest_prices[ticker] * details['shares']
                total_value += position_values[ticker]
        
        # Value at Risk
        var_data = self.calculate_value_at_risk()
        
        # Create summary
        summary = {
            'Portfolio Value': f'${total_value:,.2f}',
            'Total Return': f'{total_return:.2f}%',
            'Annualized Return': f'{annualized_return:.2f}%',
            'Sharpe Ratio': f'{sharpe:.2f}',
            'Volatility (Annual)': f'{volatility:.2f}%',
            'Maximum Drawdown': f'{max_dd:.2f}%',
            'Value at Risk (95%)': f'${abs(var_data["var_amount"]):,.2f} ({abs(var_data["var_pct"] * 100):.2f}%)'
        }
        
        # Add benchmark comparison if available
        if cum_benchmark is not None:
            benchmark_total_return = cum_benchmark.iloc[-1] * 100
            summary['Benchmark Return'] = f'{benchmark_total_return:.2f}%'
            
            if self.benchmark:
                try:
                    beta = self.calculate_beta()
                    alpha = self.calculate_alpha() * 100
                    summary['Beta'] = f'{beta:.2f}'
                    summary['Alpha (Annual)'] = f'{alpha:.2f}%'
                except:
                    pass
        
        return summary, position_values

def get_ticker_info(ticker):
    """Get info for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/A'),
            'price': info.get('currentPrice', 0),
            'marketCap': info.get('marketCap', 0),
            'pe': info.get('trailingPE', 0),
            'dividend': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except:
        return {
            'name': ticker,
            'sector': 'N/A',
            'price': 0,
            'marketCap': 0,
            'pe': 0,
            'dividend': 0
        }

def get_technical_indicators(data, ticker):
    """Calculate technical indicators for a specific ticker"""
    if isinstance(data['Close'], pd.DataFrame):
        df = pd.DataFrame({
            'Open': data['Open'][ticker],
            'High': data['High'][ticker],
            'Low': data['Low'][ticker],
            'Close': data['Close'][ticker],
            'Volume': data['Volume'][ticker]
        })
    else:
        df = pd.DataFrame({
            'Open': data['Open'],
            'High': data['High'],
            'Low': data['Low'],
            'Close': data['Close'],
            'Volume': data['Volume']
        })
    
    # Calculate indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD calculation
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def get_available_tickers():
    """Get a list of popular tickers plus user additions"""
    default_tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "WMT",
        "JNJ", "PG", "DIS", "NFLX", "ADBE", "KO", "PEP", "NKE", "MCD", "INTC"
    ]
    
    # Get custom tickers from session state
    custom_tickers = st.session_state.get('custom_tickers', set())
    return sorted(list(set(default_tickers + list(custom_tickers))))

def add_custom_ticker():
    """Add a custom ticker after validation"""
    new_ticker = st.session_state.new_ticker.strip().upper()
    if new_ticker:
        with st.spinner(f"Validating {new_ticker}..."):
            if validate_ticker(new_ticker):
                custom_tickers = st.session_state.get('custom_tickers', set())
                custom_tickers.add(new_ticker)
                st.session_state.custom_tickers = custom_tickers
                st.success(f"Added {new_ticker} to available tickers!")
                info = get_ticker_info(new_ticker)
                st.markdown(f"""
                **{info['name']}**
                - Current Price: ${info['price']:.2f}
                - Sector: {info['sector']}
                """)
            else:
                st.error(f"Invalid ticker: {new_ticker}")

def display_portfolio_management(analyzer):
    st.title("💼 Portfolio Management")
    
    with st.expander("📚 Portfolio Management Guide"):
        st.markdown(get_portfolio_management_guide())
    
    # Add custom ticker input
    st.subheader("Add Custom Stock")
    custom_col1, custom_col2 = st.columns([3, 1])
    with custom_col1:
        st.text_input("Enter Stock Ticker", key="new_ticker", 
                     help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT)")
    with custom_col2:
        st.button("Validate & Add", on_click=add_custom_ticker)
    
    st.markdown("---")
    
    # Always show add stock form at the top
    st.subheader("Add Stock to Portfolio")
    col1, col2 = st.columns(2)
    
    with col1:
        available_tickers = get_available_tickers()
        ticker = st.selectbox("Select Stock", available_tickers, key="pm_ticker")
        
        if ticker:
            info = get_ticker_info(ticker)
            st.markdown(f"""
            **{info['name']}**
            - Sector: {info['sector']}
            - Current Price: ${info['price']:.2f}
            - P/E Ratio: {info['pe']:.2f}
            - Dividend Yield: {info['dividend']:.2f}%
            """)
    
    with col2:
        shares = st.number_input("Number of Shares", min_value=0.1, value=1.0, step=0.1, key="pm_shares")
        purchase_date = st.date_input("Purchase Date", value=datetime.now(), key="pm_date")
        
        if st.button("Add to Portfolio", key="pm_add"):
            analyzer.add_stock(ticker, shares, purchase_date.strftime('%Y-%m-%d'))
            st.success(f"Added {shares} shares of {ticker} to portfolio!")
            st.rerun()
    
    st.markdown("---")
    
    # Show current holdings if any exist
    if analyzer.portfolio:
        holdings_data = []
        for ticker, details in analyzer.portfolio.items():
            info = get_ticker_info(ticker)
            holdings_data.append({
                'Ticker': ticker,
                'Shares': details['shares'],
                'Purchase Date': details['purchase_date'],
                'Current Price': info['price'],
                'Market Value': info['price'] * details['shares']
            })
        
        holdings_df = pd.DataFrame(holdings_data)
        
        # Format columns
        holdings_df['Current Price'] = holdings_df['Current Price'].map('${:,.2f}'.format)
        holdings_df['Market Value'] = holdings_df['Market Value'].map('${:,.2f}'.format)
        
        st.table(holdings_df)
        
        # Remove stock section
        st.subheader("Remove Stock")
        col1, col2 = st.columns(2)
        
        with col1:
            remove_ticker = st.selectbox("Select Stock to Remove", list(analyzer.portfolio.keys()))
        
        with col2:
            remove_shares = st.number_input(
                "Number of Shares to Remove",
                min_value=0.0,
                max_value=float(analyzer.portfolio[remove_ticker]['shares']) if remove_ticker else 0.0,
                value=float(analyzer.portfolio[remove_ticker]['shares']) if remove_ticker else 0.0,
                step=1.0
            )
            
            if st.button("Remove from Portfolio"):
                analyzer.remove_stock(remove_ticker, remove_shares)
                st.success(f"Removed {remove_shares} shares of {remove_ticker} from portfolio!")
    else:
        st.info("Your portfolio is empty. Use the form above to add your first stock.")

def display_dashboard(analyzer):
    st.title("📊 Portfolio Dashboard")
    
    with st.expander("📚 How to Use This Dashboard"):
        st.markdown(get_dashboard_guide())
    
    # Load configuration
    config = DashboardConfig.load_config()
    
    # Add dashboard customization
    with st.expander("⚙️ Dashboard Settings"):
        col1, col2 = st.columns(2)
        with col1:
            config['metrics']['enabled'] = st.checkbox("Show Metrics", value=config['metrics']['enabled'])
            config['charts']['enabled'] = st.checkbox("Show Charts", value=config['charts']['enabled'])
        with col2:
            config['metrics']['columns'] = st.selectbox("Metrics Columns", [2,3,4], index=config['metrics']['columns']-2)
            config['charts']['columns'] = st.selectbox("Charts Layout", [1,2], index=config['charts']['columns']-1)
        
        # Save configuration button
        if st.button("Save Layout"):
            DashboardConfig.save_config(config)
            st.success("Dashboard layout saved!")
    
    # Force data refresh if needed - Fix the condition
    if analyzer.data is None or st.button("🔄 Refresh Data"):
        with st.spinner("Fetching latest market data..."):
            analyzer.fetch_data()
            if analyzer.data is None:
                st.error("Failed to fetch market data. Please try again.")
                return
    
    # Show quick-add widget if portfolio is empty
    if not analyzer.portfolio:
        st.info("Your portfolio is empty. Use the Portfolio Management page to add stocks.")
        return
    
    try:
        # Get summary report
        summary, positions = analyzer.generate_summary_report()
        
        # Display metrics in configurable grid
        if config['metrics']['enabled']:
            cols = st.columns(config['metrics']['columns'])
            metrics = [
                ("Portfolio Value", summary['Portfolio Value'], "Total market value of holdings"),
                ("Total Return", summary['Total Return'], "Overall portfolio return"),
                ("Sharpe Ratio", summary['Sharpe Ratio'], "Risk-adjusted return metric"),
                ("Alpha (Annual)", summary.get('Alpha (Annual)', 'N/A'), "Excess return vs benchmark"),
                ("Beta", summary.get('Beta', 'N/A'), "Market sensitivity"),
                ("Volatility", summary['Volatility (Annual)'], "Annual price variation"),
                ("Max Drawdown", summary['Maximum Drawdown'], "Largest historical decline"),
                ("VaR (95%)", summary['Value at Risk (95%)'], "Potential daily loss")
            ]
            
            for i, (label, value, tooltip) in enumerate(metrics):
                col_idx = i % config['metrics']['columns']
                with cols[col_idx]:
                    st.metric(label, value, help=tooltip)
        
        # Display charts in configurable layout
        if config['charts']['enabled']:
            st.subheader("📈 Performance Analysis")
            
            # Returns chart
            st.plotly_chart(analyzer.plot_cumulative_returns(), use_container_width=True)
            
            # Two-column charts
            col1, col2 = st.columns(2)
            with col1:
                composition_fig = analyzer.plot_portfolio_composition()
                if composition_fig:
                    st.plotly_chart(composition_fig, use_container_width=True)
            
            with col2:
                st.plotly_chart(analyzer.plot_drawdowns(), use_container_width=True)
            
            # Monthly returns heatmap
            st.plotly_chart(analyzer.plot_monthly_returns_heatmap(), use_container_width=True)
        
        # Holdings table with enhanced formatting
        st.subheader("📊 Current Holdings")
        if positions:
            holdings_df = pd.DataFrame({
                'Asset': list(positions.keys()),
                'Value': list(positions.values())
            }).sort_values('Value', ascending=False)
            
            holdings_df['Allocation'] = holdings_df['Value'] / holdings_df['Value'].sum() * 100
            holdings_df['Value'] = holdings_df['Value'].map('${:,.2f}'.format)
            holdings_df['Allocation'] = holdings_df['Allocation'].map('{:.1f}%'.format)
            
            st.dataframe(holdings_df, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Portfolio Data",
                    analyzer.data.to_csv(),
                    "portfolio_data.csv",
                    "text/csv"
                )
            with col2:
                if st.button("📊 Generate PDF Report"):
                    report_pdf = generate_portfolio_report(analyzer)
                    st.download_button(
                        "📄 Download PDF Report",
                        report_pdf,
                        "portfolio_report.pdf"
                    )
    
    except Exception as e:
        st.error(f"Error displaying dashboard: {str(e)}")
        st.info("Try refreshing the data or check your portfolio configuration.")

def display_stock_analysis(analyzer):
    st.title("📈 Stock Analysis")
    
    with st.expander("📚 Technical Analysis Guide"):
        st.markdown("""
        Understanding the charts:
        
        **Price Analysis:**
        - **Candlesticks**: Show open, high, low, close prices
        - **Moving Averages**: Trend indicators (20, 50, 200 days)
        
        **Technical Indicators:**
        - **RSI**: Momentum indicator (>70 overbought, <30 oversold)
        - **MACD**: Trend strength and direction
        - **Volume**: Trading activity confirmation
        
        **Statistics:**
        - Price trends and momentum
        - Volume analysis
        - Volatility patterns
        """)
    
    # Always show stock search regardless of portfolio state
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        ticker = st.selectbox("Select Stock", get_available_tickers())
        if ticker:
            info = get_ticker_info(ticker)
            st.write(f"**{info['name']}** - ${info['price']:.2f}")
    
    with search_col2:
        shares = st.number_input("Number of Shares", min_value=0.1, value=1.0)
        if st.button("Add to Portfolio"):
            analyzer.add_stock(ticker, shares)
            st.success(f"Added {shares} shares of {ticker}")
            st.rerun()
    
    st.markdown("---")
    
    # Show message if portfolio is empty
    if not analyzer.portfolio:
        st.info("Add stocks above to see detailed analysis.")
        return
    
    # Stock Analysis
    ticker = st.selectbox("Select Stock for Analysis", list(analyzer.portfolio.keys()))
    
    if ticker:
        # Fetch data and calculate indicators
        with st.spinner("Loading data..."):
            analyzer.fetch_data()
            if analyzer.data is not None:
                df = get_technical_indicators(analyzer.data, ticker)
                
                # Create tabs for different analysis views
                tab1, tab2, tab3 = st.tabs(["Price Chart", "Technical Indicators", "Statistics"])
                
                with tab1:
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=ticker
                    ))
                    
                    # Add moving averages
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        name="20 SMA",
                        line=dict(color='orange')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        name="50 SMA",
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} Price Chart",
                        yaxis_title="Price",
                        xaxis_title="Date",
                        height=600,
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Technical indicators plot
                    fig2 = make_subplots(rows=2, cols=1, 
                                        subplot_titles=("RSI", "MACD"),
                                        vertical_spacing=0.2)
                    
                    # RSI
                    fig2.add_trace(
                        go.Scatter(x=df.index, y=df['RSI'], name="RSI"),
                        row=1, col=1
                    )
                    
                    # Add RSI overbought/oversold lines
                    fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                    fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                    
                    # MACD
                    fig2.add_trace(
                        go.Scatter(x=df.index, y=df['MACD'], name="MACD"),
                        row=2, col=1
                    )
                    fig2.add_trace(
                        go.Scatter(x=df.index, y=df['Signal_Line'], name="Signal"),
                        row=2, col=1
                    )
                    
                    fig2.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # Summary statistics
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        current_price = df['Close'].iloc[-1]
                        prev_close = df['Close'].iloc[-2]
                        price_change = current_price - prev_close
                        price_change_pct = (price_change / prev_close) * 100
                        
                        # Use 'normal' instead of color names
                        st.metric(
                            "Current Price",
                            f"${current_price:.2f}",
                            f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                            delta_color="normal" if price_change >= 0 else "inverse"
                        )
                        
                        st.metric("52-Week High", f"${df['High'].max():.2f}")
                        st.metric("52-Week Low", f"${df['Low'].min():.2f}")
                    
                    with stats_col2:
                        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                        st.metric("Avg Volume (30D)", f"{df['Volume'].rolling(30).mean().iloc[-1]:,.0f}")
                        
                        # Calculate volatility
                        returns = df['Close'].pct_change()
                        volatility = returns.std() * np.sqrt(252)
                        st.metric("Annual Volatility", f"{volatility:.2%}")

def display_risk_assessment(analyzer):
    st.title("🎯 Risk Assessment")
    
    with st.expander("📚 Understanding Risk Metrics"):
        st.markdown("""
        Key risk metrics explained:
        
        - **Value at Risk (VaR)**: Maximum potential loss at a given confidence level
        - **Annual Volatility**: Yearly price variation measure
        - **Maximum Drawdown**: Largest peak-to-trough decline
        - **Beta**: Market sensitivity (> 1 means more volatile than market)
        - **Sharpe Ratio**: Risk-adjusted return metric (higher is better)
        """)
    
    # Add stock search functionality
    st.subheader("Search Stock")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_ticker = st.text_input("Enter Stock Ticker", key="risk_search_ticker")
    with search_col2:
        search_button = st.button("Analyze", key="risk_search_button")
    
    if search_button and search_ticker:
        ts_analyzer, data = search_stock(search_ticker.upper())
        if ts_analyzer and data is not None:
            display_risk_metrics(ts_analyzer, search_ticker.upper())
    
    st.markdown("---")
    
    # Value at Risk
    st.subheader("Value at Risk (VaR)")
    var_data = analyzer.calculate_value_at_risk()
    
    col1, col2 = st.columns(2)
    col1.metric("1-Day Value at Risk (95%)", f"${abs(var_data['var_amount']):,.2f}")
    col2.metric("VaR as % of Portfolio", f"{abs(var_data['var_pct'] * 100):.2f}%")
    
    # Drawdown Analysis
    st.subheader("Drawdown Analysis")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(analyzer.plot_drawdowns(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Metrics
    st.subheader("Risk Metrics")
    col1, col2, col3 = st.columns(3)
    
    volatility = analyzer.calculate_volatility() * 100
    max_drawdown = analyzer.calculate_max_drawdown() * 100
    sharpe = analyzer.calculate_sharpe_ratio()
    
    col1.metric("Annual Volatility", f"{volatility:.2f}%")
    col2.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    if analyzer.benchmark:
        try:
            beta = analyzer.calculate_beta()
            alpha = analyzer.calculate_alpha() * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Beta", f"{beta:.2f}")
            col2.metric("Alpha (Annual)", f"{alpha:.2f}%")
        except:
            st.warning("Unable to calculate beta/alpha. Check benchmark data.")

def display_news_sentiment(analyzer):
    st.title("📰 News & Sentiment Analysis")
    
    with st.expander("📚 Understanding News Analysis"):
        st.markdown("""
        This page analyzes news and market sentiment:
        
        **Sentiment Categories:**
        - 📈 **Positive**: Bullish news, potential upside
        - 📉 **Negative**: Bearish news, potential risks
        - ➖ **Neutral**: Balanced or factual reporting
        
        **Filtering Options:**
        - **Time Period**: Recent to historical news
        - **Categories**: Earnings, Market, Company, Analysis
        - **Sentiment**: Filter by market sentiment
        
        **Relevance Score:**
        Indicates how relevant the news is to your portfolio (0-10 scale)
        """)
    
    # Add stock search functionality
    st.subheader("Search Stock News")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_ticker = st.text_input("Enter Stock Ticker", key="news_search_ticker")
    with search_col2:
        search_button = st.button("Search News", key="news_search_button")
    
    if search_button and search_ticker:
        search_ticker = search_ticker.upper()
        try:
            news_api = NewsAPI()
            articles = news_api.get_stock_news(
                symbols=[search_ticker],
                days=7
            )
            if articles:
                st.success(f"Found news for {search_ticker}")
                for article in articles:
                    render_news_card(article)
            else:
                st.info(f"No recent news found for {search_ticker}")
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
    
    st.markdown("---")
    
    # Enhanced filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_back = st.selectbox(
            "Time Period",
            options=[1, 7, 30],
            index=1,  # Default to 7 days
            format_func=lambda x: f"Last {x} days",
            key="news_days"
        )
    
    with col2:
        category = st.selectbox(
            "News Category",
            options=[None, "earnings", "market", "company", "analysis"],
            format_func=lambda x: x.title() if x else "All Categories",
            key="news_category"
        )
    
    with col3:
        sentiment = st.selectbox(
            "Sentiment",
            options=[None, "positive", "negative", "neutral"],
            format_func=lambda x: x.title() if x else "All Sentiments",
            key="news_sentiment"
        )
    
    # Stock selection with multi-select option
    selected_stocks = st.multiselect(
        "Select Stocks",
        options=list(analyzer.portfolio.keys()),
        default=list(analyzer.portfolio.keys())[:3],
        key="news_stocks"
    )
    
    if selected_stocks:
        try:
            news_api = NewsAPI()
            articles = news_api.get_stock_news(
                symbols=selected_stocks,
                days=days_back,
                category=category,
                sentiment=sentiment
            )
            
            if articles:
                # Sort options
                sort_by = st.selectbox(
                    "Sort By",
                    options=["relevance", "date", "sentiment"],
                    key="news_sort"
                )
                
                if sort_by == "date":
                    articles.sort(key=lambda x: x['publishedAt'], reverse=True)
                elif sort_by == "sentiment":
                    articles.sort(key=lambda x: x['sentiment']['score'], reverse=True)
                
                # Display articles
                for article in articles:
                    render_news_card(article)
            else:
                st.info("No news articles found matching the selected criteria.")
                
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
    else:
        st.warning("Please select at least one stock to view news.")

def display_ai_insights(analyzer):
    st.title("🤖 AI-Driven Portfolio Insights")
    
    with st.expander("📚 Understanding AI Insights"):
        st.markdown("""
        This page uses advanced machine learning models to analyze your portfolio:
        
        **Models Used:**
        - **Prophet**: For trend forecasting and seasonality analysis
        - **XGBoost**: For complex pattern recognition
        - **LSTM**: For sequence learning and future price prediction
        
        **Key Components:**
        - **Multi-Model Forecast**: Combines predictions from multiple models
        - **Market Regimes**: Identifies different market states (Bull/Bear/Sideways)
        - **Volatility Analysis**: Predicts future market volatility
        """)
    
    # Initialize data and analyzers
    if analyzer.data is None:
        try:
            analyzer.fetch_data()
        except:
            # Load default AAPL data
            default_ticker = "AAPL"
            data = yf.download(default_ticker, start="2023-01-01", end=datetime.now().strftime('%Y-%m-%d'))
            analyzer.data = data
            if default_ticker not in analyzer.portfolio:
                analyzer.add_stock(default_ticker, 1)
    
    if not analyzer.portfolio:
        st.warning("Portfolio is empty. Add stocks to see AI insights.")
        return
    
    # Create main layout
    overview_tab, forecast_tab, regime_tab = st.tabs(["📊 Overview", "🔮 Forecasting", "📈 Market Analysis"])
    
    with overview_tab:
        # Portfolio summary and key metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Total Assets", len(analyzer.portfolio))
            
        with metrics_col2:
            summary, _ = analyzer.generate_summary_report()
            st.metric("Portfolio Value", summary['Portfolio Value'])
            
        with metrics_col3:
            st.metric("Active Models", "3 Models")
        
        # Performance overview
        st.subheader("Portfolio Performance")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(analyzer.plot_cumulative_returns(), use_container_width=True)
        with col2:
            st.plotly_chart(analyzer.plot_portfolio_composition(), use_container_width=True)
        
        # Add Factor Analysis
        st.subheader("📊 Factor Analysis")
        factor_analysis = FactorAnalysis()
        returns, _ = analyzer.calculate_returns()  # Get returns first
        
        if factor_analysis.load_factor_data(analyzer.start_date, analyzer.end_date):
            factor_results = factor_analysis.analyze_portfolio(returns)
            
            if factor_results:
                cols = st.columns(4)
                cols[0].metric("Alpha (Annual)", f"{factor_results['alpha']*100:.2f}%")
                cols[1].metric("Beta", f"{factor_results['beta']:.2f}")
                cols[2].metric("R-Squared", f"{factor_results['r_squared']:.2f}")
                cols[3].metric("P-Value", f"{factor_results['p_value']:.3f}")
        
        # Add Risk Analysis with existing returns
        st.subheader("🎯 Risk Analysis")
        risk_model = RiskModel()
        
        # Calculate diversification score
        corr_matrix = analyzer.calculate_correlation_matrix()
        div_score = risk_model.get_diversification_score(corr_matrix)
        
        # Run stress tests with already calculated returns
        scenarios = {
            'Market Crash': -0.20,
            'Recession': -0.10,
            'Rate Hike': -0.05,
            'Recovery': 0.10
        }
        
        stress_results = risk_model.run_stress_test(returns, scenarios)
        
        cols = st.columns(2)
        cols[0].metric("Diversification Score", f"{div_score:.2f}")
        
        # Show stress test results
        cols[1].write("Stress Test Results:")
        for scenario, impact in stress_results.items():
            cols[1].text(f"{scenario}: {impact*100:.1f}%")
    
    with forecast_tab:
        # Stock selection and forecast settings
        st.subheader("🎯 Price & Volatility Forecasting")
        settings_col1, settings_col2, settings_col3 = st.columns([2, 1, 1])
        
        with settings_col1:
            ticker = st.selectbox("Select Stock for Analysis", list(analyzer.portfolio.keys()))
        with settings_col2:
            forecast_days = st.slider("Forecast Horizon", 30, 252, 75)
        with settings_col3:
            confidence = st.select_slider("Confidence Level", options=[80, 85, 90, 95, 99], value=95)
        
        if ticker:
            ts_analyzer = TimeSeriesAnalyzer(analyzer.data)
            
            # Show forecasts in tabs
            model_tab1, model_tab2, model_tab3 = st.tabs(["Multi-Model", "SARIMA", "Volatility"])
            
            with model_tab1:
                try:
                    forecast_fig = ts_analyzer.plot_multi_forecast(ticker, days=forecast_days)
                    if forecast_fig is not None:
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Add model comparison metrics
                        metric_cols = st.columns(3)
                        forecasts = ts_analyzer.forecast_models(ticker, days=forecast_days)
                        if forecasts:
                            for i, (model, data) in enumerate(forecasts.items()):
                                with metric_cols[i]:
                                    mean_forecast = data['mean'].mean()
                                    st.metric(
                                        f"{model} Forecast",
                                        f"${mean_forecast:.2f}",
                                        f"{((mean_forecast/data['mean'].iloc[0] - 1) * 100):.1f}%"
                                    )
                except Exception as e:
                    st.error(f"Error generating price forecast: {str(e)}")
            
            with model_tab2:
                try:
                    sarima_fig, forecast_mean, conf_int, metrics = ts_analyzer.forecast_price(ticker, days=forecast_days)
                    if sarima_fig is not None:
                        st.plotly_chart(sarima_fig, use_container_width=True)
                        
                        # Show SARIMA metrics
                        metric_cols = st.columns(4)
                        metric_cols[0].metric("Mean Forecast", f"${metrics['forecast_mean']:.2f}")
                        metric_cols[1].metric("Std Dev", f"${metrics['forecast_std']:.2f}")
                        metric_cols[2].metric("Confidence Width", f"${metrics['confidence_width']:.2f}")
                        metric_cols[3].metric("Trend", metrics['forecast_trend'])
                except Exception as e:
                    st.error(f"Error generating SARIMA forecast: {str(e)}")
            
            with model_tab3:
                try:
                    vol_fig, volatility = ts_analyzer.volatility_forecast(ticker, days=forecast_days)
                    if vol_fig is not None:
                        st.plotly_chart(vol_fig, use_container_width=True)
                        
                        # Add volatility metrics
                        vol_cols = st.columns(3)
                        vol_cols[0].metric("Current Volatility", f"{volatility[0]*100:.1f}%")
                        vol_cols[1].metric("Mean Forecast", f"{np.mean(volatility)*100:.1f}%")
                        vol_cols[2].metric("Max Forecast", f"{np.max(volatility)*100:.1f}%")
                except Exception as e:
                    st.error(f"Error generating volatility forecast: {str(e)}")
    
    with regime_tab:
        st.subheader("📊 Market Regime Analysis")
        if ticker:
            try:
                regimes = ts_analyzer.detect_regimes(ticker)
                if regimes:
                    # Enhanced regime visualization
                    regime_col1, regime_col2 = st.columns([2, 1])
                    
                    with regime_col1:
                        # Create regime timeline plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=regimes['regimes'].index,
                            y=regimes['volatility'],
                            name='Volatility',
                            line=dict(color='blue')
                        ))
                        
                        # Add regime backgrounds
                        for regime in regimes['regimes'].unique():
                            mask = regimes['regimes'] == regime
                            fig.add_trace(go.Scatter(
                                x=regimes['regimes'][mask].index,
                                y=regimes['volatility'][mask],
                                fill='tonexty',
                                name=regime,
                                line=dict(width=0)
                            ))
                            
                        fig.update_layout(
                            title="Regime Timeline",
                            height=400,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with regime_col2:
                        # Current regime metrics
                        st.metric("Current Regime", regimes['regimes'].iloc[-1])
                        st.metric("Market Trend", regimes['trend'].iloc[-1])
                        st.metric("Volatility", f"{regimes['volatility'].iloc[-1]:.2%}")
                        
                        # Regime distribution
                        regime_counts = regimes['regimes'].value_counts()
                        st.bar_chart(regime_counts)
                        
            except Exception as e:
                st.error(f"Error analyzing market regimes: {str(e)}")
    
    # Display AI-generated insights at the bottom
    st.markdown("---")
    insights = PortfolioInsights(analyzer).generate_insights()
    categories = {
        'performance': '📈 Performance',
        'risk': '⚠️ Risk',
        'diversification': '🔄 Diversification',
        'momentum': '🔋 Momentum',
        'pattern': '🎯 Pattern'
    }
    
    for category, icon in categories.items():
        insights_filtered = [i for i in insights if i['type'] == category]
        if insights_filtered:
            st.subheader(icon + " " + category.title() + " Insights")
            for insight in sorted(insights_filtered, key=lambda x: x['priority']):
                with st.expander(f"Priority {insight['priority']}: {insight['title']}"):
                    st.write(insight['description'])
                    if 'metrics' in insight:
                        st.json(insight['metrics'])

def generate_insights_report(insights, analyzer):
    """Generate a formatted insights report"""
    report = []
    report.append("AI-Driven Portfolio Insights Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add portfolio summary
    summary, _ = analyzer.generate_summary_report()
    report.append("Portfolio Summary:")
    for key, value in summary.items():
        report.append(f"{key}: {value}")
    
    # Add insights
    report.append("\nDetailed Insights:")
    for insight in insights:
        report.append(f"\n{insight['type'].upper()} - {insight['title']}")
        report.append(f"Priority Level: {insight['priority']}")
        report.append(f"Analysis: {insight['description']}")
    
    return "\n".join(report)

def get_glossary():
    """Return financial terms glossary"""
    return {
        "Portfolio Metrics": {
            "Sharpe Ratio": "Measures risk-adjusted returns. Higher values indicate better risk-adjusted performance.",
            "Alpha": "Excess return compared to the benchmark after adjusting for market risk.",
            "Beta": "Measure of market sensitivity. Beta of 1 means the portfolio moves with the market.",
            "Value at Risk (VaR)": "Statistical measure of potential loss over a specific time period.",
            "Maximum Drawdown": "Largest peak-to-trough decline in portfolio value.",
            "Volatility": "Degree of variation in returns, measured by standard deviation."
        },
        "Technical Indicators": {
            "SMA": "Simple Moving Average - Average price over a specific period.",
            "RSI": "Relative Strength Index - Momentum indicator measuring speed of price changes.",
            "MACD": "Moving Average Convergence Divergence - Trend-following momentum indicator."
        }
    }

def search_stock(ticker):
    """Search for stock data without adding to portfolio"""
    try:
        stock = yf.Ticker(ticker)
        # Get historical data with all required columns
        data = stock.history(period="1y")
        
        if data.empty:
            st.error(f"No data available for {ticker}")
            return None, None
            
        # Convert to proper format for TimeSeriesAnalyzer
        df = pd.DataFrame({
            'Open': data['Open'],
            'High': data['High'],
            'Low': data['Low'],
            'Close': data['Close'],
            'Volume': data['Volume']
        }, index=data.index)
        
        ts_analyzer = TimeSeriesAnalyzer(df)
        return ts_analyzer, df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def display_risk_metrics(ts_analyzer, ticker):
    """Display risk metrics for a given stock"""
    try:
        if ts_analyzer is None or ts_analyzer.data is None:
            st.error(f"No data available for {ticker}")
            return
            
        # Calculate and display basic risk metrics
        data = ts_analyzer.data
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 2:
            st.error("Insufficient data for analysis")
            return
            
        volatility = returns.std() * np.sqrt(252)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Volatility", f"{volatility*100:.2f}%")
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        col2.metric("Maximum Drawdown", f"{max_dd*100:.2f}%")
        
        # Calculate Value at Risk
        var_95 = returns.quantile(0.05)
        col3.metric("Daily VaR (95%)", f"{abs(var_95*100):.2f}%")
        
        # Plot volatility forecast if available
        vol_forecast = ts_analyzer.volatility_forecast(ticker)
        if vol_forecast[0] is not None:
            st.plotly_chart(vol_forecast[0], use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")

def display_stock_insights(ts_analyzer, ticker):
    """Display AI insights for a given stock"""
    try:
        if ts_analyzer is None or ts_analyzer.data is None:
            st.error(f"No data available for {ticker}")
            return
            
        # Show forecasts
        forecast_days = st.slider(
            "Forecast Horizon (Days)", 
            30, 252, 75,
            key=f"forecast_slider_{ticker}"
        )
        
        # Get forecasts with error handling
        forecast_result = ts_analyzer.plot_multi_forecast(ticker, days=forecast_days)
        if forecast_result is not None:
            st.plotly_chart(forecast_result, use_container_width=True)
        
        # Show regime analysis with error handling
        regimes = ts_analyzer.detect_regimes(ticker)
        if regimes is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Regime", regimes['regimes'].iloc[-1])
                st.metric("Market Trend", regimes['trend'].iloc[-1])
            with col2:
                st.metric("Volatility", f"{regimes['volatility'].iloc[-1]:.2%}")
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")

def generate_portfolio_report(analyzer):
    """Generate PDF report for portfolio analysis"""
    from utils.report_generator import PortfolioReport
    report = PortfolioReport(analyzer)
    return report.generate_pdf()

# Add glossary to sidebar
def main():
    st.set_page_config(
        page_title="Financial Portfolio Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set dark mode in session state
    st.session_state.dark_mode = True
    
    # Apply custom CSS
    st.markdown(get_css(), unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Navigation Menu",
            [
                "📊 Portfolio Dashboard", 
                "💼 Portfolio Management", 
                "📈 Stock Analysis",
                "🎯 Risk Assessment",
                "📰 News & Sentiment",
                "🤖 AI Insights"
            ]
        )
        
        # Financial terms in single expander
        with st.expander("📖 Financial Terms"):
            glossary = get_glossary()
            for category, terms in glossary.items():
                st.markdown(f"**{category}**")
                for term, definition in terms.items():
                    st.markdown(f"- **{term}**: _{definition}_")
                st.markdown("---")
                
    # Initialize portfolio analyzer
    if 'portfolio_analyzer' not in st.session_state:
        st.session_state.portfolio_analyzer = PortfolioAnalyzer()
        
        # Initialize demo portfolio if needed
        if 'initialized' not in st.session_state:
            analyzer = st.session_state.portfolio_analyzer
            analyzer.set_timeframe("2023-01-01")
            analyzer.set_benchmark("^GSPC")
            
            # Add demo stocks
            analyzer.add_stock("AAPL", 15)
            analyzer.add_stock("MSFT", 13)
            analyzer.add_stock("AMZN", 7)
            analyzer.add_stock("GOOGL", 10)
            
            st.session_state.initialized = True
    
    analyzer = st.session_state.portfolio_analyzer
    
    # Display appropriate page based on selection
    if page == "📊 Portfolio Dashboard":
        display_dashboard(analyzer)
    elif page == "💼 Portfolio Management":
        display_portfolio_management(analyzer)
    elif page == "📈 Stock Analysis":
        display_stock_analysis(analyzer)
    elif page == "🎯 Risk Assessment":
        display_risk_assessment(analyzer)
    elif page == "📰 News & Sentiment":
        display_news_sentiment(analyzer)
    elif page == "🤖 AI Insights":
        display_ai_insights(analyzer)

if __name__ == "__main__":
    main()