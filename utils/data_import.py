import pandas as pd
import streamlit as st
from datetime import datetime

def validate_ticker(ticker):
    """Placeholder for ticker validation logic"""
    # Implement actual ticker validation logic here
    return True

def validate_portfolio_data(df):
    """Enhanced portfolio data validation"""
    try:
        required_columns = ['Ticker', 'Shares', 'Purchase Date']
        if not all(col in df.columns for col in required_columns):
            return False, "Missing required columns"
        
        # Validate data types
        if not pd.to_numeric(df['Shares'], errors='coerce').notnull().all():
            return False, "Invalid share quantities"
        
        # Validate dates
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
        if df['Purchase Date'].isnull().any():
            return False, "Invalid dates found"
            
        # Validate tickers
        invalid_tickers = [ticker for ticker in df['Ticker'] if not validate_ticker(ticker)]
        if invalid_tickers:
            return False, f"Invalid tickers found: {', '.join(invalid_tickers)}"
        
        return True, df
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def import_portfolio_from_csv(file):
    """Process CSV with enhanced validation"""
    try:
        df = pd.read_csv(file)
        is_valid, result = validate_portfolio_data(df)
        if is_valid:
            return {'success': True, 'data': result.to_dict('records')}
        return {'success': False, 'error': result}
    except Exception as e:
        return {'success': False, 'error': f"Import error: {str(e)}"}

def export_portfolio_to_csv(portfolio):
    """Export portfolio to CSV format"""
    data = []
    for ticker, details in portfolio.items():
        data.append({
            'Ticker': ticker,
            'Shares': details['shares'],
            'Purchase Date': details.get('purchase_date', datetime.now().strftime('%Y-%m-%d'))
        })
    return pd.DataFrame(data)
