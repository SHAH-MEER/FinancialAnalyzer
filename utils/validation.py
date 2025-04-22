import yfinance as yf

def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker exists and has data available
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        
        # Try to get current price - this will fail if ticker is invalid
        info = stock.info
        if 'regularMarketPrice' not in info:
            return False
            
        return True
    except:
        return False
