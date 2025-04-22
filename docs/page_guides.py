def get_dashboard_guide():
    return """
    ### Dashboard Guide 📊
    
    This dashboard provides a comprehensive overview of your portfolio performance.
    
    **Key Features:**
    1. **Portfolio Summary Metrics**
        - View total portfolio value and key performance indicators
        - Hover over metrics for detailed explanations
        - Compare performance against selected benchmark
    
    2. **Interactive Charts**
        - **Cumulative Returns:** Shows total return over time vs benchmark
        - **Portfolio Composition:** Pie chart showing asset allocation
        - **Drawdowns:** Visualizes periods of decline from peaks
        - **Monthly Returns:** Heatmap of monthly performance
    
    **Tips:**
    - Use date range selector to analyze different time periods
    - Click on the pie chart segments to view individual stock performance
    - Hover over the charts for tooltips with exact values
    - Download performance data for offline analysis
    
    - Click legend items to show/hide data series
    - Hover over charts for detailed data points
    - Download raw data for offline analysis
    """

def get_portfolio_management_guide():
    return """
    ## Portfolio Management Guide
    
    ### Adding Stocks
    1. **Custom Stock Entry**
       - Enter any valid stock ticker symbol in the "Add Custom Stock" section
       - Click "Validate & Add" to verify and add the stock to available options
    
    2. **Adding to Portfolio**
       - Select a stock from the dropdown menu
       - Enter the number of shares
       - Select the purchase date
       - Click "Add to Portfolio" to include in your portfolio
    
    ### Managing Portfolio
    - View current holdings in the table showing:
      - Ticker symbol
      - Number of shares
      - Purchase date
      - Current price
      - Market value
    
    ### Removing Stocks
    - Select the stock to remove from your portfolio
    - Enter the number of shares to remove
    - Use partial removal to reduce position size
    - Click "Remove from Portfolio" to confirm
    
    ### Tips
    - Use the current price and market info to make informed decisions
    - Track sector diversification through portfolio analytics
    - Monitor dividend yields for income-generating stocks
    - Keep purchase dates for tax planning purposes
    """

def get_stock_analysis_guide():
    return """
    ### Stock Analysis Guide 📈
    
    Detailed technical analysis and price charts for individual stocks.
    
    **Chart Types:**
    1. **Price Chart**
        - Candlestick patterns show daily price movement
        - Moving averages indicate trends (20-day and 50-day)
        - Volume bars show trading activity
    
    2. **Technical Indicators**
        - RSI (Relative Strength Index): Overbought >70, Oversold <30
        - MACD: Trend and momentum indicator
        - Moving Averages: Trend direction and support/resistance
    
    **Interactive Features:**
    - Zoom: Click and drag on chart
    - Pan: Hold shift while dragging
    - Reset: Double click
    - Hover for exact values
    """

def get_risk_assessment_guide():
    return """
    ### Risk Assessment Guide 🎯
    
    Understand and analyze portfolio risk metrics.
    
    **Key Risk Measures:**
    1. **Value at Risk (VaR)**
        - Potential loss in normal market conditions
        - 95% confidence level
        - Both dollar and percentage terms
    
    2. **Risk Metrics**
        - Volatility: Price variation over time
        - Beta: Market sensitivity
        - Sharpe Ratio: Risk-adjusted returns
        - Maximum Drawdown: Worst historical loss
    
    **Using This Section:**
    - Monitor risk levels regularly
    - Compare against benchmark
    - Use metrics to optimize portfolio allocation
    """

def get_news_guide():
    return """
    ## News & Sentiment Analysis Guide
    
    ### Filtering News
    1. **Time Period**
       - Select news coverage period (1-90 days)
       - Recent news provides actionable insights
       - Historical news helps understand trends
    
    2. **Categories**
       - Earnings: Financial results and forecasts
       - Market: General market analysis and trends
       - Company: Corporate news and developments
       - Analysis: Expert analysis and recommendations
    
    3. **Sentiment Filter**
       - Positive: Favorable news and upgrades
       - Negative: Concerning news and downgrades
       - Neutral: Factual or balanced coverage
    
    ### Stock Selection
    - Select multiple stocks for comprehensive coverage
    - Default shows top holdings in your portfolio
    - Add custom tickers for expanded news monitoring
    
    ### Sorting and Organization
    - Sort by:
      - Relevance: Most important news first
      - Date: Latest news first
      - Sentiment: Group by market sentiment
    
    ### Understanding Indicators
    - 📈 Positive sentiment
    - 📉 Negative sentiment
    - ➖ Neutral sentiment
    - 🔗 Related stocks mentioned
    - 📊 Relevance score to your portfolio
    
    ### Tips
    - Use sentiment analysis for trading decisions
    - Track news patterns across your holdings
    - Monitor sector-wide news impact
    - Save important articles for later reference
    """