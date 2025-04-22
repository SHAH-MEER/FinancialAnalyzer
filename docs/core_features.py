def get_metric_tooltips():
    """Core metrics documentation"""
    return {
        "Portfolio Metrics": {
            "Portfolio Value": "Total current market value of all holdings in your portfolio",
            "Sharpe Ratio": "Risk-adjusted return measure. Higher is better. Above 1 is good, above 2 is very good.",
            "Alpha": "Excess return compared to benchmark. Positive values indicate outperformance.",
            "Beta": "Market sensitivity. 1 = moves with market, <1 = less volatile, >1 = more volatile.",
            "Value at Risk": "Maximum expected loss within confidence interval. Lower is better.",
            "Volatility": "Price variation over time. Lower means more stable returns.",
            "Maximum Drawdown": "Largest peak-to-trough decline. Shows worst historical loss."
        },
        "Technical Analysis": {
            "Moving Averages": "Trend indicators showing average price over specific periods",
            "RSI": "Relative Strength Index (30=oversold, 70=overbought)",
            "MACD": "Moving Average Convergence/Divergence, shows momentum"
        },
        "Portfolio Management": {
            "Adding Stocks": "Enter ticker symbol and number of shares to add to portfolio",
            "Removing Stocks": "Select stock and specify shares to remove from portfolio",
            "Updating Data": "Click 'Update Data' to refresh prices and calculations"
        }
    }

def get_quick_start_guide():
    """Core quick start documentation"""
    return """
    ### Quick Start Guide
    
    1. **Portfolio Management**
       - Use the stock selector to add stocks
       - Enter number of shares owned
       - Set purchase date (optional)
    
    2. **Dashboard**
       - View portfolio performance
       - Compare against benchmark
       - Monitor key metrics
    
    3. **Analysis**
       - Check individual stock performance
       - View technical indicators
       - Track risk metrics
    
    4. **Tips**
       - Hover over metrics for explanations
       - Use date range selector for specific periods
       - Download data for offline analysis
    """

def get_feature_help():
    """Core feature documentation"""
    return {
        "dashboard": {
            "title": "Portfolio Dashboard",
            "description": "Central view of portfolio performance and metrics",
            "key_features": [
                "Performance tracking",
                "Risk metrics",
                "Holdings breakdown",
                "Benchmark comparison"
            ]
        },
        "portfolio": {
            "title": "Portfolio Management",
            "description": "Add, remove, and manage portfolio holdings",
            "key_features": [
                "Stock addition/removal",
                "Position sizing",
                "Current holdings view",
                "Portfolio composition"
            ]
        },
        "analysis": {
            "title": "Stock Analysis",
            "description": "Detailed analysis of individual stocks",
            "key_features": [
                "Price charts",
                "Technical indicators",
                "Trading volumes",
                "Performance metrics"
            ]
        }
    }

def get_dashboard_guide():
    return """
    ## Portfolio Dashboard Guide
    
    ### Overview Metrics
    - **Portfolio Value**: Total current market value of all holdings
    - **Total Return**: Overall portfolio return since inception
    - **Sharpe Ratio**: Risk-adjusted return metric (higher is better)
    - **Alpha**: Excess return compared to benchmark
    - **Beta**: Market sensitivity measure
    - **Value at Risk**: Potential loss estimate
    
    ### Performance Charts
    - **Cumulative Returns**: Portfolio vs benchmark performance over time
    - **Portfolio Composition**: Current asset allocation breakdown
    - **Drawdowns**: Historical portfolio value declines
    - **Monthly Returns**: Calendar heatmap of returns
    
    ### Features
    - Use the date range selector to adjust the analysis period
    - Choose different benchmarks for comparison
    - Export data and reports for offline analysis
    - Customize dashboard layout and metrics display
    
    ### Tips
    - Click on chart legends to show/hide specific data
    - Hover over metrics for detailed explanations
    - Use the export button to download portfolio data
    - Save custom dashboard layouts for future sessions
    """