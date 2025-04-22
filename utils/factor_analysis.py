import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf

class FactorAnalysis:
    def __init__(self):
        self.factors = {
            'Market': '^GSPC',  # S&P 500
            'SMB': None,        # Small Minus Big
            'HML': None,        # High Minus Low
            'Momentum': None    # Momentum Factor
        }
        self.factor_data = None
        
    def load_factor_data(self, start_date, end_date):
        """Load market factor data"""
        try:
            market = yf.download(self.factors['Market'], start=start_date, end=end_date)['Adj Close']
            self.factor_data = pd.DataFrame({
                'Market': market.pct_change().dropna()
            })
            return True
        except:
            return False
    
    def analyze_portfolio(self, returns):
        """Run factor analysis on portfolio returns"""
        if self.factor_data is None:
            return None
            
        # Calculate market factor exposure
        X = self.factor_data['Market']
        y = returns
        
        # Run regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        
        return {
            'alpha': intercept * 252,  # Annualized alpha
            'beta': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }

class RiskModel:
    @staticmethod
    def run_stress_test(portfolio_returns, scenarios):
        """Run stress test scenarios"""
        results = {}
        for scenario, shock in scenarios.items():
            stressed_return = portfolio_returns.mean() + shock
            results[scenario] = stressed_return
        return results
    
    @staticmethod
    def calculate_risk_contribution(weights, cov_matrix):
        """Calculate risk contribution of each asset"""
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / port_vol
        risk_contrib = np.multiply(weights, marginal_contrib)
        return risk_contrib / port_vol
    
    @staticmethod
    def get_diversification_score(correlation_matrix):
        """Calculate portfolio diversification score"""
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        # Normalized diversification score (0-1)
        return 1 - (max(eigenvalues) - 1) / (len(correlation_matrix) - 1)
