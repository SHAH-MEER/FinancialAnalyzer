import numpy as np
from scipy.optimize import minimize
import pandas as pd
from .news_api import NewsAPI

class BenchmarkManager:
    def __init__(self):
        self.custom_benchmarks = {}
    
    def create_custom_benchmark(self, name, components):
        """
        components = {
            '^GSPC': 0.6,  # 60% S&P 500
            '^IXIC': 0.4   # 40% NASDAQ
        }
        """
        if sum(components.values()) != 1:
            raise ValueError("Weights must sum to 1")
        self.custom_benchmarks[name] = components

def optimize_portfolio(returns_data, risk_free_rate=0.02, target_return=None, include_news=False):
    """
    Optimize portfolio using Modern Portfolio Theory
    """
    try:
        def portfolio_volatility(weights):
            cov_matrix = returns_data.cov() * 252
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def portfolio_return(weights):
            mean_returns = returns_data.mean() * 252
            return np.sum(mean_returns * weights)

        def sharpe_ratio(weights):
            mean_returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            portfolio_ret = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_ret - risk_free_rate) / portfolio_vol

        num_assets = len(returns_data.columns)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: portfolio_return(x) - target_return
            })

        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1/num_assets] * num_assets)

        result = minimize(
            sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimized_weights = result.x
        opt_portfolio_ret = portfolio_return(optimized_weights)
        opt_portfolio_vol = portfolio_volatility(optimized_weights)
        opt_sharpe = -result.fun  # Convert back to positive

        result_dict = {
            'weights': optimized_weights,
            'metrics': {
                'return': opt_portfolio_ret,
                'volatility': opt_portfolio_vol,
                'sharpe': opt_sharpe
            },
            'assets': list(returns_data.columns)
        }

        if include_news:
            news_api = NewsAPI()
            result_dict['news'] = news_api.get_stock_news(returns_data.columns)
            result_dict['market_news'] = news_api.get_market_news()

        return result_dict

    except Exception as e:
        print(f"Advanced optimization failed: {str(e)}. Using fallback strategy.")
        
        # Fallback to simple equal-weight portfolio
        num_assets = len(returns_data.columns)
        equal_weights = np.array([1/num_assets] * num_assets)
        
        return {
            'weights': equal_weights,
            'metrics': {
                'return': np.sum(returns_data.mean() * equal_weights) * 252,
                'volatility': np.sqrt(np.dot(equal_weights.T, np.dot(returns_data.cov() * 252, equal_weights))),
                'method': 'equal_weight_fallback'
            },
            'assets': list(returns_data.columns)
        }

def export_results(optimization_results, format='json'):
    """Export optimization results to various formats"""
    if format == 'json':
        return pd.DataFrame({
            'Asset': optimization_results['assets'],
            'Weight': optimization_results['weights'],
        }).to_json(orient='records')
    elif format == 'csv':
        return pd.DataFrame({
            'Asset': optimization_results['assets'],
            'Weight': optimization_results['weights'],
            'Portfolio Return': optimization_results['metrics']['return'],
            'Portfolio Volatility': optimization_results['metrics']['volatility'],
            'Sharpe Ratio': optimization_results['metrics']['sharpe']
        }).to_csv(index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def suggest_rebalancing(current_weights, target_weights, threshold=0.05):
    """
    Returns suggested trades to rebalance portfolio
    """
    suggestions = {}
    for ticker in current_weights:
        diff = target_weights.get(ticker, 0) - current_weights[ticker]
        if abs(diff) > threshold:
            suggestions[ticker] = diff
    return suggestions