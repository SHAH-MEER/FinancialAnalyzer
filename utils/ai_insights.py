import numpy as np
from typing import List, Dict, Any
from scipy import stats

class PortfolioInsights:
    """AI-driven portfolio analysis and insights generation"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.insights = []
    
    def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate comprehensive AI-driven portfolio insights"""
        self.insights = []
        
        # Performance Analysis
        self._analyze_performance()
        # Risk Analysis
        self._analyze_risk()
        # Diversification Analysis
        self._analyze_diversification()
        # Momentum Analysis
        self._analyze_momentum()
        # Pattern Recognition
        self._detect_patterns()
        
        return sorted(self.insights, key=lambda x: x['priority'], reverse=True)
    
    def _analyze_performance(self):
        """Analyze portfolio performance metrics"""
        try:
            returns, bench_returns = self.analyzer.calculate_returns()
            
            # Alpha analysis
            if bench_returns is not None:
                alpha = self.analyzer.calculate_alpha()
                if abs(alpha) > 0.05:  # 5% threshold
                    self.insights.append({
                        'type': 'performance',
                        'title': 'Significant Alpha Detected',
                        'description': f"Portfolio is {'outperforming' if alpha > 0 else 'underperforming'} "
                                     f"the benchmark by {abs(alpha*100):.1f}% annually.",
                        'priority': 1
                    })
            
            # Return distribution analysis
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            if abs(skew) > 0.5:
                self.insights.append({
                    'type': 'performance',
                    'title': 'Return Distribution Asymmetry',
                    'description': f"Returns show {'positive' if skew > 0 else 'negative'} skewness, "
                                 f"suggesting {'higher upside' if skew > 0 else 'downside'} potential.",
                    'priority': 2
                })
        except:
            pass
    
    def _analyze_risk(self):
        """Analyze portfolio risk metrics"""
        try:
            volatility = self.analyzer.calculate_volatility()
            max_dd = self.analyzer.calculate_max_drawdown()
            var_data = self.analyzer.calculate_value_at_risk()
            
            # Risk concentration analysis
            if volatility > 0.25:  # 25% annual volatility threshold
                self.insights.append({
                    'type': 'risk',
                    'title': 'High Volatility Alert',
                    'description': f"Portfolio volatility of {volatility*100:.1f}% exceeds typical market levels. "
                                 f"Consider risk reduction strategies.",
                    'priority': 1
                })
            
            # Drawdown analysis
            if abs(max_dd) > 0.15:  # 15% drawdown threshold
                self.insights.append({
                    'type': 'risk',
                    'title': 'Significant Drawdown',
                    'description': f"Maximum drawdown of {abs(max_dd*100):.1f}% detected. "
                                 f"Review position sizing and stop-loss strategies.",
                    'priority': 1
                })
        except:
            pass
    
    def _analyze_diversification(self):
        """Analyze portfolio diversification"""
        try:
            corr_matrix = self.analyzer.calculate_correlation_matrix()
            
            # High correlation analysis
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i,j] > 0.8:  # 80% correlation threshold
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )
            
            if high_corr_pairs:
                self.insights.append({
                    'type': 'diversification',
                    'title': 'High Correlation Detected',
                    'description': f"High correlation between: " + 
                                 ", ".join([f"{a}-{b}" for a,b in high_corr_pairs[:3]]) +
                                 ". Consider diversifying across different sectors.",
                    'priority': 2
                })
        except:
            pass
    
    def _analyze_momentum(self):
        """Analyze price momentum and trends"""
        try:
            for ticker in self.analyzer.portfolio:
                # Get price data
                prices = self.analyzer.data['Close'][ticker]
                returns = prices.pct_change()
                
                # Calculate momentum indicators
                sma_20 = prices.rolling(20).mean()
                sma_50 = prices.rolling(50).mean()
                
                # Trend analysis
                if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
                    self.insights.append({
                        'type': 'momentum',
                        'title': f'Golden Cross - {ticker}',
                        'description': f"{ticker} shows bullish momentum with 20-day MA crossing above 50-day MA.",
                        'priority': 2
                    })
        except:
            pass
    
    def _detect_patterns(self):
        """Detect technical patterns in price data"""
        try:
            for ticker in self.analyzer.portfolio:
                prices = self.analyzer.data['Close'][ticker]
                returns = prices.pct_change()
                
                # Volatility pattern
                vol = returns.rolling(20).std()
                if vol.iloc[-1] > 2 * vol.iloc[-20]:
                    self.insights.append({
                        'type': 'pattern',
                        'title': f'Volatility Breakout - {ticker}',
                        'description': f"Unusual price volatility detected in {ticker}. "
                                     f"Monitor position closely.",
                        'priority': 1
                    })
        except:
            pass