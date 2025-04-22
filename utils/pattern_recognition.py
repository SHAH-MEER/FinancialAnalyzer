import numpy as np
import pandas as pd
from scipy import signal

class PatternRecognition:
    @staticmethod
    def find_double_bottom(prices, window=20, tolerance=0.02):
        """Detect double bottom patterns"""
        # Find local minima
        peaks, _ = signal.find_peaks(-prices, distance=window)
        patterns = []
        
        for i in range(len(peaks)-1):
            price1 = prices[peaks[i]]
            price2 = prices[peaks[i+1]]
            
            # Check if prices are within tolerance
            if abs(price1 - price2) / price1 < tolerance:
                patterns.append({
                    'type': 'Double Bottom',
                    'first_bottom': peaks[i],
                    'second_bottom': peaks[i+1],
                    'price_level': (price1 + price2) / 2
                })
        
        return patterns
    
    @staticmethod
    def detect_trend_reversal(prices, window=20):
        """Detect potential trend reversals"""
        ma_short = prices.rolling(window=window).mean()
        ma_long = prices.rolling(window=window*2).mean()
        
        # Detect crossovers
        crossovers = pd.Series(0, index=prices.index)
        crossovers[ma_short > ma_long] = 1
        crossovers[ma_short < ma_long] = -1
        
        # Find signal changes
        signals = crossovers.diff().fillna(0)
        
        return signals[signals != 0]

    @staticmethod
    def momentum_signals(prices, rsi_window=14, rsi_thresholds=(30, 70)):
        """Generate momentum-based trading signals"""
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[rsi < rsi_thresholds[0]] = 1  # Oversold
        signals[rsi > rsi_thresholds[1]] = -1  # Overbought
        
        return signals, rsi
