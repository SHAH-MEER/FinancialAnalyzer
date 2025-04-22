import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

class AnomalyDetector:
    def __init__(self, data):
        self.data = data
        
    def detect_price_anomalies(self, ticker, contamination=0.1):
        """Detect price anomalies using Isolation Forest"""
        # Prepare features
        df = self.data[['Close', 'Volume']][ticker].copy()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Scale features
        scaler = StandardScaler()
        features = scaler.fit_transform(df.dropna())
        
        # Fit Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        anomalies = clf.fit_predict(features)
        
        # Create visualization
        fig = go.Figure()
        
        # Normal points
        normal_idx = anomalies == 1
        fig.add_trace(go.Scatter(
            x=df.index[len(df)-len(anomalies):][normal_idx],
            y=df['Close'][len(df)-len(anomalies):][normal_idx],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=5)
        ))
        
        # Anomaly points
        anomaly_idx = anomalies == -1
        fig.add_trace(go.Scatter(
            x=df.index[len(df)-len(anomalies):][anomaly_idx],
            y=df['Close'][len(df)-len(anomalies):][anomaly_idx],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig.update_layout(
            title=f"{ticker} Price Anomalies",
            yaxis_title='Price',
            xaxis_title='Date',
            showlegend=True
        )
        
        return fig, df[anomaly_idx]
