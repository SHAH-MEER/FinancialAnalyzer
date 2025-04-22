import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from functools import lru_cache
import hashlib
from utils.alpha_vantage import AlphaVantageAPI
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from utils.chart_themes import CHART_THEMES

class TimeSeriesAnalyzer:
    def __init__(self, data):
        """Initialize with data validation"""
        if data is not None and not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("Data must be a pandas DataFrame or Series")
        self.data = data

    def _prepare_series(self, ticker):
        """Prepare time series with proper index"""
        if self.data is None:
            return None
            
        try:
            if isinstance(self.data, pd.DataFrame):
                # Handle multi-index DataFrame
                if isinstance(self.data.columns, pd.MultiIndex):
                    if ('Close', ticker) in self.data.columns:
                        series = self.data['Close'][ticker].copy()
                    else:
                        st.warning(f"No data found for {ticker}")
                        return None
                # Handle single-index DataFrame
                elif 'Close' in self.data.columns:
                    if ticker in self.data.columns:
                        series = self.data['Close'].copy()
                    else:
                        st.warning(f"No data found for {ticker}")
                        return None
                else:
                    st.warning("Invalid data format")
                    return None
            else:
                series = self.data.copy()

            # Validate data
            if series.empty:
                st.warning(f"No data available for {ticker}")
                return None

            # Ensure datetime index without timezone
            series.index = pd.DatetimeIndex(series.index).tz_localize(None)
            
            # Fill missing values and drop duplicates
            series = series.asfreq('B').ffill().drop_duplicates()

            return series

        except Exception as e:
            st.warning(f"Error preparing data: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def detect_regimes(_self, ticker):
        """Detect market regimes using statistical tests"""
        series = _self._prepare_series(ticker)
        if series is None:
            return None
            
        try:
            returns = series.pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.rolling(window=21).std() * np.sqrt(252)
            
            # Define regimes
            regimes = pd.Series(index=volatility.index, dtype=str)
            regimes[volatility <= volatility.quantile(0.33)] = 'Low Volatility'
            regimes[volatility > volatility.quantile(0.67)] = 'High Volatility'
            regimes[regimes.isna()] = 'Normal'
            
            # Detect trends
            ma_50 = series.rolling(window=50).mean()
            ma_200 = series.rolling(window=200).mean()
            
            trend = pd.Series(index=series.index, dtype=str)
            trend[ma_50 > ma_200] = 'Bullish'
            trend[ma_50 < ma_200] = 'Bearish'
            trend[trend.isna()] = 'Neutral'
            
            return {
                'regimes': regimes,
                'trend': trend,
                'volatility': volatility
            }
            
        except Exception as e:
            st.error(f"Error detecting regimes: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def detect_structural_breaks(_self, ticker):
        """Detect structural breaks using rolling ADF test"""
        series = _self._prepare_series(ticker)
        window_size = 252  # 1 year
        
        # Rolling ADF test
        adf_stats = []
        p_values = []
        
        for i in range(window_size, len(series)):
            window = series.iloc[i-window_size:i]
            result = adfuller(window)
            adf_stats.append(result[0])
            p_values.append(result[1])
        
        breaks = pd.Series(p_values, index=series.index[window_size:])
        significant_breaks = breaks[breaks < 0.05]
        
        return significant_breaks

    @st.cache_data(ttl=3600)
    def forecast_models(_self, ticker, days=252):
        """Generate forecasts using multiple models"""
        series = _self._prepare_series(ticker)
        if series is None or len(series) < 30:
            return None
        
        try:
            forecasts = {}
            last_date = series.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
            
            # Prepare data for all models
            data = series.values.reshape(-1, 1)
            train_size = int(len(data) * 0.8)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # 1. Prophet Model
            df_prophet = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.15
            )
            prophet_model.fit(df_prophet)
            
            future_df = pd.DataFrame({'ds': forecast_dates})
            prophet_forecast = prophet_model.predict(future_df)
            
            forecasts['Prophet'] = {
                'mean': pd.Series(prophet_forecast['yhat'].values, index=forecast_dates),
                'conf_int': pd.DataFrame({
                    'lower': prophet_forecast['yhat_lower'].values,
                    'upper': prophet_forecast['yhat_upper'].values
                }, index=forecast_dates)
            }
            
            # 2. XGBoost Model
            n_features = 20
            X, y = [], []
            for i in range(n_features, len(scaled_data)):
                X.append(scaled_data[i-n_features:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1
            )
            xgb_model.fit(X_train, y_train)
            
            # Generate XGBoost predictions
            last_sequence = scaled_data[-n_features:]
            xgb_preds = []
            for _ in range(days):
                next_pred = xgb_model.predict([last_sequence.flatten()])
                xgb_preds.append(next_pred[0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred
            
            # Scale back predictions
            xgb_preds = np.array(xgb_preds).reshape(-1, 1)
            xgb_preds = scaler.inverse_transform(xgb_preds)
            
            forecasts['XGBoost'] = {
                'mean': pd.Series(xgb_preds.flatten(), index=forecast_dates),
                'conf_int': pd.DataFrame({
                    'lower': xgb_preds.flatten() * 0.95,
                    'upper': xgb_preds.flatten() * 1.05
                }, index=forecast_dates)
            }
            
            # 3. LSTM Model
            X_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(n_features, 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_lstm, y_train, epochs=10, batch_size=32, verbose=0)
            
            # Generate LSTM predictions
            last_sequence = scaled_data[-n_features:]
            lstm_preds = []
            for _ in range(days):
                next_pred = lstm_model.predict(last_sequence.reshape(1, n_features, 1), verbose=0)
                lstm_preds.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred
            
            # Scale back predictions
            lstm_preds = np.array(lstm_preds).reshape(-1, 1)
            lstm_preds = scaler.inverse_transform(lstm_preds)
            
            forecasts['LSTM'] = {
                'mean': pd.Series(lstm_preds.flatten(), index=forecast_dates),
                'conf_int': pd.DataFrame({
                    'lower': lstm_preds.flatten() * 0.95,
                    'upper': lstm_preds.flatten() * 1.05
                }, index=forecast_dates)
            }
            
            return forecasts
            
        except Exception as e:
            st.warning(f"Forecast generation failed: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def plot_multi_forecast(_self, ticker, days=252):
        """Plot multiple forecast models comparison"""
        try:
            series = _self._prepare_series(ticker)
            if series is None:
                return go.Figure()
                
            forecasts = _self.forecast_models(ticker, days)
            if forecasts is None:
                return go.Figure()
            
            fig = go.Figure()
            
            # Enhanced color scheme
            colors = {
                'Historical': {'line': 'rgb(59, 130, 246)'},
                'Prophet': {'line': 'rgb(16, 185, 129)', 'fill': 'rgba(16, 185, 129, 0.1)'},
                'XGBoost': {'line': 'rgb(245, 158, 11)', 'fill': 'rgba(245, 158, 11, 0.1)'},
                'LSTM': {'line': 'rgb(236, 72, 153)', 'fill': 'rgba(236, 72, 153, 0.1)'}
            }
            
            # Plot historical data
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                name='Historical',
                line=dict(color=colors['Historical']['line'], width=2)
            ))
            
            # Plot each model's forecast
            for model_name, forecast in forecasts.items():
                if model_name not in colors:
                    continue
                
                # Plot mean forecast
                fig.add_trace(go.Scatter(
                    x=forecast['mean'].index,
                    y=forecast['mean'].values,
                    name=f'{model_name}',
                    line=dict(color=colors[model_name]['line'], dash='dash')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast['conf_int'].index,
                    y=forecast['conf_int']['upper'],
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast['conf_int'].index,
                    y=forecast['conf_int']['lower'],
                    fill='tonexty',
                    fillcolor=colors[model_name]['fill'],
                    line=dict(width=0),
                    name=f'{model_name} CI'
                ))
            
            # Enhanced layout
            fig.update_layout(
                title={
                    'text': f'{ticker} Price Forecast ({days} days)',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(0,0,0,0.5)'
                ),
                height=600,
                hovermode='x unified',
                margin=dict(t=100, l=50, r=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error generating forecast plot: {str(e)}")
            return go.Figure()

    def decompose_series(self, ticker, period=252):
        """Decompose time series into trend, seasonal, and residual"""
        series = self._prepare_series(ticker)
        decomposition = seasonal_decompose(series, period=period)
        
        fig = make_subplots(rows=4, cols=1,
                           subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        
        # Plot components
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.resid, name='Residual'), row=4, col=1)
        
        fig.update_layout(height=800, title_text=f"{ticker} Time Series Decomposition")
        return fig
    
    @st.cache_data(ttl=3600)
    def forecast_price(_self, ticker, days=252):  # Default to 1 year
        """Generate price forecast using SARIMA"""
        series = _self._prepare_series(ticker)
        
        if series is None:
            return None, None, None, None
        
        # Create date index for forecast
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
        
        # Fit SARIMA model with enforced frequency
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Generate forecast with proper index
        forecast = results.get_forecast(steps=days)
        forecast_mean = pd.Series(forecast.predicted_mean.values, index=forecast_dates)
        conf_int = pd.DataFrame(forecast.conf_int().values, index=forecast_dates)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=series.index, 
            y=series.values,
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_mean.index, 
            y=forecast_mean.values,
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int.iloc[:, 0],
            fill=None,
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Lower Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int.iloc[:, 1],
            fill='tonexty',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Upper Bound'
        ))
        
        fig.update_layout(
            title=f"{ticker} Price Forecast ({days} trading days)",
            yaxis_title='Price',
            xaxis_title='Date',
            hovermode='x unified'
        )
        
        # Add forecast metrics
        metrics = {
            'forecast_mean': forecast_mean.mean(),
            'forecast_std': forecast_mean.std(),
            'confidence_width': (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).mean(),
            'forecast_trend': 'Upward' if forecast_mean.iloc[-1] > forecast_mean.iloc[0] else 'Downward'
        }
        
        return fig, forecast_mean, conf_int, metrics  # Now returning 4 values
    
    @st.cache_data(ttl=3600)
    def volatility_forecast(_self, ticker, days=252):
        """Generate volatility forecast using GARCH"""
        try:
            series = _self._prepare_series(ticker)
            if series is None:
                return None, None
                
            # Calculate returns correctly
            returns = series.pct_change().dropna()
            
            # Calculate historical volatility (annualized)
            hist_vol = returns.rolling(window=21).std() * np.sqrt(252)
            
            # Fit GARCH model
            model = arch_model(
                returns,
                vol='Garch',
                p=1, q=1,
                mean='Constant',
                dist='normal'
            )
            
            try:
                results = model.fit(disp=False)
                forecast = results.forecast(horizon=days)
                
                # Convert to annualized volatility
                volatility = np.sqrt(forecast.variance.mean(axis=1)) * np.sqrt(252)
                
                # Create forecast dates
                last_date = returns.index[-1]
                forecast_dates = pd.date_range(last_date, periods=days+1, freq='B')[1:]
                
                # Create figure
                fig = go.Figure()
                
                # Plot historical volatility
                fig.add_trace(go.Scatter(
                    x=hist_vol.index,
                    y=hist_vol * 100,  # Convert to percentage
                    name='Historical',
                    line=dict(color='blue', width=1)
                ))
                
                # Plot forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=volatility * 100,  # Convert to percentage
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence bands
                std_bands = np.sqrt(forecast.variance.std(axis=1)) * np.sqrt(252)
                upper_band = (volatility + 2 * std_bands) * 100
                lower_band = np.maximum((volatility - 2 * std_bands), 0) * 100
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=upper_band,
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=lower_band,
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(width=0),
                    name='95% Confidence'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} Annualized Volatility Forecast",
                    yaxis_title='Volatility (%)',
                    xaxis_title='Date',
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified',
                    height=400,
                    yaxis=dict(
                        tickformat='.1f',
                        ticksuffix='%',
                        range=[0, max(upper_band.max() * 1.1, hist_vol.max() * 100 * 1.1)]
                    )
                )
                
                return fig, volatility
                
            except Exception as e:
                st.error(f"Error fitting GARCH model: {str(e)}")
                return None, None
                
        except Exception as e:
            st.error(f"Error generating volatility forecast: {str(e)}")
            return None, None

    def plot_regime_transitions(self, ticker):
        """Plot regime transitions and characteristics"""
        regimes = self.detect_regimes(ticker)
        if not regimes:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Regime Timeline',
                'Regime Distribution',
                'Volatility by Regime',
                'Trend Analysis'
            )
        )
        
        # Regime Timeline
        for regime in regimes['regimes'].unique():
            mask = regimes['regimes'] == regime
            fig.add_trace(
                go.Scatter(
                    x=regimes['regimes'][mask].index,
                    y=[1]*mask.sum(),
                    name=regime,
                    mode='markers',
                    marker=dict(size=10)
                ),
                row=1, col=1
            )
        
        # Regime Distribution
        regime_counts = regimes['regimes'].value_counts()
        fig.add_trace(
            go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                name='Distribution'
            ),
            row=1, col=2
        )
        
        # Volatility by Regime
        vol_by_regime = {
            regime: regimes['volatility'][regimes['regimes'] == regime].mean()
            for regime in regimes['regimes'].unique()
        }
        fig.add_trace(
            go.Bar(
                x=list(vol_by_regime.keys()),
                y=list(vol_by_regime.values()),
                name='Avg Volatility'
            ),
            row=2, col=1
        )
        
        # Trend Analysis
        trend_counts = regimes['trend'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=trend_counts.index,
                values=trend_counts.values,
                name='Trends'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

    def plot_performance_metrics(self, ticker):
        """Plot key performance metrics"""
        series = self._prepare_series(ticker)
        if series is None:
            return None
            
        returns = series.pct_change().dropna()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rolling Returns',
                'Return Distribution',
                'Rolling Volatility',
                'QQ Plot'
            )
        )
        
        # Rolling Returns
        rolling_returns = returns.rolling(window=21).mean() * 252
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns.values,
                name='Rolling Returns'
            ),
            row=1, col=1
        )
        
        # Return Distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name='Returns Dist',
                nbinsx=50
            ),
            row=1, col=2
        )
        
        # Rolling Volatility
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Rolling Vol'
            ),
            row=2, col=1
        )
        
        # QQ Plot
        from scipy import stats
        qq = stats.probplot(returns.values)
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name='QQ Plot'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
