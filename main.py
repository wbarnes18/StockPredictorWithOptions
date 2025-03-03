# Stock Predictor of the Third Age - Updated Main.py (March 03, 2025)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.trend import SMAIndicator, MACD, ADXIndicator, IchimokuIndicator, PSARIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, MFIIndicator
from ta.trend import CCIIndicator
import streamlit as st
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import json
from io import StringIO
import matplotlib.pyplot as plt

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('stock_predictor.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (ticker TEXT, granularity TEXT, prediction REAL, actual REAL, accuracy REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Fetch stock data
def fetch_stock_data(ticker):
    end_date = datetime.datetime.now()
    # Fetch only 7 days of minute-level data due to yfinance limitation
    start_date_minute = end_date - timedelta(days=7)
    # Fetch 60 days for hour and day data
    start_date_hour_day = end_date - timedelta(days=60)
    
    # Fetch data for different granularities
    try:
        minute_data = yf.download(ticker, start=start_date_minute, end=end_date, interval="1m")
    except Exception as e:
        st.error(f"Error fetching minute-level data: {str(e)}")
        minute_data = pd.DataFrame()  # Return empty DataFrame on failure
    
    try:
        hour_data = yf.download(ticker, start=start_date_hour_day, end=end_date, interval="1h")
    except Exception as e:
        st.error(f"Error fetching hour-level data: {str(e)}")
        hour_data = pd.DataFrame()  # Return empty DataFrame on failure
    
    try:
        day_data = yf.download(ticker, start=start_date_hour_day, end=end_date, interval="1d")
    except Exception as e:
        st.error(f"Error fetching day-level data: {str(e)}")
        day_data = pd.DataFrame()  # Return empty DataFrame on failure
    
    return minute_data, hour_data, day_data

# Calculate technical indicators
def calculate_indicators(minute_data, hour_data, day_data):
    indicators = {"minute": None, "hour": None, "day": None}
    
    for data, granularity in [(minute_data, "minute"), (hour_data, "hour"), (day_data, "day")]:
        # Skip if data is empty
        if data.empty:
            continue
            
        # Check if 'Close' column exists and has enough data
        if 'Close' not in data or len(data['Close']) < 20:  # Need at least 20 data points for SMA_20
            continue
            
        ind = {}
        try:
            # SMA
            ind["SMA_20"] = SMAIndicator(data['Close'], window=20).sma_indicator().iloc[-1]
            # MACD
            macd = MACD(data['Close'])
            ind["MACD"] = macd.macd().iloc[-1]
            ind["MACD_Signal"] = macd.macd_signal().iloc[-1]
            # ADX
            ind["ADX"] = ADXIndicator(data['High'], data['Low'], data['Close']).adx().iloc[-1]
            # RSI
            ind["RSI"] = RSIIndicator(data['Close']).rsi().iloc[-1]
            # Stochastic Oscillator
            ind["Stoch"] = StochasticOscillator(data['High'], data['Low'], data['Close']).stoch().iloc[-1]
            # Bollinger Bands
            bb = BollingerBands(data['Close'])
            ind["BB_upper"] = bb.bollinger_hband().iloc[-1]
            ind["BB_lower"] = bb.bollinger_lband().iloc[-1]
            # ATR
            ind["ATR"] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range().iloc[-1]
            # Volume Weighted Average Price
            ind["VWAP"] = VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price().iloc[-1]
            # OBV
            ind["OBV"] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume().iloc[-1]
            
            indicators[granularity] = ind
        except Exception as e:
            st.error(f"Error calculating {granularity} indicators: {str(e)}")
            indicators[granularity] = None
    
    return indicators

# Sentiment analysis (placeholder)
def analyze_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    # In a real app, you'd fetch news or social media data here
    # For now, we'll use a dummy score
    return 0.5  # Placeholder sentiment score

# Predict stock price using XGBoost
def predict_stock(minute_data, hour_data, day_data):
    scaler = MinMaxScaler()
    
    def prepare_data(data):
        if data.empty:
            return np.array([]), np.array([])
        scaled_data = scaler.fit_transform(data[['Close']].values)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y)

    # Minute prediction
    minute_pred = None
    if not minute_data.empty:
        X_minute, y_minute = prepare_data(minute_data)
        if len(X_minute) > 0:
            model_minute = xgb.XGBRegressor(n_estimators=100)
            model_minute.fit(X_minute.reshape(X_minute.shape[0], -1), y_minute.ravel())
            minute_pred = model_minute.predict(X_minute[-1].reshape(1, -1))
            minute_pred = scaler.inverse_transform(minute_pred.reshape(-1, 1))[0][0]
        else:
            minute_pred = minute_data['Close'].iloc[-1]

    # Hour prediction
    hour_pred = None
    if not hour_data.empty:
        X_hour, y_hour = prepare_data(hour_data)
        if len(X_hour) > 0:
            model_hour = xgb.XGBRegressor(n_estimators=100)
            model_hour.fit(X_hour.reshape(X_hour.shape[0], -1), y_hour.ravel())
            hour_pred = model_hour.predict(X_hour[-1].reshape(1, -1))
            hour_pred = scaler.inverse_transform(hour_pred.reshape(-1, 1))[0][0]
        else:
            hour_pred = hour_data['Close'].iloc[-1]

    # Day prediction
    day_pred = None
    if not day_data.empty:
        X_day, y_day = prepare_data(day_data)
        if len(X_day) > 0:
            model_day = RandomForestRegressor(n_estimators=100)
            model_day.fit(X_day.reshape(X_day.shape[0], -1), y_day.ravel())
            day_pred = model_day.predict(X_day[-1].reshape(1, -1))
            day_pred = scaler.inverse_transform(day_pred.reshape(-1, 1))[0][0]
        else:
            day_pred = day_data['Close'].iloc[-1]

    return minute_pred, hour_pred, day_pred

# Backtest model
def backtest_model(ticker, minute_data, hour_data, day_data, minute_pred, hour_pred, day_pred):
    actual_minute = minute_data['Close'].iloc[-1] if not minute_data.empty else minute_pred
    actual_hour = hour_data['Close'].iloc[-1] if not hour_data.empty else hour_pred
    actual_day = day_data['Close'].iloc[-1] if not day_data.empty else day_pred

    minute_accuracy = 1 - abs(minute_pred - actual_minute) / actual_minute if actual_minute and minute_pred and actual_minute != 0 else 0
    hour_accuracy = 1 - abs(hour_pred - actual_hour) / actual_hour if actual_hour and hour_pred and actual_hour != 0 else 0
    day_accuracy = 1 - abs(day_pred - actual_day) / actual_day if actual_day and day_pred and actual_day != 0 else 0

    # Store in database
    conn = sqlite3.connect('stock_predictor.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
              (ticker, "minute", minute_pred, actual_minute, minute_accuracy, timestamp))
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
              (ticker, "hour", hour_pred, actual_hour, hour_accuracy, timestamp))
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
              (ticker, "day", day_pred, actual_day, day_accuracy, timestamp))
    conn.commit()
    conn.close()

    return minute_accuracy, hour_accuracy, day_accuracy

# Self-improvement loop using XGBoost
def train_with_self_improvement(minute_data, hour_data, day_data):
    minute_accuracy, hour_accuracy, day_accuracy = 0.0, 0.0, 0.0
    
    scaler = MinMaxScaler()
    
    def prepare_data(data):
        if data.empty:
            return np.array([]), np.array([])
        scaled_data = scaler.fit_transform(data[['Close']].values)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y)

    # Train for each granularity
    for data, granularity in [(minute_data, "minute"), (hour_data, "hour"), (day_data, "day")]:
        X, y = prepare_data(data)
        if len(X) == 0:
            continue
        
        # Use XGBoost for self-improvement
        model = xgb.XGBRegressor(n_estimators=100)
        
        # Iterative training until accuracy threshold
        accuracy = 0
        while accuracy < 0.75:  # Target accuracy
            model.fit(X.reshape(X.shape[0], -1), y.ravel())
            pred = model.predict(X.reshape(X.shape[0], -1))
            accuracy = r2_score(y, pred)
            if granularity == "minute":
                minute_accuracy = accuracy
            elif granularity == "hour":
                hour_accuracy = accuracy
            else:
                day_accuracy = accuracy
    
    return minute_accuracy, hour_accuracy, day_accuracy

# Main Streamlit app
def main():
    st.write("App is running!")  # Debug statement to confirm app is rendering
    
    # Initialize session state for accuracies
    if "minute_accuracy" not in st.session_state:
        st.session_state.minute_accuracy = 0.0
    if "hour_accuracy" not in st.session_state:
        st.session_state.hour_accuracy = 0.0
    if "day_accuracy" not in st.session_state:
        st.session_state.day_accuracy = 0.0
    if "backtest_history" not in st.session_state:
        st.session_state.backtest_history = []

    # Create tabs
    tabs = st.tabs([
        "Stock Prediction",
        "Technical Indicators",
        "Sentiment Analysis",
        "Backtesting Results",
    ])

    # Stock Prediction tab
    with tabs[0]:
        st.header("Stock Prediction")
        ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):", "AAPL").upper()
        if st.button("Predict"):
            try:
                minute_data, hour_data, day_data = fetch_stock_data(ticker)
                # Check if all data is empty
                if minute_data.empty and hour_data.empty and day_data.empty:
                    st.error("Failed to fetch stock data for all granularities. The market may be closed, or the ticker symbol may be invalid. Please try again during market hours (9:30 AM - 4:00 PM ET, Monday-Friday).")
                else:
                    minute_pred, hour_pred, day_pred = predict_stock(minute_data, hour_data, day_data)
                    if minute_pred is not None:
                        st.write(f"Minute Prediction: {minute_pred:.2f}")
                    else:
                        st.write("Minute Prediction: Not available (data unavailable, possibly due to market closure)")
                    if hour_pred is not None:
                        st.write(f"Hour Prediction: {hour_pred:.2f}")
                    else:
                        st.write("Hour Prediction: Not available (data unavailable, possibly due to market closure)")
                    if day_pred is not None:
                        st.write(f"Day Prediction: {day_pred:.2f}")
                    else:
                        st.write("Day Prediction: Not available (data unavailable, possibly due to market closure)")

                    # Backtest the predictions
                    minute_acc, hour_acc, day_acc = backtest_model(ticker, minute_data, hour_data, day_data, minute_pred, hour_pred, day_pred)
                    st.session_state.backtest_history.append({
                        "Ticker": ticker,
                        "Minute Prediction": minute_pred,
                        "Minute Actual": minute_data['Close'].iloc[-1] if not minute_data.empty else None,
                        "Minute Accuracy": minute_acc,
                        "Hour Prediction": hour_pred,
                        "Hour Actual": hour_data['Close'].iloc[-1] if not hour_data.empty else None,
                        "Hour Accuracy": hour_acc,
                        "Day Prediction": day_pred,
                        "Day Actual": day_data['Close'].iloc[-1] if not day_data.empty else None,
                        "Day Accuracy": day_acc,
                        "Timestamp": datetime.datetime.now().isoformat()
                    })
            except Exception as e:
                st.error(f"Error fetching data or making predictions: {str(e)}")

    # Technical Indicators tab
    with tabs[1]:
        st.header("Technical Indicators")
        if ticker:
            try:
                minute_data, hour_data, day_data = fetch_stock_data(ticker)
                indicators = calculate_indicators(minute_data, hour_data, day_data)
                if indicators["minute"]:
                    st.write("Minute Indicators:", indicators["minute"])
                else:
                    st.write("Minute Indicators: Not available (data unavailable, possibly due to market closure)")
                if indicators["hour"]:
                    st.write("Hour Indicators:", indicators["hour"])
                else:
                    st.write("Hour Indicators: Not available (data unavailable, possibly due to market closure)")
                if indicators["day"]:
                    st.write("Day Indicators:", indicators["day"])
                else:
                    st.write("Day Indicators: Not available (data unavailable, possibly due to market closure)")
            except Exception as e:
                st.error(f"Error calculating indicators: {str(e)}")
        else:
            st.write("Enter a ticker symbol in the Stock Prediction tab to see indicators.")

    # Sentiment Analysis tab
    with tabs[2]:
        st.header("Sentiment Analysis")
        if ticker:
            try:
                sentiment_score = analyze_sentiment(ticker)
                st.write(f"Sentiment Score for {ticker}: {sentiment_score:.2f}")
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
        else:
            st.write("Enter a ticker symbol in the Stock Prediction tab to see sentiment analysis.")

    # Backtesting Results tab
    with tabs[3]:
        st.header("Backtesting Results")
        if st.session_state.backtest_history:
            backtest_history_df = pd.DataFrame(st.session_state.backtest_history)
            st.dataframe(backtest_history_df)
        else:
            st.write("No backtesting results available yet. Run a backtest to see results.")

    # Self-improvement loop
    target_accuracy = 0.75
    if any(
        acc < target_accuracy
        for acc in [
            st.session_state.minute_accuracy,
            st.session_state.hour_accuracy,
            st.session_state.day_accuracy,
        ]
    ):
        try:
            minute_data, hour_data, day_data = fetch_stock_data(ticker)
            # Only run self-improvement if at least one dataset is non-empty
            if not (minute_data.empty and hour_data.empty and day_data.empty):
                st.session_state.minute_accuracy, st.session_state.hour_accuracy, st.session_state.day_accuracy = train_with_self_improvement(
                    minute_data, hour_data, day_data
                )
            else:
                st.warning("Self-improvement loop skipped: No data available for training (possibly due to market closure).")
        except Exception as e:
            st.error(f"Error during self-improvement loop: {str(e)}")

if __name__ == "__main__":
    init_db()
    main()