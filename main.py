# Test update via patch file - Added by Grok on March 04, 2025
# Stock Predictor of the Third Age - Updated Main.py (March 03, 2025)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ta.trend import SMAIndicator, MACD, ADXIndicator, IchimokuIndicator, PSARIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, \
    ForceIndexIndicator, MFIIndicator
from ta.trend import CCIIndicator
import streamlit as st
from datetime import datetime, timedelta
import mplfinance as mpf
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import json
from io import StringIO
import matplotlib.pyplot as plt

# Initialize SQLite database
conn = sqlite3.connect('stock_predictor.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS historical_data (
    ticker TEXT PRIMARY KEY,
    minutes_data TEXT,
    hours_data TEXT,
    days_data TEXT
)''')

c.execute('''CREATE TABLE IF NOT EXISTS model_configs (
    ticker TEXT PRIMARY KEY,
    model_type TEXT,
    selected_indicators TEXT,
    use_sentiment INTEGER
)''')

c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    predicted_at TEXT,
    horizon_datetime TEXT,
    predicted_price FLOAT,
    actual_price FLOAT,
    metrics TEXT,
    FOREIGN KEY(ticker) REFERENCES historical_data(ticker)
)''')

c.execute('''CREATE TABLE IF NOT EXISTS options_recommendations (
    prediction_id INTEGER,
    type TEXT,
    strike FLOAT,
    buy_price FLOAT,
    sell_price FLOAT,
    exit_time TEXT,
    predicted_profit FLOAT,
    actual_profit FLOAT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
)''')

conn.commit()

# Initialize session state
if 'prediction_steps' not in st.session_state:
    st.session_state.prediction_steps = []
if 'backtest_history' not in st.session_state:
    st.session_state.backtest_history = []
if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []
if 'stock_symbol' not in st.session_state:
    st.session_state.stock_symbol = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'selected_indicators' not in st.session_state:
    st.session_state.selected_indicators = []
if 'use_sentiment' not in st.session_state:
    st.session_state.use_sentiment = False
if 'refined_model' not in st.session_state:
    st.session_state.refined_model = None
if 'options_q_table' not in st.session_state:
    st.session_state.options_q_table = {}
if 'refinement_granularity' not in st.session_state:
    st.session_state.refinement_granularity = "days"
if 'current_accuracies' not in st.session_state:
    st.session_state.current_accuracies = {"minutes": 0, "hours": 0, "days": 0}
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = {"minutes": False, "hours": False, "days": False}


# Function to add step to prediction process
def add_prediction_step(step):
    st.session_state.prediction_steps.append(step)


# Analysis Log (to track steps, lessons, and corrections)
def log_analysis_step(step, details, lessons="", corrections="", suggestions=""):
    st.session_state.analysis_log.append({
        "Step": step,
        "Details": details,
        "Lessons Learned": lessons,
        "Corrections": corrections,
        "Suggestions": suggestions
    })


# Save historical data to database
def save_historical_data(ticker, historical_data):
    minutes_data = historical_data.get("minutes", (pd.DataFrame(), None))[0].to_json() if not \
        historical_data.get("minutes", (pd.DataFrame(), None))[0].empty else None
    hours_data = historical_data.get("hours", (pd.DataFrame(), None))[0].to_json() if not \
        historical_data.get("hours", (pd.DataFrame(), None))[0].empty else None
    days_data = historical_data.get("days", (pd.DataFrame(), None))[0].to_json() if not \
        historical_data.get("days", (pd.DataFrame(), None))[0].empty else None
    c.execute(
        "INSERT OR REPLACE INTO historical_data (ticker, minutes_data, hours_data, days_data) VALUES (?, ?, ?, ?)",
        (ticker, minutes_data, hours_data, days_data))
    conn.commit()


# Load historical data from database (Updated to fix pd.read_json warning)
def load_historical_data(ticker):
    c.execute("SELECT minutes_data, hours_data, days_data FROM historical_data WHERE ticker = ?", (ticker,))
    result = c.fetchone()
    if result:
        historical_data = {}
        if result[0]:
            historical_data["minutes"] = (pd.read_json(StringIO(result[0])), "5m")
        if result[1]:
            historical_data["hours"] = (pd.read_json(StringIO(result[1])), "1h")
        if result[2]:
            historical_data["days"] = (pd.read_json(StringIO(result[2])), "1d")
        return historical_data
    return None


# Save model configuration to database
def save_model_config(ticker, model_type, selected_indicators, use_sentiment):
    c.execute(
        "INSERT OR REPLACE INTO model_configs (ticker, model_type, selected_indicators, use_sentiment) VALUES (?, ?, ?, ?)",
        (ticker, model_type, json.dumps(selected_indicators), int(use_sentiment)))
    conn.commit()


# Load model configuration from database
def load_model_config(ticker):
    c.execute("SELECT model_type, selected_indicators, use_sentiment FROM model_configs WHERE ticker = ?", (ticker,))
    result = c.fetchone()
    if result:
        return {
            "model_type": result[0],
            "selected_indicators": json.loads(result[1]),
            "use_sentiment": bool(result[2])
        }
    return None


# Fetch X posts using tweepy
def fetch_x_posts(ticker):
    add_prediction_step("Sentiment Analysis: Fetching recent X posts...")
    try:
        api_key = st.secrets["x_api"]["API_KEY"]
        api_secret = st.secrets["x_api"]["API_SECRET_KEY"]
        bearer_token = st.secrets["x_api"]["BEARER_TOKEN"]

        client = tweepy.Client(bearer_token=bearer_token)
        query = f"{ticker} -is:retweet lang:en"
        tweets = client.search_recent_tweets(query=query, max_results=10)

        posts = []
        if tweets.data:
            for tweet in tweets.data:
                posts.append({"text": tweet.text})
        else:
            posts = [{"text": "No recent tweets found for this ticker."}]

        add_prediction_step(f"Sentiment Analysis: Fetched {len(posts)} X posts related to {ticker}.")
        log_analysis_step(
            "Sentiment Analysis - Fetch X Posts",
            f"Fetched {len(posts)} X posts related to {ticker}.",
            "",
            "",
            ""
        )

        return posts
    except Exception as e:
        add_prediction_step(f"Sentiment Analysis: Failed to fetch X posts. Error: {str(e)}.")
        log_analysis_step(
            "Sentiment Analysis - Fetch X Posts",
            f"Failed to fetch X posts for {ticker}. Error: {str(e)}.",
            "",
            "",
            "Suggestions: Check API credentials or try again later."
        )
        return [
            {"text": "SPY is looking bullish today! Great time to buy. #stocks"},
            {"text": "Market is crashing, SPY down 2%. Time to sell? #finance"},
            {"text": "SPY holding steady at 592. Neutral outlook for now. #investing"},
            {"text": "Loving the gains on SPY this week! #bullmarket"},
            {"text": "SPY might dip soon, be cautious. #trading"}
        ]


# Compute sentiment scores using VADER
@st.cache_data
def compute_sentiment(posts):
    add_prediction_step("Sentiment Analysis: Computing sentiment scores using VADER...")
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for post in posts:
        sentiment = analyzer.polarity_scores(post['text'])
        compound_score = sentiment['compound']
        sentiments.append({"text": post['text'], "sentiment": compound_score})

    avg_sentiment = np.mean([s['sentiment'] for s in sentiments]) if sentiments else 0

    add_prediction_step(
        f"Sentiment Analysis: Computed sentiment for {len(sentiments)} posts. Average sentiment score: {avg_sentiment:.2f}.")
    log_analysis_step(
        "Sentiment Analysis - Compute Sentiment",
        f"Computed sentiment for {len(sentiments)} posts using VADER. Average sentiment score: {avg_sentiment:.2f}.",
        "",
        "",
        ""
    )

    return sentiments, avg_sentiment


# Fetch VIX data
@st.cache_data
def fetch_vix_data():
    try:
        vix = yf.Ticker("^VIX")
        real_world_end_date = datetime.now()
        start_date = real_world_end_date - timedelta(days=30)
        vix_data = vix.history(start=start_date.strftime("%Y-%m-%d"), end=real_world_end_date.strftime("%Y-%m-%d"),
                               interval="1d")
        return vix_data['Close'].mean() if not vix_data.empty else 0
    except Exception as e:
        add_prediction_step(f"Data Retrieval: Failed to fetch VIX data. Error: {str(e)}.")
        return 0


# Fetch historical data based on granularity
@st.cache_data
def fetch_historical_data(ticker, granularity):
    add_prediction_step(f"Data Retrieval: Fetching historical data for {ticker} ({granularity} granularity)...")
    try:
        stock = yf.Ticker(ticker)
        real_world_end_date = datetime.now()
        if granularity == "minutes":
            start_date = real_world_end_date - timedelta(days=5)
            interval = "5m"
        elif granularity == "hours":
            start_date = real_world_end_date - timedelta(days=30)
            interval = "1h"
        else:  # days
            start_date = real_world_end_date - timedelta(days=365)
            interval = "1d"

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = real_world_end_date.strftime("%Y-%m-%d")
        data = stock.history(start=start_date_str, end=end_date_str, interval=interval)

        if data.empty:
            error_msg = (
                f"No data available for ticker {ticker} with interval {interval} from {start_date_str} to {end_date_str}.")
            add_prediction_step(f"Data Retrieval: Failed. {error_msg}")
            log_analysis_step(
                "Data Retrieval",
                f"Failed to fetch historical data for {ticker} ({granularity} granularity). Interval: {interval}, Start: {start_date_str}, End: {end_date_str}.",
                "",
                "",
                "Suggestions: Try a different ticker or adjust the date range."
            )
            return None, interval

        add_prediction_step(
            f"Data Retrieval: Fetched {len(data)} data points for {ticker} ({granularity} granularity) from {data.index[0]} to {data.index[-1]}.")
        log_analysis_step(
            "Data Retrieval",
            f"Fetched historical data for {ticker} ({granularity} granularity). Interval: {interval}, Start: {start_date_str}, End: {end_date_str}. "
            f"Retrieved {len(data)} data points spanning {data.index[0]} to {data.index[-1]}.",
            "",
            "",
            ""
        )
        return data, interval
    except Exception as e:
        add_prediction_step(f"Data Retrieval: Failed with error: {str(e)}.")
        log_analysis_step(
            "Data Retrieval",
            f"Attempted to fetch historical data for {ticker} ({granularity} granularity). Failed with error: {str(e)}.",
            "",
            "",
            "Suggestions: Try a different ticker or check network connectivity."
        )
        return None, None


# Update historical data with latest information
def update_historical_data(ticker, historical_data):
    add_prediction_step(f"Data Update: Updating historical data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        real_world_end_date = datetime.now()
        for granularity, (data, interval) in historical_data.items():
            last_date = data.index[-1]
            if last_date < real_world_end_date:
                new_data = stock.history(start=last_date.strftime("%Y-%m-%d"),
                                         end=real_world_end_date.strftime("%Y-%m-%d"), interval=interval)
                if not new_data.empty:
                    historical_data[granularity] = (pd.concat([data, new_data]).drop_duplicates(), interval)
        save_historical_data(ticker, historical_data)
        add_prediction_step("Data Update: Successfully updated historical data.")
    except Exception as e:
        add_prediction_step(f"Data Update: Failed to update historical data. Error: {str(e)}.")


# Add indicators to historical data
@st.cache_data
def add_indicators(data, selected_indicators, use_sentiment, ticker):
    add_prediction_step("Data Processing: Adding indicators to historical data...")
    try:
        # Base features
        data['Returns'] = data['Close'].pct_change()
        data['Stock'] = ticker
        data['VIX'] = fetch_vix_data()  # Add VIX as a feature

        # Add user-selected indicators (max 5)
        if selected_indicators:
            add_prediction_step(f"Data Processing: Adding indicators: {', '.join(selected_indicators)}...")
            if 'ATR' in selected_indicators:
                data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
            if 'VWAP' in selected_indicators:
                data['VWAP'] = VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume'],
                                                          window=14).volume_weighted_average_price()
            if 'CCI' in selected_indicators:
                data['CCI'] = CCIIndicator(data['High'], data['Low'], data['Close'], window=20).cci()
            if 'Stochastic Oscillator' in selected_indicators:
                data['Stochastic'] = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14).stoch()
            if 'ADX' in selected_indicators:
                data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
            if 'Ichimoku Cloud' in selected_indicators:
                ichimoku = IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
                data['Ichimoku_A'] = ichimoku.ichimoku_a()
                data['Ichimoku_B'] = ichimoku.ichimoku_b()
            if 'Wyckoff Phase Detection' in selected_indicators:
                rolling_high = data['Close'].rolling(window=20).max()
                rolling_low = data['Close'].rolling(window=20).min()
                data['PriceRange'] = (data['Close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
                data['WyckoffPhase'] = 0
                data.loc[(data['Volume'] < data['Volume'].rolling(window=20).mean()) & (
                        data['PriceRange'] < 0.3), 'WyckoffPhase'] = 1
                data.loc[(data['Volume'] > data['Volume'].rolling(window=20).mean()) & (
                        data['PriceRange'] > 0.7), 'WyckoffPhase'] = -1
            if 'MACD Histogram' in selected_indicators:
                macd = MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
                data['MACD'] = macd.macd()
                data['MACD_Signal'] = macd.macd_signal()
                data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            if 'MFI' in selected_indicators:
                data['MFI'] = MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume'],
                                           window=14).money_flow_index()
            if 'RVI' in selected_indicators:
                data['RVI'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
                data['RVI'] = data['RVI'].rolling(window=14).mean()
            if 'Keltner Channels' in selected_indicators:
                kc = KeltnerChannel(data['High'], data['Low'], data['Close'], window=20)
                data['Keltner_Upper'] = kc.keltner_channel_hband()
                data['Keltner_Lower'] = kc.keltner_channel_lband()
            if 'Donchian Channels' in selected_indicators:
                dc = DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
                data['Donchian_Upper'] = dc.donchian_channel_hband()
                data['Donchian_Lower'] = dc.donchian_channel_lband()
            if 'Parabolic SAR' in selected_indicators:
                data['Parabolic_SAR'] = PSARIndicator(data['High'], data['Low'], data['Close']).psar()
            if 'CMF' in selected_indicators:
                data['CMF'] = ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'],
                                                        window=20).chaikin_money_flow()
            if 'Force Index' in selected_indicators:
                data['Force_Index'] = ForceIndexIndicator(data['Close'], data['Volume'], window=13).force_index()
            if 'OBV' in selected_indicators:
                data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            if 'Volume Delta' in selected_indicators:
                data['Direction'] = np.sign(data['Close'].diff())
                data['Buying_Volume'] = np.where(data['Direction'] > 0, data['Volume'], 0)
                data['Selling_Volume'] = np.where(data['Direction'] < 0, data['Volume'], 0)
                data['Volume_Delta'] = data['Buying_Volume'] - data['Selling_Volume']
            if 'Elder’s Impulse System' in selected_indicators:
                macd = MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
                data['MACD'] = macd.macd()
                data['MACD_Signal'] = macd.macd_signal()
                data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
                data['EMA13'] = EMAIndicator(data['Close'], window=13).ema_indicator()
                data['MACD_Hist_Diff'] = data['MACD_Hist'].diff()
                data['EMA13_Diff'] = data['EMA13'].diff()
                data['Elder_Impulse'] = 0
                data.loc[(data['MACD_Hist_Diff'] > 0) & (data['EMA13_Diff'] > 0), 'Elder_Impulse'] = 1  # Bullish
                data.loc[(data['MACD_Hist_Diff'] < 0) & (data['EMA13_Diff'] < 0), 'Elder_Impulse'] = -1  # Bearish
            if 'RSI' in selected_indicators:
                data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
            if 'Bollinger Bands Width' in selected_indicators:
                bb_indicator = BollingerBands(data['Close'], window=20, window_dev=2)
                data['BB_Upper'] = bb_indicator.bollinger_hband()
                data['BB_Lower'] = bb_indicator.bollinger_lband()
                data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']

        # Add sentiment as a feature if enabled
        if use_sentiment:
            posts = fetch_x_posts(ticker)
            _, avg_sentiment = compute_sentiment(tuple(posts))  # Convert to tuple for caching
            data['Sentiment'] = avg_sentiment

        data = data.dropna()
        return data
    except Exception as e:
        add_prediction_step(f"Data Processing: Failed to add indicators. Error: {str(e)}.")
        log_analysis_step(
            "Data Processing",
            f"Failed to add indicators to data for {ticker}. Error: {str(e)}.",
            "",
            "",
            "Suggestions: Check indicator implementation or reduce the number of indicators."
        )
        return None


# UPDATED: Self-improvement training function to continuously improve until target accuracy
def train_with_self_improvement(data, model_type, granularity, selected_indicators, use_sentiment, ticker,
                                target_accuracy):
    add_prediction_step(f"Self-Improvement: Starting iterative training for {model_type} on {granularity} data...")
    current_accuracy = st.session_state.current_accuracies[granularity]
    iteration = 0
    n_estimators = 50 if model_type in ["Random Forest", "XGBoost"] else None
    lstm_epochs = 5 if model_type == "LSTM" else None
    model = None

    # Log initial state
    add_prediction_step(f"Initial Accuracy for {granularity}: {current_accuracy:.2f}% (Target: {target_accuracy}%)")

    # Limit to a small number of iterations for testing
    max_iterations = 3  # Temporarily limit to 3 iterations for debugging
    while current_accuracy < target_accuracy and st.session_state.is_training and not \
            st.session_state.training_completed[granularity] and iteration < max_iterations:
        iteration += 1
        add_prediction_step(f"Training Iteration {iteration} for {granularity}...")

        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        features = ['Returns', 'VIX']
        if use_sentiment:
            features.append('Sentiment')
        additional_features = ['ATR', 'VWAP', 'CCI', 'Stochastic', 'ADX', 'Ichimoku_A', 'Ichimoku_B', 'WyckoffPhase',
                               'MACD_Hist', 'BB_Width', 'OBV', 'MFI', 'RVI', 'Keltner_Upper', 'Keltner_Lower',
                               'Donchian_Upper', 'Donchian_Lower', 'Parabolic_SAR', 'CMF', 'Force_Index',
                               'Volume_Delta', 'Elder_Impulse',
                               'RSI']
        features.extend([f for f in additional_features if f in data.columns])

        X_train = train_data[features].shift(1).dropna()
        y_train = train_data['Close'][1:len(X_train) + 1]
        X_test = test_data[features].shift(1).dropna()
        y_test = test_data['Close'][1:len(X_test) + 1]

        if len(X_train) == 0 or len(X_test) == 0:
            add_prediction_step(f"Error: Insufficient data for training on {granularity} granularity. Skipping...")
            st.session_state.training_completed[granularity] = True
            return None, 0

        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == "LSTM":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_train.values)
            y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
            X_lstm = []
            y_lstm = []
            timesteps = 10
            for i in range(timesteps, len(X_scaled)):
                X_lstm.append(X_scaled[i - timesteps:i])
                y_lstm.append(y_scaled[i])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            if len(X_lstm) == 0:
                add_prediction_step(f"Error: Insufficient timesteps for LSTM on {granularity} granularity. Skipping...")
                return None, 0
            model = Sequential()
            model.add(Input(shape=(X_lstm.shape[1], X_lstm.shape[2])))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, y_lstm, epochs=lstm_epochs, batch_size=32, verbose=0)

            X_test_scaled = scaler.transform(X_test.values)
            X_test_lstm = []
            for i in range(timesteps, len(X_test_scaled)):
                X_test_lstm.append(X_test_scaled[i - timesteps:i])
            X_test_lstm = np.array(X_test_lstm)
            y_pred_scaled = model.predict(X_test_lstm, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        else:
            add_prediction_step(f"Error: Unsupported model type: {model_type}")
            return None, 0

        if model_type != "LSTM":
            add_prediction_step(f"Fitting {model_type} model with {len(X_train)} training samples...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception as e:
                add_prediction_step(f"Error during model training: {str(e)}")
                return None, 0

        actual_directions = np.sign(y_test.values[1:] - y_test.values[:-1])
        predicted_directions = np.sign(y_pred[1:] - y_pred[:-1])
        matches = (actual_directions == predicted_directions).astype(int)
        current_accuracy = np.mean(matches) * 100 if len(matches) > 0 else 0

        # Update session state with the new accuracy
        st.session_state.current_accuracies[granularity] = current_accuracy
        add_prediction_step(f"Updated Accuracy for {granularity}: {current_accuracy:.2f}%")

        # Store iteration result
        st.session_state.backtest_history.append({
            "Granularity": granularity,
            "Model": model_type,
            "Iteration": iteration,
            "Accuracy (%)": current_accuracy
        })

        # Adjust hyperparameters
        if model_type in ["Random Forest", "XGBoost"]:
            n_estimators += 10  # Increase trees
        elif model_type == "LSTM":
            lstm_epochs += 1  # Increase epochs

        add_prediction_step(
            f"Self-Improvement: Iteration {iteration} on {granularity} - Accuracy: {current_accuracy:.2f}% (Target: {target_accuracy}%)")

        # Check if target accuracy is reached
        if current_accuracy >= target_accuracy:
            st.session_state.training_completed[granularity] = True
            add_prediction_step(
                f"Self-Improvement: Target accuracy {target_accuracy}% reached for {granularity} granularity!")

        # Force a UI update to reflect the new accuracy
        st.rerun()

    if iteration >= max_iterations:
        add_prediction_step(f"Reached maximum iterations ({max_iterations}) for {granularity}. Stopping training.")
        st.session_state.training_completed[granularity] = True

    return model, current_accuracy


# Backtesting function
def backtest_model(data, model_type, granularity, selected_indicators, use_sentiment, ticker):
    add_prediction_step(f"Backtesting: Starting backtesting for {granularity} granularity...")
    try:
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        features = ['Returns', 'VIX']
        if use_sentiment:
            features.append('Sentiment')
        additional_features = ['ATR', 'VWAP', 'CCI', 'Stochastic', 'ADX', 'Ichimoku_A', 'Ichimoku_B', 'WyckoffPhase',
                               'MACD_Hist',
                               'BB_Width', 'OBV', 'MFI', 'RVI', 'Keltner_Upper', 'Keltner_Lower', 'Donchian_Upper',
                               'Donchian_Lower',
                               'Parabolic_SAR', 'CMF', 'Force_Index', 'Volume_Delta', 'Elder_Impulse', 'RSI']
        features.extend([f for f in additional_features if f in data.columns])

        X_train = train_data[features].shift(1).dropna()
        y_train = train_data['Close'][1:len(X_train) + 1]
        X_test = test_data[features].shift(1).dropna()
        y_test = test_data['Close'][1:len(X_test) + 1]

        # Train the selected model
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=50, random_state=42)
        elif model_type == "LSTM":
            scaler = MinMaxScaler()
            X_train_values = X_train.values
            y_train_values = y_train.values
            X_scaled = scaler.fit_transform(X_train_values)
            y_scaled = scaler.fit_transform(y_train_values.reshape(-1, 1))
            X_lstm = []
            y_lstm = []
            timesteps = 10
            for i in range(timesteps, len(X_scaled)):
                X_lstm.append(X_scaled[i - timesteps:i])
                y_lstm.append(y_scaled[i])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

            if len(X_lstm) == 0:
                add_prediction_step(
                    f"Backtesting: Failed. Insufficient timesteps for LSTM model ({granularity} granularity).")
                return None, None, None, None, None, None, None, None

            model = Sequential()
            model.add(Input(shape=(X_lstm.shape[1], X_lstm.shape[2])))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0)

            X_test_values = X_test.values
            X_test_scaled = scaler.transform(X_test_values)
            X_test_lstm = []
            for i in range(timesteps, len(X_test_scaled)):
                X_test_lstm.append(X_test_scaled[i - timesteps:i])
            X_test_lstm = np.array(X_test_lstm)
            y_pred_scaled = model.predict(X_test_lstm, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type != "LSTM":
            add_prediction_step(
                f"Backtesting: Training {model_type} model for {granularity} granularity with {len(X_train)} samples...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Calculate directional accuracy
        y_test_values = y_test.values
        y_pred_values = y_pred
        actual_directions = np.sign(y_test_values[1:] - y_test_values[:-1])
        predicted_directions = np.sign(y_pred_values[1:] - y_pred_values[:-1])
        matches = (actual_directions == predicted_directions).astype(int)
        accuracy = np.mean(matches) * 100 if len(matches) > 0 else 0

        # Calculate total return
        predicted_returns = (y_pred[1:] - y_pred[:-1]) / y_pred[:-1]
        total_return = np.prod(1 + predicted_returns) - 1 if len(predicted_returns) > 0 else 0

        # Calculate Sharpe Ratio (simplified)
        returns = (y_pred[1:] - y_pred[:-1]) / y_pred[:-1]
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0

        backtest_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }, index=y_test.index)

        # Store backtest result in history
        backtest_entry = {
            "Granularity": granularity,
            "Model": model_type,
            "Indicators": ", ".join(selected_indicators) if selected_indicators else "None",
            "Sentiment": "Yes" if use_sentiment else "No",
            "MSE": mse,
            "MAE": mae,
            "Accuracy (%)": accuracy,
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Train Period": f"{train_data.index[0]} to {train_data.index[-1]}",
            "Test Period": f"{test_data.index[0]} to {test_data.index[-1]}"
        }
        st.session_state.backtest_history.append(backtest_entry)

        add_prediction_step(
            f"Backtesting: Completed for {granularity} granularity. MSE: {mse:.4f}, MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%, Total Return: {total_return * 100:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}.")
        log_analysis_step(
            "Backtesting",
            f"Backtested {model_type} model for {ticker} ({granularity} granularity). Train period: {train_data.index[0]} to {train_data.index[-1]}. "
            f"Test period: {test_data.index[0]} to {test_data.index[-1]}. "
            f"MSE: {mse:.4f}, MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%, Total Return: {total_return * 100:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}.",
            "",
            "",
            ""
        )

        # Compute feature importance for Random Forest
        if model_type == "Random Forest":
            feature_importance = pd.Series(model.feature_importances_, index=features)
        else:
            feature_importance = None

        return backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, feature_importance, model
    except Exception as e:
        add_prediction_step(f"Backtesting: Failed for {granularity} granularity with error: {str(e)}.")
        log_analysis_step(
            "Backtesting",
            f"Attempted to backtest model for {ticker} ({granularity} granularity). Failed with error: {str(e)}.",
            "",
            "",
            "Suggestions: Check data availability or model configuration."
        )
        return None, None, None, None, None, None, None, None


# Suggest indicator to improve accuracy
def suggest_indicator(feature_importance, current_indicators):
    if feature_importance is None:
        return "Run the prediction with Random Forest to get feature importance for indicator suggestions."

    available_indicators = [
        "ATR", "VWAP", "CCI", "Stochastic Oscillator", "ADX", "Ichimoku Cloud",
        "Wyckoff Phase Detection", "MACD Histogram", "Bollinger Bands Width",
        "OBV", "MFI", "RVI", "Keltner Channels", "Donchian Channels",
        "Parabolic SAR", "CMF", "Force Index", "RSI", "Volume Delta", "Elder’s Impulse System"
    ]
    current_features = ['Returns', 'VIX']
    if 'Sentiment' in feature_importance:
        current_features.append('Sentiment')
    current_features.extend(current_indicators)
    available_to_add = [ind for ind in available_indicators if ind not in current_features]

    if not available_to_add:
        return "All available indicators are already in use."

    suggestion = available_to_add[0]
    explanation = f"Adding {suggestion} may improve accuracy as it introduces new patterns (e.g., {suggestion} captures short-term trends)."
    return f"Consider adding {suggestion}. {explanation}"


# Predict future prices based on time horizon
def predict_future(data, model, model_type, horizon_value, horizon_unit, selected_indicators, use_sentiment, ticker,
                   interval):
    add_prediction_step(f"Prediction: Predicting future prices for {horizon_value} {horizon_unit}...")
    try:
        # Convert horizon to minutes and select appropriate granularity
        if horizon_unit == "minutes":
            horizon_minutes = horizon_value
            granularity = "minutes"  # Use minute data for short horizons
        elif horizon_unit == "hours":
            horizon_minutes = horizon_value * 60
            granularity = "hours" if horizon_value <= 24 else "days"
        else:
            horizon_minutes = horizon_value * 24 * 60
            granularity = "days"

        # Fetch appropriate data if granularity differs
        if granularity != interval:
            data, interval = st.session_state.historical_data[granularity]
            data = add_indicators(data, tuple(st.session_state.selected_indicators), st.session_state.use_sentiment,
                                  ticker)

        # Define features
        features = ['Returns', 'VIX']
        if use_sentiment:
            features.append('Sentiment')
        additional_features = ['ATR', 'VWAP', 'CCI', 'Stochastic', 'ADX', 'Ichimoku_A', 'Ichimoku_B', 'WyckoffPhase',
                               'MACD_Hist',
                               'BB_Width', 'OBV', 'MFI', 'RVI', 'Keltner_Upper', 'Keltner_Lower', 'Donchian_Upper',
                               'Donchian_Lower',
                               'Parabolic_SAR', 'CMF', 'Force_Index', 'Volume_Delta', 'Elder_Impulse', 'RSI']
        features.extend([f for f in additional_features if f in data.columns])

        X = data[features].shift(1).dropna()
        y = data['Close'][1:len(X) + 1]

        # Calculate n_steps based on horizon_minutes and interval
        if interval == "5m":
            n_steps = horizon_minutes // 5
        elif interval == "1h":
            n_steps = horizon_minutes // 60
        elif interval == "1d":
            n_steps = horizon_minutes // (24 * 60)
        else:
            n_steps = horizon_minutes // 60

        if horizon_minutes > 0 and n_steps <= 0:
            n_steps = 1  # Ensure at least one prediction step

        if n_steps <= 0:
            add_prediction_step(f"Prediction: Failed. Calculated n_steps ({n_steps}) is not positive.")
            return None, None

        # Predict future prices
        if model_type == "LSTM":
            scaler = MinMaxScaler()
            X_values = X.values
            y_values = y.values
            X_scaled = scaler.fit_transform(X_values)
            y_scaled = scaler.fit_transform(y_values.reshape(-1, 1))

            X_lstm = []
            y_lstm = []
            timesteps = 10
            for i in range(timesteps, len(X_scaled)):
                X_lstm.append(X_scaled[i - timesteps:i])
                y_lstm.append(y_scaled[i])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

            predictions = []
            last_X = X_scaled[-timesteps:]
            last_close = data['Close'].iloc[-1]
            progress_bar = st.progress(0)
            for i in range(n_steps):
                last_X_reshaped = last_X.reshape((1, timesteps, len(features)))
                pred_scaled = model.predict(last_X_reshaped, verbose=0)
                pred = scaler.inverse_transform(pred_scaled)[0, 0]
                predictions.append(pred)
                last_X = np.vstack((last_X[1:], last_X[-1:]))
                last_X[-1, 0] = (pred - last_close) / last_close  # Update Returns
                last_close = pred
                progress = (i + 1) / n_steps
                progress_bar.progress(progress)
            progress_bar.empty()
        else:
            predictions = []
            last_X = X.iloc[-1:].copy()
            last_close = data['Close'].iloc[-1]
            progress_bar = st.progress(0)
            for i in range(n_steps):
                pred = model.predict(last_X)[0]
                predictions.append(pred)
                last_X['Returns'] = (pred - last_close) / last_close
                last_close = pred
                for f in additional_features:
                    if f in last_X:
                        last_X[f] = last_X[f].values[0]
                progress = (i + 1) / n_steps
                progress_bar.progress(progress)
            progress_bar.empty()

        last_date = data.index[-1]
        if interval == "5m":
            future_dates = [last_date + timedelta(minutes=5 * i) for i in range(1, n_steps + 1)]
        elif interval == "1h":
            future_dates = [last_date + timedelta(hours=i) for i in range(1, n_steps + 1)]
        elif interval == "1d":
            future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps + 1)]
        else:
            future_dates = [last_date + timedelta(hours=i) for i in range(1, n_steps + 1)]

        pred_df = pd.DataFrame({
            'Stock': ticker,
            'Close': predictions,
            '': [''] * len(predictions)
        }, index=future_dates)

        # Store prediction in database
        predicted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        horizon_datetime = future_dates[-1].strftime("%Y-%m-%d %H:%M:%S")
        predicted_price = predictions[-1]
        c.execute(
            "INSERT INTO predictions (ticker, predicted_at, horizon_datetime, predicted_price) VALUES (?, ?, ?, ?)",
            (ticker, predicted_at, horizon_datetime, predicted_price))
        conn.commit()
        prediction_id = c.lastrowid

        add_prediction_step(f"Prediction: Generated {len(pred_df)} predictions for {ticker}.")
        log_analysis_step(
            "Prediction",
            f"Generated {len(pred_df)} predictions for {ticker} over {horizon_value} {horizon_unit}.",
            "",
            "",
            ""
        )

        return pred_df, prediction_id
    except Exception as e:
        add_prediction_step(f"Prediction: Failed with error: {str(e)}.")
        log_analysis_step(
            "Prediction",
            f"Failed to predict for {ticker} over {horizon_value} {horizon_unit}. Error: {str(e)}.",
            "",
            "",
            "Suggestions: Check model configuration or extend the horizon."
        )
        return None, None


# Recommend options contracts
def recommend_options(ticker, predictions, horizon_minutes, prediction_id):
    add_prediction_step("Options Recommendation: Evaluating options contracts...")
    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain(stock.options[0])

        valley = predictions['Close'].min()
        peak = predictions['Close'].max()
        valley_time = predictions['Close'].idxmin()
        peak_time = predictions['Close'].idxmax()

        # Use Q-learning to adjust selection criteria
        state = f"volatility_{options.calls['impliedVolatility'].mean():.2f}_range_{(peak - valley) / valley:.2f}"
        if state not in st.session_state.options_q_table:
            st.session_state.options_q_table[state] = {'strike_range': 5, 'iv_threshold': 0.5}

        # Relax criteria to find more options
        strike_range = st.session_state.options_q_table[state]['strike_range'] * 1.5  # Increase range
        iv_threshold = st.session_state.options_q_table[state]['iv_threshold'] * 1.2  # Relax IV threshold

        otm_calls = options.calls[
            (options.calls['strike'] > valley) &
            (options.calls['strike'] <= peak + strike_range) &
            (options.calls['impliedVolatility'] < iv_threshold)
            ]
        call_profit = 0
        best_call = None
        if not otm_calls.empty:
            best_call = otm_calls.sort_values('volume', ascending=False).iloc[0]
            call_buy = best_call['lastPrice']
            call_sell = max(call_buy * (1 + best_call['impliedVolatility'] * (peak - valley) / valley), call_buy + 0.10)
            call_profit = call_sell - call_buy

        otm_puts = options.puts[
            (options.puts['strike'] < peak) &
            (options.puts['strike'] >= valley - strike_range) &
            (options.puts['impliedVolatility'] < iv_threshold)
            ]
        put_profit = 0
        best_put = None
        if not otm_puts.empty:
            best_put = otm_puts.sort_values('volume', ascending=False).iloc[0]
            put_buy = best_put['lastPrice']
            put_sell = max(put_buy * (1 + best_put['impliedVolatility'] * (peak - valley) / valley), put_buy + 0.10)
            put_profit = put_sell - put_buy

        exit_time = peak_time if call_profit > put_profit else valley_time
        if horizon_minutes >= 1440:
            exit_time += timedelta(days=1)

        if call_profit > put_profit and best_call is not None:
            result = {
                "type": "Call",
                "strike": best_call['strike'],
                "buy_price": call_buy,
                "sell_price": call_sell,
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_profit": call_profit
            }
        elif put_profit > 0 and best_put is not None:
            result = {
                "type": "Put",
                "strike": best_put['strike'],
                "buy_price": put_buy,
                "sell_price": put_sell,
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_profit": put_profit
            }
        else:
            result = None
            add_prediction_step("Options Recommendation: No suitable option found based on current criteria.")

        # Visualize the predicted price movement and recommended option
        if result:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Close'], mode='lines', name='Predicted Price',
                                     line=dict(color='blue')))
            fig.add_hline(y=result['strike'], line_dash="dash", line_color="red",
                          annotation_text=f"{result['type']} Strike: ${result['strike']:.2f}",
                          annotation_position="top left")
            fig.update_layout(title=f"Options Recommendation for {ticker}: Predicted Price Movement",
                              xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
            st.plotly_chart(fig)

            c.execute(
                "INSERT INTO options_recommendations (prediction_id, type, strike, buy_price, sell_price, exit_time, predicted_profit) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (prediction_id, result['type'], result['strike'], result['buy_price'], result['sell_price'],
                 result['exit_time'], result['predicted_profit']))
            conn.commit()
            add_prediction_step(
                f"Options Recommendation: Selected {result['type']} option with strike ${result['strike']:.2f}, "
                f"buy at ${result['buy_price']:.2f}, sell at ${result['sell_price']:.2f}, "
                f"exit by {result['exit_time']}, predicted profit ${result['predicted_profit']:.2f}.")
        else:
            add_prediction_step("Options Recommendation: No options recommended. Consider adjusting criteria.")

        log_analysis_step(
            "Options Recommendation",
            f"Evaluated options for {ticker}. Selected {result['type'] if result else 'None'} option. "
            f"Valley: ${valley:.2f} at {valley_time}, Peak: ${peak:.2f} at {peak_time}.",
            "",
            "",
            "Consider relaxing strike range or IV threshold if no options are found."
        )

        return result
    except Exception as e:
        add_prediction_step(f"Options Recommendation: Failed with error: {str(e)}.")
        log_analysis_step(
            "Options Recommendation",
            f"Failed to evaluate options for {ticker}. Error: {str(e)}.",
            "",
            "",
            "Suggestions: Check options data availability or try a different ticker."
        )
        return None


# Evaluate predictions and options
def evaluate_predictions(ticker, historical_data):
    add_prediction_step("Prediction Evaluation: Checking for completed predictions...")
    c.execute(
        "SELECT id, predicted_at, horizon_datetime, predicted_price, actual_price FROM predictions WHERE ticker = ? AND actual_price IS NULL",
        (ticker,))
    predictions_to_evaluate = c.fetchall()

    for pred_id, predicted_at, horizon_datetime, predicted_price, actual_price in predictions_to_evaluate:
        horizon_dt = datetime.strptime(horizon_datetime, "%Y-%m-%d %H:%M:%S")
        if datetime.now() >= horizon_dt:
            try:
                stock = yf.Ticker(ticker)
                actual_data = stock.history(start=horizon_dt.strftime("%Y-%m-%d"),
                                            end=(horizon_dt + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1h")
                actual_price = actual_data['Close'].iloc[-1] if not actual_data.empty else predicted_price

                # Calculate metrics
                mse = mean_squared_error([actual_price], [predicted_price])
                mae = mean_absolute_error([actual_price], [predicted_price])
                accuracy = 100 if (actual_price > predicted_price and actual_price > actual_data['Close'].iloc[-2]) or (
                        actual_price < predicted_price and actual_price < actual_data['Close'].iloc[-2]) else 0
                total_return = (actual_price - actual_data['Close'].iloc[-2]) / actual_data['Close'].iloc[-2] if len(
                    actual_data) > 1 else 0
                returns = [total_return]
                mean_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0
                sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0

                # Plot Prediction vs Actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[horizon_dt], y=[predicted_price], mode='markers', name='Predicted',
                                         marker=dict(color='red', size=10)))
                fig.add_trace(go.Scatter(x=[horizon_dt], y=[actual_price], mode='markers', name='Actual',
                                         marker=dict(color='blue', size=10)))
                fig.update_layout(title="Prediction vs Actual", xaxis_title="Date", yaxis_title="Price",
                                  template="plotly_dark")
                st.plotly_chart(fig)

                # ML Adjustment Summary
                error = actual_price - predicted_price
                adjustment = "Increase model complexity (e.g., more trees for Random Forest)" if error > 0 else "Reduce overfitting (e.g., increase regularization)"
                st.write(f"**ML Adjustment Summary**: Error: {error:.2f}. Suggested adjustment: {adjustment}.")

                metrics = {
                    "mse": mse,
                    "mae": mae,
                    "accuracy": accuracy,
                    "total_return": total_return * 100,
                    "sharpe_ratio": sharpe_ratio
                }

                # Update prediction with actual price and metrics
                c.execute("UPDATE predictions SET actual_price = ?, metrics = ? WHERE id = ?",
                          (actual_price, json.dumps(metrics), pred_id))

                # Evaluate options
                c.execute(
                    "SELECT type, strike, buy_price, sell_price, predicted_profit FROM options_recommendations WHERE prediction_id = ?",
                    (pred_id,))
                option = c.fetchone()
                if option:
                    option_type, strike, buy_price, sell_price, predicted_profit = option
                    if option_type == "Call":
                        actual_profit = max(0,
                                            actual_price - strike) - buy_price if actual_price > strike else -buy_price
                    else:  # Put
                        actual_profit = max(0,
                                            strike - actual_price) - buy_price if actual_price < strike else -buy_price

                    c.execute("UPDATE options_recommendations SET actual_profit = ? WHERE prediction_id = ?",
                              (actual_profit, pred_id))

                    # Update Q-table for options (reinforcement learning)
                    state = f"volatility_{options.calls['impliedVolatility'].mean():.2f}_range_{(peak - valley) / valley:.2f}"
                    reward = actual_profit - predicted_profit
                    alpha, gamma = 0.1, 0.9  # Learning rate and discount factor
                    if state not in st.session_state.options_q_table:
                        st.session_state.options_q_table[state] = {'strike_range': 5, 'iv_threshold': 0.5}
                    q_table = st.session_state.options_q_table[state]
                    q_table['strike_range'] += alpha * (reward - q_table['strike_range'])
                    q_table['iv_threshold'] += alpha * (reward - q_table['iv_threshold'])

                conn.commit()

                # Adjust model with new data
                update_historical_data(ticker, historical_data)
                for granularity, (data, interval) in historical_data.items():
                    data_with_indicators = add_indicators(data, st.session_state.selected_indicators,
                                                          st.session_state.use_sentiment, ticker)
                    if data_with_indicators is not None:
                        backtest_model(data_with_indicators, st.session_state.model_type, granularity,
                                       st.session_state.selected_indicators, st.session_state.use_sentiment, ticker)

                add_prediction_step(
                    f"Prediction Evaluation: Evaluated prediction ID {pred_id}. Actual Price: ${actual_price:.2f}, Predicted: ${predicted_price:.2f}.")
            except Exception as e:
                add_prediction_step(
                    f"Prediction Evaluation: Failed to evaluate prediction ID {pred_id}. Error: {str(e)}.")


# Streamlit UI (Updated to ensure continuous self-improvement and proper backtesting flow)
def main():
    st.title("Stock Predictor of the Third Age")

    tab1, tab2, tab3, tab4 = st.tabs(["Main App", "Analysis Log", "Backtesting Results", "Prediction Outcomes"])

    with tab1:
        # Prediction Process Sidebar
        with st.sidebar:
            st.markdown('<h3 style="color: #DAA520;">Prediction Process</h3>', unsafe_allow_html=True)
            st.markdown(
                """
                <div style="height: 300px; overflow-y: auto; border: 2px solid #aaa; padding: 15px; background-color: #e6d7a3; border-radius: 5px;">
                """
            )
            if st.session_state.prediction_steps:
                for step in st.session_state.prediction_steps:
                    st.markdown(f"**{step}**")
            else:
                st.write("No prediction steps yet. Start the analysis to see the process.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Step 1: Initial Stock and Model Selection
        if not st.session_state.stock_symbol or not st.session_state.model_type:
            st.subheader("Step 1: Select Stock and Model")
            ticker = st.text_input("Enter Stock Ticker, O Hobbit of the Shire", "SPY")
            model_options = ["Random Forest", "XGBoost", "LSTM"]
            model_type = st.selectbox("Select Prediction Model", model_options, index=0)

            if st.button("Start Analysis"):
                st.session_state.stock_symbol = ticker
                st.session_state.model_type = model_type
                st.session_state.prediction_steps = []  # Reset prediction steps
                st.session_state.current_accuracies = {"minutes": 0, "hours": 0, "days": 0}  # Initialize accuracies
                st.session_state.training_completed = {"minutes": False, "hours": False, "days": False}
                st.session_state.backtest_results = {}  # Reset backtest results

                # Load existing data if available
                historical_data = load_historical_data(ticker)
                if historical_data:
                    st.session_state.historical_data = historical_data
                    config = load_model_config(ticker)
                    if config:
                        st.session_state.model_type = config["model_type"]
                        st.session_state.selected_indicators = config["selected_indicators"]
                        st.session_state.use_sentiment = config["use_sentiment"]
                    add_prediction_step(f"Loaded existing data and configuration for {ticker}.")
                else:
                    st.session_state.historical_data = {}

                add_prediction_step(f"User Input: Set ticker to {ticker}.")
                add_prediction_step(f"User Input: Selected {model_type} as the prediction model.")
                st.rerun()
        else:
            ticker = st.session_state.stock_symbol
            model_type = st.session_state.model_type

            # Reset Button to Choose a New Stock
            if st.button("Reset Analysis and Choose New Stock"):
                st.session_state.stock_symbol = None
                st.session_state.model_type = None
                st.session_state.historical_data = {}
                st.session_state.backtest_results = {}
                st.session_state.feature_importance = None
                st.session_state.selected_indicators = []
                st.session_state.use_sentiment = False
                st.session_state.refined_model = None
                st.session_state.options_q_table = {}
                st.session_state.refinement_granularity = "days"
                st.session_state.prediction_steps = []
                st.session_state.backtest_history = []
                st.session_state.analysis_log = []
                st.session_state.current_accuracies = {"minutes": 0, "hours": 0, "days": 0}
                st.session_state.is_training = False
                st.session_state.training_completed = {"minutes": False, "hours": False, "days": False}
                st.rerun()

            # Step 2: Initial Backtesting (Fetch Data)
            if not st.session_state.historical_data:
                granularities = ["minutes", "hours", "days"]
                for granularity in granularities:
                    data, interval = fetch_historical_data(ticker, granularity)
                    if data is not None:
                        st.session_state.historical_data[granularity] = (data, interval)
                    else:
                        st.error(f"Failed to fetch data for {granularity} granularity. Please try a different ticker.")
                        return
                save_historical_data(ticker, st.session_state.historical_data)

            # Model Selection (Always Accessible)
            st.subheader("Step 1: Select Model and Settings")
            model_options = ["Random Forest", "XGBoost", "LSTM"]
            model_type = st.selectbox("Select Prediction Model", model_options,
                                      index=model_options.index(model_type) if model_type in model_options else 0)
            st.session_state.model_type = model_type

            # Indicator selection (max 5)
            available_indicators = [
                "ATR", "VWAP", "CCI", "Stochastic Oscillator", "ADX", "Ichimoku Cloud",
                "Wyckoff Phase Detection", "MACD Histogram", "Bollinger Bands Width",
                "OBV", "MFI", "RVI", "Keltner Channels", "Donchian Channels",
                "Parabolic SAR", "CMF", "Force Index", "RSI", "Volume Delta", "Elder’s Impulse System"
            ]
            selected_indicators = st.session_state.selected_indicators
            with st.expander("Select Indicators (Max 5)"):
                new_selected_indicators = []
                for indicator in available_indicators:
                    if indicator in selected_indicators:
                        if st.checkbox(indicator, value=True, key=f"ind_{indicator}"):
                            new_selected_indicators.append(indicator)
                    elif len(new_selected_indicators) < 5:
                        if st.checkbox(indicator, key=f"ind_{indicator}"):
                            new_selected_indicators.append(indicator)
                st.session_state.selected_indicators = new_selected_indicators

            # Sentiment analysis toggle
            use_sentiment = st.checkbox("Use Sentiment Analysis", value=st.session_state.use_sentiment)
            st.session_state.use_sentiment = use_sentiment

            # Self-Improvement Phase
            st.subheader("Step 2: Model Self-Improvement Phase")
            target_accuracy = st.slider("Target Accuracy (%)", 50, 95, 75)

            # Display training controls
            if not st.session_state.is_training:
                if st.button("Start Model Training"):
                    st.session_state.is_training = True
                    st.session_state.training_completed = {"minutes": False, "hours": False, "days": False}
                    st.session_state.current_accuracies = {"minutes": 0, "hours": 0, "days": 0}
                    st.session_state.backtest_results = {}  # Reset backtest results to ensure fresh start
                    st.session_state.prediction_steps = []  # Reset prediction steps
                    add_prediction_step("Training started.")
                    st.rerun()
            else:
                st.warning("Training in progress...")
                if st.button("Stop Training"):
                    st.session_state.is_training = False
                    for granularity in ["minutes", "hours", "days"]:
                        st.session_state.training_completed[granularity] = True
                    add_prediction_step("Training stopped by user.")
                    st.rerun()

            # Training Loop for Each Granularity
            if st.session_state.is_training:
                for granularity in ["minutes", "hours", "days"]:
                    if not st.session_state.training_completed[granularity]:
                        data, interval = st.session_state.historical_data[granularity]
                        data_with_indicators = add_indicators(data, tuple(st.session_state.selected_indicators),
                                                              st.session_state.use_sentiment, ticker)
                        if data_with_indicators is not None:
                            model, accuracy = train_with_self_improvement(
                                data_with_indicators, model_type, granularity, st.session_state.selected_indicators,
                                st.session_state.use_sentiment, ticker, target_accuracy
                            )
                            if model is not None:
                                st.session_state.backtest_results[granularity] = (
                                    None, None, None, accuracy, None, None, model)

            # Display Separate Accuracy Meters for Each Granularity
            st.subheader("Training Progress")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Minutes Accuracy", f"{st.session_state.current_accuracies['minutes']:.2f}%", delta=None)
            with col2:
                st.metric("Hours Accuracy", f"{st.session_state.current_accuracies['hours']:.2f}%", delta=None)
            with col3:
                st.metric("Days Accuracy", f"{st.session_state.current_accuracies['days']:.2f}%", delta=None)
            # Proceed to Backtesting Only After Training is Complete or Stopped
            all_training_completed = all(st.session_state.training_completed.values())
            if not st.session_state.is_training and all_training_completed:
                # Perform backtesting for all granularities
                if not st.session_state.backtest_results or any(
                        result[0] is None for result in st.session_state.backtest_results.values()):
                    st.session_state.backtest_results = {}  # Reset to ensure fresh backtesting
                    for granularity, (data, interval) in st.session_state.historical_data.items():
                        data_with_indicators = add_indicators(data, tuple(st.session_state.selected_indicators),
                                                              st.session_state.use_sentiment, ticker)
                        if data_with_indicators is not None:
                            backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, feature_importance, model = backtest_model(
                                data_with_indicators, model_type, granularity, st.session_state.selected_indicators,
                                st.session_state.use_sentiment, ticker
                            )
                            if backtest_results is not None:
                                st.session_state.backtest_results[granularity] = (
                                    backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, model)
                                if feature_importance is not None:
                                    st.session_state.feature_importance = feature_importance
                    save_model_config(ticker, model_type, st.session_state.selected_indicators,
                                      st.session_state.use_sentiment)

                # Display backtesting results
                st.subheader("Current Backtesting Results")
                for granularity, (backtest_results, mse, mae, accuracy, total_return, sharpe_ratio,
                                  model) in st.session_state.backtest_results.items():
                    if backtest_results is not None:  # Only show if backtest results exist
                        st.write(f"**{granularity.capitalize()} Granularity**")
                        st.write(
                            f"MSE: {mse:.4f}, MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%, Total Return: {total_return * 100:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}")
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(x=backtest_results.index, y=backtest_results['Actual'], mode='lines',
                                       name='Actual',
                                       line=dict(color='blue')))
                        fig.add_trace(
                            go.Scatter(x=backtest_results.index, y=backtest_results['Predicted'], mode='lines',
                                       name='Predicted', line=dict(color='red', dash='dash')))
                        fig.update_layout(title=f"Backtesting Results ({granularity.capitalize()})", xaxis_title="Date",
                                          yaxis_title="Price", template="plotly_dark")
                        st.plotly_chart(fig)
                        st.dataframe(backtest_results)

                # Display Live Charts and Indicators
                st.subheader("Live Stock and Indicators Visualization")
                data, interval = st.session_state.historical_data["minutes"]  # Use minute data for live chart
                data_with_indicators = add_indicators(data, tuple(st.session_state.selected_indicators),
                                                      st.session_state.use_sentiment, ticker)

    # Live Candlestick Chart (Updated to fix mplfinance display)
    st.write("Live Candlestick Chart (Last 5 Days, 5-Minute Intervals)")
    fig = mpf.figure(style='charles', figsize=(10, 6))
    ax = fig.add_subplot()
    mpf.plot(data, type='candle', ax=ax, volume=True, title=f"{ticker} Candlestick Chart", ylabel='Price')
    st.pyplot(fig)

    # Indicators and Sentiment
    if data_with_indicators is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Close'], mode='lines',
                       name='Close Price', line=dict(color='blue')))

        for indicator in st.session_state.selected_indicators:
            if indicator in data_with_indicators.columns:
                fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators[indicator],
                                         mode='lines', name=indicator))
            elif indicator == "MACD Histogram":
                fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD_Hist'],
                                         mode='lines', name='MACD Histogram'))
            elif indicator == "Ichimoku Cloud":
                fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Ichimoku_A'],
                                         mode='lines', name='Ichimoku A'))
                fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Ichimoku_B'],
                                         mode='lines', name='Ichimoku B'))

        if st.session_state.use_sentiment:
            fig.add_trace(
                go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Sentiment'], mode='lines',
                           name='Sentiment', yaxis="y2"))

        fig.update_layout(
            title=f"{ticker} Indicators and Sentiment",
            xaxis_title="Date",
            yaxis_title="Price/Indicator Value",
            yaxis2=dict(title="Sentiment", overlaying="y", side="right"),
            template="plotly_dark"
        )
        st.plotly_chart(fig)
    # Step 3: Refinement Phase
    st.subheader("Step 3: Refine the Model")
    model_options = ["Random Forest", "XGBoost", "LSTM"]
    new_model_type = st.selectbox("Change Prediction Model", model_options,
                                  index=model_options.index(model_type) if model_type in model_options else 0)
    if new_model_type != model_type:
        st.session_state.model_type = new_model_type
        st.session_state.backtest_results = {}  # Reset backtest results
        for granularity in ["minutes", "hours", "days"]:
            if granularity in st.session_state.historical_data:
                data, interval = st.session_state.historical_data[granularity]
                data_with_indicators = add_indicators(data, tuple(st.session_state.selected_indicators),
                                                      st.session_state.use_sentiment, ticker)
                if data_with_indicators is not None:
                    backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, feature_importance, model = backtest_model(
                        data_with_indicators, new_model_type, granularity,
                        st.session_state.selected_indicators,
                        st.session_state.use_sentiment, ticker
                    )
                    if backtest_results is not None:
                        st.session_state.backtest_results[granularity] = (
                            backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, model)
                        if feature_importance is not None:
                            st.session_state.feature_importance = feature_importance
        save_model_config(ticker, new_model_type, st.session_state.selected_indicators,
                          st.session_state.use_sentiment)
        st.rerun()

    # Granularity selector for refinement
    refinement_granularity = st.selectbox("Select Backtesting Granularity for Refinement",
                                          ["minutes", "hours", "days", "all"], index=2)
    st.session_state.refinement_granularity = refinement_granularity

    # ML suggestion
    st.write("**ML Suggestion**:",
             suggest_indicator(st.session_state.feature_importance, st.session_state.selected_indicators))

    if st.button("Refine Model"):
        st.session_state.backtest_results = {}
        granularities = [refinement_granularity] if refinement_granularity != "all" else ["minutes",
                                                                                          "hours",
                                                                                          "days"]
        for granularity in granularities:
            data, interval = st.session_state.historical_data[granularity]
            data_with_indicators = add_indicators(data, tuple(st.session_state.selected_indicators),
                                                  st.session_state.use_sentiment, ticker)
            if data_with_indicators is not None:
                backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, feature_importance, model = backtest_model(
                    data_with_indicators, model_type, granularity, st.session_state.selected_indicators,
                    st.session_state.use_sentiment, ticker
                )
                if backtest_results is not None:
                    st.session_state.backtest_results[granularity] = (
                        backtest_results, mse, mae, accuracy, total_return, sharpe_ratio, model)
                    if feature_importance is not None:
                        st.session_state.feature_importance = feature_importance
        save_model_config(ticker, model_type, st.session_state.selected_indicators,
                          st.session_state.use_sentiment)
        st.rerun()
    # Step 4: Time Horizon Prediction
    st.subheader("Step 4: Predict with Time Horizon")
    horizon_value = st.number_input("Time Horizon Value", min_value=1, value=3)
    horizon_unit = st.selectbox("Time Horizon Unit", ["minutes", "hours", "days"])

    if st.button("Predict with Time Horizon"):
        data, interval = st.session_state.historical_data["days"]
        model = st.session_state.backtest_results["days"][-1]  # Get the trained model
        data_with_indicators = add_indicators(data, tuple(st.session_state.selected_indicators),
                                              st.session_state.use_sentiment, ticker)
        if data_with_indicators is not None:
            predictions, prediction_id = predict_future(
                data_with_indicators, model, model_type, horizon_value, horizon_unit,
                st.session_state.selected_indicators, st.session_state.use_sentiment, ticker, interval
            )
            if predictions is not None:
                st.session_state.refined_model = (predictions, horizon_value, horizon_unit, prediction_id)
                st.rerun()

    # Display predictions and options recommendations
    if st.session_state.refined_model:
        predictions, horizon_value, horizon_unit, prediction_id = st.session_state.refined_model
        st.subheader(f"Predictions for {horizon_value} {horizon_unit}")
        st.dataframe(predictions)

        horizon_minutes = horizon_value if horizon_unit == "minutes" else horizon_value * 60 if horizon_unit == "hours" else horizon_value * 24 * 60
        result = recommend_options(ticker, predictions, horizon_minutes, prediction_id)
        if result:
            st.success(
                f"Recommended {result['type']} Option: Strike ${result['strike']:.2f}, Buy at ${result['buy_price']:.2f}, Sell at ${result['sell_price']:.2f}, Exit by {result['exit_time']}, Predicted Profit ${result['predicted_profit']:.2f}")
        else:
            st.warning(
                "No options recommendations available. Try adjusting prediction horizon or criteria.")

    # Evaluate predictions
    evaluate_predictions(ticker, st.session_state.historical_data)


with tab2:
    st.header("Analysis Log of the Third Age")
    st.write("This log tracks the steps taken, lessons learned, corrections made, and suggestions for improvement.")
    if st.session_state.analysis_log:
        for entry in st.session_state.analysis_log:
            st.subheader(entry["Step"])
            st.write(f"**Details**: {entry['Details']}")
            st.write(f"**Lessons Learned**: {entry['Lessons Learned']}")
            st.write(f"**Corrections**: {entry['Corrections']}")
            st.write(f"**Suggestions**: {entry['Suggestions']}")
            st.markdown("---")
    else:
        st.write("No analysis steps logged yet. Run the app to populate the log.")

with tab3:
    st.header("Backtesting Results")
    if st.session_state.backtest_history:
        backtest_history_df = pd.DataFrame(st.session_state.backtest_history)
        st.dataframe(backtest_history_df)
    else:
        st.write("No backtesting results available yet. Run a backtest to see results.")
with tab4:
    st.header("Prediction Outcomes")
    c.execute(
        "SELECT id, ticker, predicted_at, horizon_datetime, predicted_price, actual_price, metrics FROM predictions WHERE ticker = ?",
        (ticker,))
    predictions = c.fetchall()
    if predictions:
        prediction_data = []
        for pred in predictions:
            pred_id, ticker, predicted_at, horizon_datetime, predicted_price, actual_price, metrics = pred
            metrics_dict = json.loads(metrics) if metrics else {}
            prediction_data.append({
                "ID": pred_id,
                "Ticker": ticker,
                "Predicted At": predicted_at,
                "Horizon": horizon_datetime,
                "Predicted Price": predicted_price,
                "Actual Price": actual_price if actual_price is not None else "Pending",
                "MSE": metrics_dict.get('mse', 'N/A'),
                "MAE": metrics_dict.get('mae', 'N/A'),
                "Accuracy (%)": metrics_dict.get('accuracy', 'N/A'),
                "Total Return (%)": metrics_dict.get('total_return', 'N/A'),
                "Sharpe Ratio": metrics_dict.get('sharpe_ratio', 'N/A')
            })
        st.dataframe(pd.DataFrame(prediction_data))

        # Display options outcomes
        st.subheader("Options Outcomes")
        c.execute(
            "SELECT prediction_id, type, strike, buy_price, sell_price, exit_time, predicted_profit, actual_profit FROM options_recommendations WHERE prediction_id IN (SELECT id FROM predictions WHERE ticker = ?)",
            (ticker,))
        options = c.fetchall()
        if options:
            options_data = []
            for opt in options:
                pred_id, opt_type, strike, buy_price, sell_price, exit_time, predicted_profit, actual_profit = opt
                options_data.append({
                    "Prediction ID": pred_id,
                    "Type": opt_type,
                    "Strike": strike,
                    "Buy Price": buy_price,
                    "Sell Price": sell_price,
                    "Exit Time": exit_time,
                    "Predicted Profit": predicted_profit,
                    "Actual Profit": actual_profit if actual_profit is not None else "Pending"
                })
            st.dataframe(pd.DataFrame(options_data))
        else:
            st.write("No options recommendations available yet.")
    else:
        st.write("No predictions available yet. Run a prediction to see outcomes.")

if __name__ == "__main__":
    main()
