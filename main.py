import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Streamlit app title
st.title("Stock Predictor with Options")

# User input for stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", "AAPL")

# Function to fetch minute-level data in chunks
def fetch_minute_data(ticker_symbol, total_days=28, chunk_days=7):
    stock = yf.Ticker(ticker_symbol)
    end_date = datetime.now()
    all_data = pd.DataFrame()

    # Fetch data in chunks
    for i in range(total_days // chunk_days):
        start_date = end_date - timedelta(days=chunk_days)
        chunk_data = stock.history(start=start_date, end=end_date, interval="1m")
        if not chunk_data.empty:
            # Append to the main DataFrame
            all_data = pd.concat([chunk_data, all_data])
        end_date = start_date

    # Remove duplicates (if any) based on timestamp
    if not all_data.empty:
        all_data = all_data[~all_data.index.duplicated(keep="first")]
        all_data.sort_index(inplace=True)
    return all_data

if ticker:
    # Fetch stock data
    try:
        stock = yf.Ticker(ticker)

        # Fetch extended minute-level data (e.g., 28 days)
        st.subheader("Fetching Minute-Level Data")
        minute_data = fetch_minute_data(ticker, total_days=28, chunk_days=7)
        hour_data = stock.history(period="60d", interval="1h")
        day_data = stock.history(period="60d", interval="1d")

        # Check if data is available
        if minute_data.empty and hour_data.empty and day_data.empty:
            st.error("No data available for this ticker. Market might be closed, or the ticker is invalid.")
        else:
            # Display minute-level data
            st.subheader(f"Minute-Level Data for {ticker} (Last 28 Days)")
            if not minute_data.empty:
                st.write(minute_data.tail())
            else:
                st.write("Minute-level data unavailable (market may be closed).")

            # Display daily data
            st.subheader(f"Daily Data for {ticker}")
            st.write(day_data.tail())

            # --- Technical Indicators ---
            st.subheader("Technical Indicators")

            # Calculate SMA (20-day), RSI (14-day), and MACD using daily data
            day_data["SMA_20"] = SMAIndicator(close=day_data["Close"], window=20).sma_indicator()
            day_data["RSI_14"] = RSIIndicator(close=day_data["Close"], window=14).rsi()
            macd = MACD(close=day_data["Close"])
            day_data["MACD"] = macd.macd()
            day_data["MACD_Signal"] = macd.macd_signal()

            # Plot 1: Stock Price with SMA
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=day_data.index, y=day_data["Close"], mode="lines", name="Close Price"))
            fig_price.add_trace(go.Scatter(x=day_data.index, y=day_data["SMA_20"], mode="lines", name="SMA 20", line=dict(color="orange")))
            fig_price.update_layout(title=f"{ticker} Close Price and 20-Day SMA", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig_price)

            # Plot 2: RSI Gauge
            latest_rsi = day_data["RSI_14"].iloc[-1] if not day_data["RSI_14"].empty else 50
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_rsi,
                title={"text": "RSI (14-Day)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "red"},  # Oversold
                        {"range": [30, 70], "color": "gray"},  # Neutral
                        {"range": [70, 100], "color": "green"},  # Overbought
                    ],
                }
            ))
            st.plotly_chart(fig_rsi)

            # --- Sentiment Analysis ---
            st.subheader("Sentiment Analysis")

            # Placeholder news headline
            news_headline = f"{ticker} announces a major product launch next quarter!"
            analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = analyzer.polarity_scores(news_headline)

            # Display sentiment scores
            st.write("Sample News Headline:", news_headline)
            st.write("Sentiment Scores:", sentiment_scores)

            # Plot 3: Sentiment Bar Chart
            sentiment_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive", "Compound"],
                "Score": [sentiment_scores["neg"], sentiment_scores["neu"], sentiment_scores["pos"], sentiment_scores["compound"]]
            })
            fig_sentiment = px.bar(sentiment_df, x="Score", y="Sentiment", orientation="h", title="Sentiment Analysis")
            st.plotly_chart(fig_sentiment)

            # --- XGBoost Predictions ---
            st.subheader("Price Predictions")

            # Prepare data for prediction (using daily data)
            if not day_data.empty:
                # Feature engineering: Use SMA, RSI, MACD, Volume, and Sentiment as features
                day_data["Sentiment"] = sentiment_scores["compound"]  # Same sentiment for all rows (placeholder)
                df = day_data[["Close", "SMA_20", "RSI_14", "MACD", "MACD_Signal", "Volume", "Sentiment"]].dropna()
                X = df[["SMA_20", "RSI_14", "MACD", "MACD_Signal", "Volume", "Sentiment"]]
                y = df["Close"]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Train XGBoost model
                model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse:.2f}")

                # Predict the next day's price
                last_features = X.iloc[-1:].values
                next_day_pred = model.predict(last_features)[0]
                st.write(f"Predicted Next Day's Closing Price: ${next_day_pred:.2f}")

                # Plot actual vs predicted
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=X_test.index, y=y_test, mode="lines", name="Actual Price"))
                fig_pred.add_trace(go.Scatter(x=X_test.index, y=y_pred, mode="lines", name="Predicted Price", line=dict(color="red")))
                fig_pred.update_layout(title=f"{ticker} Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(fig_pred)

                # Feature Importance
                st.subheader("Feature Importance")
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": feature_importance
                }).sort_values(by="Importance", ascending=False)
                fig_importance = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Feature Importance in XGBoost Model")
                st.plotly_chart(fig_importance)
            else:
                st.write("Not enough data to make predictions.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # extra line for notable change in file information