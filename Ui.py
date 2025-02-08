import streamlit as st
import pandas as pd
import numpy as np
import talib
import ccxt
from reservoirpy import ESN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# Constants
INITIAL_BALANCE = 10000

# Step 1: Fetch Historical Data
def fetch_historical_data(symbol, timeframe, limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    data = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
    return data

# Step 2: Calculate Technical Indicators
def calculate_indicators(data):
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data.dropna(inplace=True)
    return data

# Step 3: Train Echo State Network (ESN)
def train_esn(data):
    X = data[['Close']].values
    y = data['Close'].shift(-1).dropna().values
    X = X[:-1]
    esn = ESN(n_inputs=1, n_outputs=1, n_reservoir=100, spectral_radius=0.9)
    esn.fit(X, y)
    return esn

# Step 4: Trading Logic
def trading_logic(data, esn):
    signals = []
    for i in range(len(data)):
        future_price = esn.predict(data[['Close']].iloc[i].values.reshape(1, -1))[0][0]
        if future_price > data['Close'].iloc[i] and data['SMA_50'].iloc[i] > data['EMA_20'].iloc[i]:
            signals.append(1)  # Buy
        elif future_price < data['Close'].iloc[i] and data['SMA_50'].iloc[i] < data['EMA_20'].iloc[i]:
            signals.append(-1)  # Sell
        else:
            signals.append(0)  # Hold
    return signals

# Step 5: Backtesting
def backtest(data, signals, balance):
    for i in range(len(signals)):
        if signals[i] == 1:  # Buy
            balance += balance * 0.01  # Simulate profit
        elif signals[i] == -1:  # Sell
            balance -= balance * 0.01  # Simulate loss
    return balance

# Streamlit App
def main():
    st.title("AI Trading Bot 🚀")
    st.sidebar.header("Settings")

    # User Inputs
    symbol = st.sidebar.text_input("Symbol (e.g., BTC/USDT)", "BTC/USDT")
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])
    risk_percent = st.sidebar.slider("Risk Percentage", 0.1, 5.0, 1.0)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.1, 10.0, 2.0)
    take_profit = st.sidebar.slider("Take Profit (%)", 0.1, 10.0, 5.0)

    if st.sidebar.button("Run Bot"):
        st.write("Fetching data...")
        data = fetch_historical_data(symbol, timeframe)
        st.write("Calculating indicators...")
        data = calculate_indicators(data)
        st.write("Training ESN...")
        esn = train_esn(data)
        st.write("Generating signals...")
        signals = trading_logic(data, esn)
        st.write("Backtesting...")
        final_balance = backtest(data, signals, INITIAL_BALANCE)
        st.success(f"Initial Balance: ${INITIAL_BALANCE}")
        st.success(f"Final Balance: ${final_balance:.2f}")

        # Display Results
        st.subheader("Trading Signals")
        st.write(data.tail(10))

        st.subheader("Performance Metrics")
        st.write(f"Total Trades: {len(signals)}")
        st.write(f"Profit/Loss: ${final_balance - INITIAL_BALANCE:.2f}")

# Run the app
if __name__ == "__main__":
    main()