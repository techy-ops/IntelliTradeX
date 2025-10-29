import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Choose stock symbol
stock_symbol = input("Enter stock symbol (e.g. AAPL, TSLA, INFY): ").upper()

# Step 2: Fetch data
data = yf.download(stock_symbol, period="1y", interval="1d")
data = data.dropna()

if data.empty:
    print(f"No data found for symbol '{stock_symbol}'. Please try another one.")
    exit()

close_prices = data[['Close']].values

# Step 3: Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Step 4: Create sequences
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i-sequence_length:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 5: Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train Model
model.fit(X, y, epochs=20, batch_size=32)

# Step 7: Test on last 60 days
test_data = scaled_prices[-(sequence_length+30):]
X_test, y_test = [], scaled_prices[-30:]

for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i-sequence_length:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Step 8: Plot Predictions and Save Instead of Displaying
plt.figure(figsize=(10, 5))
plt.plot(data.index[-30:], scaler.inverse_transform(y_test), label="Actual Price", color='blue')
plt.plot(data.index[-30:], predictions, label="Predicted Price", color='red')
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()

# Save plot to file instead of showing
plot_filename = f"{stock_symbol}_prediction_plot.png"
plt.savefig(plot_filename)
plt.close()

print(f"✅ Model saved as {stock_symbol}_lstm_model.h5")
print(f"📊 Prediction plot saved as {plot_filename}")
