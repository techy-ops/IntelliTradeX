import tensorflow as tf
import numpy as np
import joblib
import yfinance as yf

# Load trained model and scaler
model = tf.keras.models.load_model("hybrid_model_AAPL.h5")
scaler = joblib.load("scaler_AAPL.pkl")

# Fetch the latest 90 days of AAPL stock data
data = yf.download("AAPL", period="90d", interval="1d")[['Close']].values

# Scale data
scaled_data = scaler.transform(data)

# Prepare input (last 60 days)
X_input = scaled_data[-60:].reshape(1, 60, 1)

# Predict next-day price
prediction = model.predict(X_input)
predicted_price = scaler.inverse_transform(prediction)

print(f"📈 Predicted next AAPL price: ${predicted_price[0][0]:.2f}")
 