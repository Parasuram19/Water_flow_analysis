import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['created_at'] = pd.to_datetime(data['created_at'])
    data.sort_values(by='created_at', inplace=True)
    data.set_index('created_at', inplace=True)
    return data

# Constants
TANK_CAPACITY = 500  # Tank capacity in liters
PUMP_ENERGY_RATE = 0.2  # kWh per liter
COST_PER_KWH = 5  # INR per kWh
WATER_FLOW_COLUMN = 'Water flow'

# Section 1: Load and Display Data
st.title("Water Tank Management with LSTM Forecasting, Anomaly Detection, and Optimization")
# st.sidebar.title("Settings")
file_path = 'dataset.csv'

st.header("1. Data Overview")
data = load_data(file_path)
st.write("### Raw Data")
st.dataframe(data)

# Section 2: Anomaly Detection
st.header("2. Anomaly Detection")
st.write("Detecting anomalies in water flow data...")
data['anomaly'] = (data['Water flow'].diff().abs() > 10) | (data['Water flow'] < 0)  # Customize threshold
anomalies = data[data['anomaly']]

st.write("### Anomalies Detected")
if anomalies.empty:
    st.success("No anomalies detected!")
else:
    st.warning(f"{len(anomalies)} anomalies detected.")
    st.dataframe(anomalies)

# Plot water flow with anomalies
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data['Water flow'], label='Water Flow (ml/s)', alpha=0.6)
ax.scatter(anomalies.index, anomalies['Water flow'], color='red', label='Anomalies', zorder=5)
ax.set_title("Water Flow with Anomalies")
ax.set_xlabel("Time")
ax.set_ylabel("Water Flow (ml/s)")
ax.legend()
st.pyplot(fig)


# Section 3: Preprocess Data for LSTM
st.header("3. Time-Series Forecasting with LSTM")
st.write("Preparing data for LSTM model...")

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[[WATER_FLOW_COLUMN]])

# Create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 10
X, y = create_sequences(data_scaled, TIME_STEPS)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and Train LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

st.write("### Training LSTM Model...")
with st.spinner("Training LSTM model... This may take a while."):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)
st.success("LSTM model trained!")

# Predictions
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot predictions
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test_rescaled, label="Actual Flow Rate")
ax.plot(predictions_rescaled, label="Predicted Flow Rate", linestyle="--")
ax.set_title("LSTM Predictions vs Actual Water Flow")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Flow Rate (liters/sec)")
ax.legend()
st.pyplot(fig)

# Section 4: Overflow Prediction
st.header("4. Overflow Prediction")
cumulative_inflow = predictions_rescaled.cumsum()
cumulative_inflow = np.clip(cumulative_inflow, 0, TANK_CAPACITY)

overflow_idx = np.argmax(cumulative_inflow > TANK_CAPACITY) if np.any(cumulative_inflow > TANK_CAPACITY) else None
if overflow_idx is not None:
    st.warning(f"Tank overflow predicted at time step {overflow_idx}.")
else:
    st.success("No overflow predicted during the forecasted period.")

# Plot cumulative inflow
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cumulative_inflow, label="Predicted Cumulative Inflow")
ax.axhline(TANK_CAPACITY, color='red', linestyle="--", label="Tank Capacity")
ax.set_title("Cumulative Inflow Prediction")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Water Volume (liters)")
ax.legend()
st.pyplot(fig)

# Section 5: Energy and Cost Analysis
st.header("5. Energy and Cost Analysis")

# Calculate energy and cost
predicted_total_flow = predictions_rescaled.sum()
energy_usage = predicted_total_flow * PUMP_ENERGY_RATE  # kWh
cost = energy_usage * COST_PER_KWH  # INR

st.write(f"### Predicted Total Water Flow: {predicted_total_flow:.2f} liters")
st.write(f"### Estimated Energy Usage: {energy_usage:.2f} kWh")
st.write(f"### Estimated Pumping Cost: ₹{cost:.2f}")

# Section 6: Optimization
st.header("6. Optimization for Water Usage")

def optimize_water_usage(flow_rates, tank_capacity):
    optimized_rates = []
    cumulative_flow = 0
    for rate in flow_rates:
        if cumulative_flow + rate > tank_capacity:
            optimized_rate = max(0, tank_capacity - cumulative_flow)
        else:
            optimized_rate = rate
        optimized_rates.append(optimized_rate)
        cumulative_flow += optimized_rate
    return optimized_rates

optimized_flow = optimize_water_usage(predictions_rescaled.flatten(), TANK_CAPACITY)

# Plot optimized flow
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(predictions_rescaled, label="Predicted Flow Rate", alpha=0.6)
ax.plot(optimized_flow, label="Optimized Flow Rate", linestyle="--")
ax.set_title("Optimized Water Flow Rate")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Flow Rate (liters/sec)")
ax.legend()
st.pyplot(fig)

st.write("Optimization ensures water flow stays within tank capacity and prevents overflow.")
