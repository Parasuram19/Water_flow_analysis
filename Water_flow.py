import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
# Load the data (Replace 'file_path.csv' with your actual file path)
data = pd.read_csv("dataset.csv")

# Parse datetime and sort by time
data['created_at'] = pd.to_datetime(data['created_at'])
data.sort_values(by='created_at', inplace=True)

# Ensure 16-second intervals and fill missing data
data.set_index('created_at', inplace=True)
data = data.resample('16s').mean()  # Resample to 16-second intervals
data.fillna(0, inplace=True)

# Add cumulative water in the tank
data['cumulative_water'] = data['Amount of water'].cumsum()

# Tank capacity
tank_capacity = 500  # ml
# Linear Regression to predict empty time
X = np.arange(len(data)).reshape(-1, 1)  # Time index
y = data['cumulative_water'].values

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict when the water will be zero
time_to_empty = (tank_capacity - model.intercept_) / model.coef_[0]
time_to_empty_seconds = int(time_to_empty * 16)  # Adjust to 16-second intervals

# print(f"Tank will be empty in approximately {time_to_empty_seconds / 60:.2f} minutes.")

# Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(data.index, data['cumulative_water'], label='Cumulative Water (Actual)')
# plt.plot(data.index, model.predict(X), label='Prediction', linestyle='--')
# plt.axhline(tank_capacity, color='red', linestyle=':', label='Tank Capacity')
# plt.title("Tank Water Prediction")
# plt.xlabel("Time")
# plt.ylabel("Cumulative Water (ml)")
# plt.legend()
# plt.show()
# Predict water flow using rolling average
data['rolling_flow'] = data['Water flow'].rolling(window=5).mean()

# plt.figure(figsize=(10, 5))
# plt.plot(data.index, data['Water flow'], label='Water Flow (Actual)', alpha=0.6)
# plt.plot(data.index, data['rolling_flow'], label='Rolling Average (Smoothed)', linestyle='--')
# plt.title("Water Flow Over Time")
# plt.xlabel("Time")
# plt.ylabel("Flow Rate (ml/s)")
# plt.legend()
# plt.show()

st.title("Water Tank Monitoring Dashboard")

st.write("### Current Data Overview")
st.dataframe(data)

# Plot Tank Water Level
st.write("### Tank Water Level Prediction")
st.line_chart(data[['cumulative_water']])

# Plot Water Flow
st.write("### Water Flow Over Time")
st.line_chart(data[['Water flow', 'rolling_flow']])

# Tank Status
if data['cumulative_water'].iloc[-1] >= tank_capacity:
    st.success("Tank is full!")
else:
    st.warning(f"Tank is at {data['cumulative_water'].iloc[-1]} ml out of {tank_capacity} ml.")

# Time to Empty Prediction
st.write(f"Tank will be empty in approximately {time_to_empty_seconds / 60:.2f} minutes.")

