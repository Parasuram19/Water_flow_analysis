import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
# Replace 'file_path.csv' with your actual data file
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['created_at'] = pd.to_datetime(data['created_at'])
    data.sort_values(by='created_at', inplace=True)
    data.set_index('created_at', inplace=True)
    data = data.resample('16s').mean()  # Resample to 16-second intervals
    data.fillna(0, inplace=True)
    data['cumulative_water'] = data['Amount of water'].cumsum()  # Tank water level
    return data

data = load_data("dataset.csv")

# Constants
tank_capacity = 500  # Tank capacity in ml
energy_per_ml = 0.05  # Example: Energy usage in joules per ml of water pumped

# Page Title
st.title("Water Tank Monitoring Dashboard")

# Section 1: Data Overview
st.header("1. Data Overview")
st.write("### Raw Data")
st.dataframe(data)

# Section 2: Anomaly Detection
st.header("2. Anomaly Detection")
# Detect anomalies (spikes, drops, negative values)
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

# Section 3: Overflow Prediction
st.header("3. Overflow Prediction")
# Predict when tank will overflow
data['net_flow'] = data['Water flow']  # Assuming inflow > outflow
cumulative_flow = data['net_flow'].cumsum()

overflow_time = None
if cumulative_flow.max() > tank_capacity:
    overflow_idx = cumulative_flow[cumulative_flow > tank_capacity].idxmin()
    overflow_time = (overflow_idx - data.index[0]).total_seconds() / 60  # Minutes until overflow
    st.warning(f"Tank will overflow in approximately {overflow_time:.2f} minutes.")
else:
    st.success("No overflow predicted.")

# Plot cumulative water and tank capacity
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, cumulative_flow, label='Cumulative Water (ml)')
ax.axhline(tank_capacity, color='red', linestyle='--', label='Tank Capacity')
if overflow_time:
    ax.axvline(overflow_idx, color='orange', linestyle=':', label='Predicted Overflow Time')
ax.set_title("Tank Water Level and Overflow Prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative Water (ml)")
ax.legend()
st.pyplot(fig)

# Section 4: Energy/Cost Analysis
st.header("4. Energy and Cost Analysis")
# Calculate energy usage
data['energy_used'] = data['Amount of water'] * energy_per_ml
total_energy = data['energy_used'].sum()

# Display energy statistics
st.write("### Energy Statistics")
st.metric("Total Energy Used (Joules)", f"{total_energy:.2f}")
st.metric("Energy Cost Estimate ($)", f"${total_energy * 0.001:.2f}")  # Assuming $0.001 per joule

# Plot energy usage over time
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data['energy_used'], label='Energy Used (Joules)', color='green')
ax.set_title("Energy Usage Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Energy (Joules)")
ax.legend()
st.pyplot(fig)

# Section 5: Tank Status and Predictions
st.header("5. Tank Status and Predictions")
# Predict when tank will be empty
X = np.arange(len(data)).reshape(-1, 1)
y = data['cumulative_water'].values
model = LinearRegression()
model.fit(X, y)

# Calculate time to empty
time_to_empty = (tank_capacity - model.intercept_) / model.coef_[0]
time_to_empty_seconds = int(time_to_empty * 16)

st.write("### Tank Predictions")
if data['cumulative_water'].iloc[-1] <= 0:
    st.success("Tank is already empty.")
else:
    st.warning(f"Tank will be empty in approximately {time_to_empty_seconds / 60:.2f} minutes.")

