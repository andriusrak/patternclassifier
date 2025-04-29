import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mplfinance as mpf

# Constants
data_file = "ohlcv_stochastic.csv"
label_store = "labels.pkl"
model_store = "model.pkl"
scaler_store = "scaler.pkl"

# Ensure directories
os.makedirs("data", exist_ok=True)

@st.cache_data
def load_data():
    df = pd.read_csv(data_file, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

# Load or init labels
if os.path.exists(label_store):
    with open(label_store, "rb") as f:
        labels = pickle.load(f)
else:
    labels = {}

# Load or init model and scaler
if os.path.exists(model_store) and os.path.exists(scaler_store):
    with open(model_store, "rb") as f:
        model = pickle.load(f)
    with open(scaler_store, "rb") as f:
        scaler = pickle.load(f)
else:
    model = SGDClassifier(loss='log', max_iter=1, warm_start=True)
    scaler = StandardScaler()
    # partial_fit will be called with classes

# Sidebar controls
st.sidebar.header("Controls")
df = load_data()
timestamps = df.index
selected_ts = st.sidebar.selectbox("Select timestamp to view", timestamps)
label = st.sidebar.selectbox("Assign label", ["buy", "sell", "hold"] )
window = st.sidebar.slider("Window size (# candles)", min_value=5, max_value=100, value=30)

# Save label
if st.sidebar.button("Save Label"):
    labels[selected_ts] = label
    with open(label_store, "wb") as f:
        pickle.dump(labels, f)
    st.sidebar.success(f"Label saved for {selected_ts}")

# Plot segment around selection
start = selected_ts - pd.Timedelta(minutes=window)
end = selected_ts + pd.Timedelta(minutes=window)
df_seg = df.loc[start:end]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
mpf.plot(df_seg[['open','high','low','close']], type='candle', ax=ax1, style='charles')
ax1.set_title('Candlestick')
ax2.plot(df_seg.index, df_seg['stoch_k'], label='%K')
ax2.plot(df_seg.index, df_seg['stoch_d'], label='%D')
ax2.legend()
ax2.set_title('Stochastic Oscillator')
st.pyplot(fig)

# Training routine if any labels exist
if labels:
    X, y = [], []
    for ts, lab in labels.items():
        seg = df.loc[ts - pd.Timedelta(candles=window): ts + pd.Timedelta(candles=window)]
        seg = seg.tail(window)
        feat = []
        # Flatten last N stochastic values
        feat.extend(seg['stoch_k'].values)
        feat.extend(seg['stoch_d'].values)
        # Optional OHLCV summary context
        feat.extend([seg['open'].mean(), seg['close'].mean(), seg['volume'].sum()])
        X.append(feat)
        y.append(lab)
    X = np.array(X)
    y = np.array(y)
    scaler.partial_fit(X)
    Xs = scaler.transform(X)
    model.partial_fit(Xs, y, classes=["buy","sell","hold"])
    with open(model_store, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_store, "wb") as f:
        pickle.dump(scaler, f)
    st.write("Model updated with latest labels.")

# Prediction interface
if st.sidebar.button("Predict for current segment"):
    seg = df_seg.tail(window)
    feat = list(seg['stoch_k'].values) + list(seg['stoch_d'].values)
    feat.extend([seg['open'].mean(), seg['close'].mean(), seg['volume'].sum()])
    Xp = scaler.transform([feat])
    pred = model.predict(Xp)[0]
    st.sidebar.info(f"Predicted label: {pred}")

# Download labeled data
if labels:
    records = []
    for ts, lab in labels.items():
        row = df.loc[ts]
        records.append({
            'timestamp': ts,
            'open': row['open'], 'high': row['high'],
            'low': row['low'], 'close': row['close'],
            'volume': row['volume'],
            'stoch_k': row['stoch_k'], 'stoch_d': row['stoch_d'],
            'label': lab
        })
    df_labels = pd.DataFrame(records)
    csv = df_labels.to_csv(index=False).encode('utf-8')
    st.download_button("Download Labeled Data", data=csv,
                       file_name='labeled_data.csv', mime='text/csv')
