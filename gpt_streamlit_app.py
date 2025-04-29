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
data_file = "data/ohlcv_stochastic.csv"
label_store = "labels.pkl"
model_store = "model.pkl"
scaler_store = "scaler.pkl"

# Ensure directories
os.makedirs("data", exist_ok=True)

@st.cache_data
def load_data():
    df = pd.read_csv(data_file, parse_dates=["timestamp"] )
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
    # We will call partial_fit with classes defined later

# Sidebar controls
st.sidebar.header("Controls")
timestamp = st.sidebar.selectbox("Select timestamp to view", load_data().index)
label = st.sidebar.selectbox("Assign label", ["buy", "sell", "hold"])
if st.sidebar.button("Save Label"):
    labels[timestamp] = label
    with open(label_store, "wb") as f:
        pickle.dump(labels, f)
    st.sidebar.success(f"Label saved for {timestamp}")

# Plotting
window = st.sidebar.slider("Window size (minutes)", min_value=1, max_value=120, value=30)

df = load_data()
start = timestamp - pd.Timedelta(minutes=window)
end = timestamp + pd.Timedelta(minutes=window)
df_segment = df.loc[start:end]

# Candlestick
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
mpf.plot(df_segment[['open','high','low','close']], type='candle', ax=axes[0], style='charles')
axes[0].set_title('Candlestick')

# Stochastic oscillator: assume %K and %D in columns 'stoch_k', 'stoch_d'
axes[1].plot(df_segment.index, df_segment['stoch_k'], label='%K')
axes[1].plot(df_segment.index, df_segment['stoch_d'], label='%D')
axes[1].hlines([20, 80], df_segment.index.min(), df_segment.index.max(), linestyles='--')
axes[1].legend()
axes[1].set_title('Stochastic Oscillator')

st.pyplot(fig)

# Prepare training if labels exist
if labels:
    # Build training set
    X = []
    y = []
    for ts, lab in labels.items():
        seg = df.loc[ts - pd.Timedelta(minutes=window): ts + pd.Timedelta(minutes=window)]
        feat = []
        # flatten OHLCV means and stochastic stats
        feat.extend([seg['open'].mean(), seg['high'].max(), seg['low'].min(), seg['close'].mean(), seg['volume'].sum()])
        feat.extend([seg['stoch_k'].mean(), seg['stoch_d'].mean()])
        X.append(feat)
        y.append(lab)
    X = np.array(X)
    y = np.array(y)
    # Scale
    scaler.partial_fit(X)
    Xs = scaler.transform(X)
    # Incremental fit
    model.partial_fit(Xs, y, classes=["buy","sell","hold"])
    # Save model and scaler
    with open(model_store, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_store, "wb") as f:
        pickle.dump(scaler, f)
    st.write("Model updated with latest labels.")

# Prediction interface
if st.sidebar.button("Predict for current segment"):
    feat = []
    seg = df_segment
    feat.extend([seg['open'].mean(), seg['high'].max(), seg['low'].min(), seg['close'].mean(), seg['volume'].sum()])
    feat.extend([seg['stoch_k'].mean(), seg['stoch_d'].mean()])
    Xp = scaler.transform([feat])
    pred = model.predict(Xp)[0]
    st.sidebar.info(f"Predicted label: {pred}")

# Option to download labeled data
if labels:
    # Build a labeled DataFrame for export
    export_data = []
    for ts, lab in labels.items():
        row = df.loc[ts]
        export_data.append({
            "timestamp": ts,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "stoch_k": row["stoch_k"],
            "stoch_d": row["stoch_d"],
            "label": lab
        })
    labeled_df = pd.DataFrame(export_data)
    csv = labeled_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Labeled Data", data=csv, file_name="labeled_data.csv", mime="text/csv")

