import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
import random

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- REPRODUCIBILITY ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- CONFIG ---
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
LOOK_BACK = 30  # Shorter window to catch trends faster
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. DATA PREP (Focus on Returns & Volatility)
# ==========================================
print("📥 1. Ingesting Data & Engineering Features...")
df = yf.download(SYMBOL, start=START_DATE)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# A. Target: Log Returns (The "Honest" Target)
# We predict how much the price will change %-wise
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

# B. Volatility (Risk context)
df['Volatility'] = df['Log_Ret'].rolling(window=14).std()

# C. Momentum (RSI)
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'])

# D. Volume (Log scaled)
df['Log_Vol'] = np.log(df['Volume'].replace(0, 1))

# Drop NaNs created by rolling windows
df.dropna(inplace=True)

# Define Target: Next Day's Return
df['Target_Next_Ret'] = df['Log_Ret'].shift(-1)
df.dropna(inplace=True)

# --- SPLITTING ---
train_size = int(len(df) * 0.85)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# --- SCALING ---
# We use RobustScaler for inputs to handle crypto outliers
feature_cols = ['Log_Ret', 'Volatility', 'RSI', 'Log_Vol']
target_col = ['Target_Next_Ret']

scaler_x = RobustScaler()
scaler_y = MinMaxScaler(feature_range=(-1, 1)) # Returns are small, stretch them to -1 to 1

scaler_x.fit(train_df[feature_cols])
scaler_y.fit(train_df[target_col])

joblib.dump(scaler_x, os.path.join(MODELS_DIR, 'scaler_x.gz'))
joblib.dump(scaler_y, os.path.join(MODELS_DIR, 'scaler_y.gz'))

X_train_scaled = scaler_x.transform(train_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[target_col])
X_test_scaled = scaler_x.transform(test_df[feature_cols])
y_test_scaled = scaler_y.transform(test_df[target_col])

# --- SEQUENCING ---
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, LOOK_BACK)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, LOOK_BACK)

# Validation Split
val_split = int(len(X_train) * 0.9)
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

print(f"📊 Features used: {feature_cols}")
print(f"📊 Training shape: {X_train.shape}")

# ==========================================
# 2. ARCHITECTURE: ATTENTION-LSTM (Consulting Brain)
# ==========================================
print("🧠 2. Training Attention-LSTM Regressor...")

# Input Layer
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM Layers (Return Sequences = True for Attention)
lstm_out = LSTM(64, return_sequences=True)(inputs)
lstm_out = Dropout(0.3)(lstm_out)

# Attention Mechanism
# Focuses on which days in the past 30 days matter most
query_value_attention_seq = Attention()([lstm_out, lstm_out])

# Global Pooling (Squash time dimension)
context_vector = GlobalAveragePooling1D()(query_value_attention_seq)

# Interpretive Dense Layers
x = Dense(32, activation='relu')(context_vector)
x = Dropout(0.2)(x)

# Output Layer (Linear for Regression)
outputs = Dense(1, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model.save(os.path.join(MODELS_DIR, 'bitcoin_consultant.keras'))
print("✅ Model saved.")

# ==========================================
# 3. RECONSTRUCTION & EVALUATION
# ==========================================
print("📈 3. Reconstructing Price & Evaluating...")

# A. Predict Returns (Scaled)
pred_scaled = model.predict(X_test)

# B. Inverse Scale Returns
pred_returns = scaler_y.inverse_transform(pred_scaled).flatten()
real_returns = scaler_y.inverse_transform(y_test).flatten()

# C. Reconstruct Prices ($)
# We need the base prices (Close price of the day BEFORE the prediction)
# Index logic: Test data starts at `train_size`. Sequences consume `LOOK_BACK`.
# So first prediction corresponds to `train_size + LOOK_BACK`.
# Base price for that is `train_size + LOOK_BACK - 1`.

base_prices = df['Close'].iloc[train_size + LOOK_BACK - 1 : -1].values

# Align lengths
min_len = min(len(base_prices), len(pred_returns))
base_prices = base_prices[:min_len]
pred_returns = pred_returns[:min_len]
real_returns = real_returns[:min_len]

# Apply returns to base prices
pred_prices = base_prices * (np.exp(pred_returns)) # Using exp for Log Returns
real_prices = base_prices * (np.exp(real_returns))

# D. Metrics
mae = mean_absolute_error(real_prices, pred_prices)
rmse = np.sqrt(mean_squared_error(real_prices, pred_prices))

# Directional Accuracy (Did we get the sign right?)
# +1 if signs match, 0 if not
direction_match = np.sign(pred_returns) == np.sign(real_returns)
dir_accuracy = np.mean(direction_match) * 100

print(f"\n=================================")
print(f"📊 CONSULTANT REPORT:")
print(f"MAE (Error in $): ${mae:.2f}")
print(f"Directional Accuracy: {dir_accuracy:.2f}%")
print(f"=================================")

# E. Visualization
plt.figure(figsize=(14, 7))
subset = 150 # Last 150 days
plt.plot(real_prices[-subset:], label='Actual Market Price', color='black', alpha=0.8)
plt.plot(pred_prices[-subset:], label='AI Prognosis (Attention-Based)', color='blue', linestyle='--', linewidth=1.5)

# Mark massive errors (Anomalies)
errors = np.abs(real_prices[-subset:] - pred_prices[-subset:])
threshold = np.mean(errors) + 2 * np.std(errors)
anomalies_idx = np.where(errors > threshold)[0]

if len(anomalies_idx) > 0:
    plt.scatter(anomalies_idx, real_prices[-subset:][anomalies_idx], color='red', label='Unexpected Volatility', zorder=5)

plt.title(f'AI Consultant Prognosis\nMAE: ${mae:.0f} | Directional Acc: {dir_accuracy:.1f}%')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'consultant_prediction.png'))
print("✅ Graph saved.")