import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import tensorflow as tf
import random

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- FIX SEED ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- CONFIG ---
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
LOOK_BACK = 35
MODELS_DIR = '../models'
PLOTS_DIR = '../plots'

os.makedirs(MODELS_DIR, exist_ok=True)

print("📥 1. Сваляне и обработка...")

df = yf.download(SYMBOL, start=START_DATE)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Feature Engineering
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['Volatility'] = df['Close'].rolling(window=7).std()
df.dropna(inplace=True)

# Target
df['Target'] = df['Close']

# Scale (на 100% от данните)
features = ['Close', 'SMA_7', 'Volatility'] # <--- 3 ПРИЗНАКА
scaler = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler.fit_transform(df[features])

joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.gz'))

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(data_scaled, LOOK_BACK)

# Валидационен сплит от края (10%)
val_split = int(len(X_all) * 0.9)
X_train, y_train = X_all[:val_split], y_all[:val_split]
X_val, y_val = X_all[val_split:], y_all[val_split:]

print(f"Shapes -> Train: {X_train.shape}")

# ==========================================
# 2. MODEL (LSTM)
# ==========================================
print("🧠 2. Обучение (LSTM)...")

model = Sequential()

model.add(LSTM(units=128, return_sequences=True, input_shape=(LOOK_BACK, 3))) # <--- 3 ПРИЗНАКА
model.add(Dropout(0.1))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='huber')

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

model.save(os.path.join(MODELS_DIR, 'bitcoin_lstm_live.keras'))
print("✅ LSTM моделът е запазен.")