import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
import random


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- FIX SEED ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- CONFIG ---
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
LOOK_BACK = 35 #35
MODELS_DIR = '../../models'
PLOTS_DIR = '../../plots'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. DATA PREP
# ==========================================
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

# Split
train_size = int(len(df) * 0.85)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Scale
features = ['Close', 'SMA_7', 'Volatility']
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.gz'))

def create_sequences(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOK_BACK)
X_test, y_test = create_sequences(test_scaled, LOOK_BACK)

# Validation Split
val_split = int(len(X_train) * 0.9)
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

print(f"Shapes -> Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 2. MODEL (LSTM)
# ==========================================
print("🧠 2. Обучение (LSTM)...")

from tensorflow.keras.callbacks import EarlyStopping # Увери се, че това е импортирано горе!

model = Sequential()

model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))

# Намаляваме леко скоростта на учене за по-голяма точност
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='huber')

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

model.save(os.path.join(MODELS_DIR, 'bitcoin_lstm.keras'))
print("✅ LSTM моделът е запазен.")

# ==========================================
# 3. EVALUATION (USD METRICS)
# ==========================================
print("📈 3. Оценка...")

preds_scaled = model.predict(X_test)

def inverse_transform_predictions(predictions, scaler):
    dummy = np.zeros((len(predictions), 3))
    dummy[:, 0] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

preds_real = inverse_transform_predictions(preds_scaled, scaler)

dummy_y = np.zeros((len(y_test), 3))
dummy_y[:, 0] = y_test
y_test_real = scaler.inverse_transform(dummy_y)[:, 0]

# --- МЕТРИКИ В ДОЛАРИ ($) ---
mae = mean_absolute_error(y_test_real, preds_real)
rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
r2 = r2_score(y_test_real, preds_real)

print(f"\n=================================")
print(f"🏆 РЕЗУЛТАТИ (LSTM):")
print(f"MAE (Средна грешка): ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"=================================")

# Графика
plt.figure(figsize=(14, 7))
plt.plot(y_test_real, label='Real Price', color='black')
plt.plot(preds_real, label='AI Prediction', color='green')
plt.title(f'LSTM Prediction (MAE: ${mae:.0f})')
plt.xlabel('Time (Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'lstm_prediction_usd.png'))
print("✅ Графиката 'lstm_prediction_usd.png' е запазена.")
plt.show()