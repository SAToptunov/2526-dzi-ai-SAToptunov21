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
from tensorflow.keras.callbacks import EarlyStopping

# --- ЗАKОВАВАНЕ НА СЛУЧАЙНОСТТА ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- CONFIG ---
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
LOOK_BACK = 20
MODELS_DIR = '../models'

os.makedirs(MODELS_DIR, exist_ok=True)

print("==================================================")
print("🚀 СТАРТИРАНЕ НА ПРОДУКЦИОННО ОБУЧЕНИЕ (100% ДАННИ)")
print("==================================================")

# 1. Сваляне на данните (до днешна дата)
print("📥 1. Сваляне и обработка...")
df = yf.download(SYMBOL, start=START_DATE)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Feature Engineering
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['Volatility'] = df['Close'].rolling(window=7).std()

# НОВО: Добавяме Momentum (Промяната спрямо преди 3 дни)
# Това казва на модела "Накъде е засилена цената в момента"
df['Momentum'] = df['Close'] - df['Close'].shift(3)

df.dropna(inplace=True)

# И не забравяй да добавиш 'Momentum' в features за скалирането:
features = ['Close', 'SMA_7', 'Volatility', 'Momentum']

print(f"📊 Налични дни за обучение: {len(df)}")

# 2. Скалиране върху ВСИЧКИ данни
scaler = MinMaxScaler(feature_range=(0, 1))

# ВАЖНО: Вече няма Train/Test split. Скалираме 100% от данните.
data_scaled = scaler.fit_transform(df[features])

# Запазваме продукционния скалер
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.gz'))
print("✅ Скалерът е запазен.")

# 3. Sequencing
def create_sequences(data, look_back=60):
    X, y = [],[]
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(data_scaled, LOOK_BACK)

# Отделяме само последните 5% като Validation, за да не "гръмне" EarlyStopping,
# но реално моделът вижда данните до самия им край.
val_split = int(len(X_all) * 0.95)
X_train, y_train = X_all[:val_split], y_all[:val_split]
X_val, y_val = X_all[val_split:], y_all[val_split:]

# 4. Обучение на Модела
print("🧠 2. Обучение на финалния модел...")

model = Sequential()

# Поправена структура
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(units=64, return_sequences=True))

model.add(LSTM(units=32, return_sequences=False))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='huber')

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# 5. Запазване
model.save(os.path.join(MODELS_DIR, 'bitcoin_lstm_live.keras'))
print("\n==================================================")
print("✅ ПРОДУКЦИОННИЯТ МОДЕЛ Е ГОТОВ И ЗАПАЗЕН!")
print("Той вече е обучен на данни до днешния ден и е готов за интеграция в уебсайта.")
print("==================================================")