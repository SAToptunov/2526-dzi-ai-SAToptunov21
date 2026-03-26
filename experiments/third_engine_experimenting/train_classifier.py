import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
import random
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- ЗАKОВАВАНЕ НА СЛУЧАЙНОСТТА ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- КОНФИГУРАЦИЯ ---
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
LOOK_BACK = 30 # По-къс прозорец за по-бързи реакции
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. ПОДГОТОВКА (CLASSIFICATION STRATEGY)
# ==========================================
print("📥 1. Сваляне и обработка за Класификация...")

df = yf.download(SYMBOL, start=START_DATE)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Feature Engineering
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['Volatility'] = df['Log_Ret'].rolling(window=14).std()
df['Log_Vol'] = np.log(df['Volume'].replace(0, 1))

# RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'])

df.dropna(inplace=True)

# --- TARGET DEFINITION (КЛЮЧОВ МОМЕНТ) ---
# 1 = Цената се качва утре
# 0 = Цената пада утре
df['Target_Class'] = np.where(df['Log_Ret'].shift(-1) > 0, 1, 0)

# Премахваме последния ред (нямаме таргет за утре)
df.dropna(inplace=True)

# Проверка на баланса
print(f"Баланс на класовете: {df['Target_Class'].value_counts(normalize=True)}")

# Разделяне
train_size = int(len(df) * 0.85)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Скалиране (Само Features)
feature_cols = ['Log_Ret', 'Volatility', 'RSI', 'Log_Vol']
target_col = 'Target_Class'

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
X_test_scaled = scaler.transform(test_df[feature_cols])

y_train = train_df[target_col].values
y_test = test_df[target_col].values

joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_classifier.gz'))

# Sequencing
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_scaled, y_train, LOOK_BACK)
X_test, y_test = create_sequences(X_test_scaled, y_test, LOOK_BACK)

# Validation Split
val_split = int(len(X_train) * 0.9)
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

print(f"Shapes -> Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 2. МОДЕЛ (Attention-Based Classifier)
# ==========================================
print("🧠 2. Обучение на Класификатор...")

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM
lstm_out = LSTM(64, return_sequences=True)(inputs)
lstm_out = Dropout(0.4)(lstm_out) # По-висок Dropout за генерализация

# Attention
query_value_attention_seq = Attention()([lstm_out, lstm_out])
context_vector = GlobalAveragePooling1D()(query_value_attention_seq)

# Dense
x = Dense(32, activation='relu')(context_vector)

# --- ПОПРАВКАТА Е ТУК ---
x = Dropout(0.3)(x)  # <--- Добавихме (x) накрая!
# ------------------------

# Output Layer: SIGMOID (Вероятност от 0 до 1)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# Loss: Binary Crossentropy (за класификация)
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

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

model.save(os.path.join(MODELS_DIR, 'bitcoin_classifier.keras'))
print("✅ Класификаторът е запазен.")

# ==========================================
# 3. ОЦЕНКА
# ==========================================
print("📈 3. Оценка на точността...")

# Прогноза (Вероятности)
y_pred_prob = model.predict(X_test)

# Превръщане в клас (0 или 1) с праг 0.5
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Метрики
acc = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n=================================")
print(f"🏆 DIRECTIONAL ACCURACY: {acc*100:.2f}%")
print(f"=================================")
print("Confusion Matrix:")
print(conf_matrix)
print("\nReport:")
print(classification_report(y_test, y_pred))

# Визуализация на Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(PLOTS_DIR, 'classifier_confusion_matrix.png'))
print("✅ Графиката е запазена.")

# Графика на вероятностите
plt.figure(figsize=(14, 6))
plt.plot(y_pred_prob[-100:], label='AI Probability (Up Signal)', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Decision Threshold')
plt.title('AI Confidence Levels (Last 100 Days)')
plt.ylabel('Probability of Price UP')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'classifier_confidence.png'))
plt.show()