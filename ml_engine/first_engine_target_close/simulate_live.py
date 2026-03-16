import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Изключване на излишни TensorFlow логове
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- КОНФИГУРАЦИЯ ---
SYMBOL = 'BTC-USD'
LOOK_BACK = 14
SIMULATION_DAYS = 100  # Колко дни назад искаме да симулираме
MODELS_DIR = '../models'
PLOTS_DIR = '../plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("==================================================")
print(f"🚀 СТАРТИРАНЕ НА СИМУЛАЦИЯ (BACKTEST): Последните {SIMULATION_DAYS} дни")
print("==================================================")

# 1. Зареждане на модела и скалера
try:
    print("⏳ Зареждане на модел и скалер...")
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'bitcoin_lstm_live.keras'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.gz'))
    print("✅ Успешно заредени!")
except Exception as e:
    print(f"❌ Грешка: {e}")
    exit()

# 2. Сваляне на данни (взимаме 1 година назад, за да имаме достатъчно история)
print("\n📡 Сваляне на исторически данни...")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

df = yf.download(SYMBOL, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 3. Feature Engineering (За целия период предварително)
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['Volatility'] = df['Close'].rolling(window=7).std()
df['Momentum'] = df['Close'] - df['Close'].shift(3)
df.dropna(inplace=True)

# Проверка дали имаме достатъчно данни
if len(df) < LOOK_BACK + SIMULATION_DAYS:
    print(f"❌ Няма достатъчно данни за симулация. Нужни: {LOOK_BACK + SIMULATION_DAYS}, Налични: {len(df)}")
    exit()

# 4. СИМУЛАЦИОНЕН ЦИКЪЛ (Walk-Forward)
print(f"\n🔄 Започва симулация ден по ден...")

actual_prices = []
predicted_prices = []
dates = []

# Започваме от (Края - 100 дни) и вървим до днес
start_idx = len(df) - SIMULATION_DAYS
features = ['Close', 'SMA_7', 'Volatility', 'Momentum']  # 4 признака
num_features = len(features)  # Взимаме бройката автоматично (4)

for current_day_idx in range(start_idx, len(df)):

    window_df = df.iloc[current_day_idx - LOOK_BACK: current_day_idx]
    real_price_today = df['Close'].iloc[current_day_idx]
    current_date = df.index[current_day_idx]

    # Подготовка за модела
    window_values = window_df[features].values
    window_scaled = scaler.transform(window_values)

    # ПОПРАВКА: Използваме num_features вместо твърдо кодирано 3
    X_input = window_scaled.reshape(1, LOOK_BACK, num_features)

    # Прогноза
    pred_scaled = model.predict(X_input, verbose=0)

    # ПОПРАВКА: dummy масивът също трябва да е с размер num_features (4)
    dummy = np.zeros((1, num_features))
    dummy[0, 0] = pred_scaled[0, 0]
    predicted_price_today = scaler.inverse_transform(dummy)[0, 0]

    # Записваме резултатите
    dates.append(current_date)
    actual_prices.append(real_price_today)
    predicted_prices.append(predicted_price_today)

    if len(dates) % 10 == 0:
        print(
            f"[{current_date.strftime('%Y-%m-%d')}] Реална: ${real_price_today:,.0f} | Прогноза: ${predicted_price_today:,.0f}")

# 5. ИЗЧИСЛЯВАНЕ НА МЕТРИКИ
actual_prices = np.array(actual_prices)
predicted_prices = np.array(predicted_prices)

mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
accuracy = 100 - mape

print("\n==================================================")
print(f"📊 ФИНАЛНИ РЕЗУЛТАТИ ОТ СИМУЛАЦИЯТА ({SIMULATION_DAYS} дни)")
print(f"MAE (Средна грешка в долари): ${mae:,.2f}")
print(f"RMSE (Корен от средноквадратичната): ${rmse:,.2f}")
print(f"MAPE (Средна грешка в %): {mape:.2f}%")
print(f"Точност на модела (Accuracy): {accuracy:.2f}%")
print("==================================================")

# 6. ВИЗУАЛИЗАЦИЯ
plt.figure(figsize=(14, 7))
plt.plot(dates, actual_prices, label='Реална цена (Actual)', color='black', linewidth=2)
plt.plot(dates, predicted_prices, label='Симулирана Прогноза (AI)', color='green', linestyle='--', linewidth=2)

# Маркиране на аномалии (грешка > 5%)
anomalies_x = []
anomalies_y = []
for i in range(len(actual_prices)):
    diff_pct = abs((predicted_prices[i] - actual_prices[i]) / actual_prices[i]) * 100
    if diff_pct > 5.0:  # Праг за аномалия
        anomalies_x.append(dates[i])
        anomalies_y.append(actual_prices[i])

if anomalies_x:
    plt.scatter(anomalies_x, anomalies_y, color='red', s=50, label='Засечена Аномалия (>5% разлика)', zorder=5)

plt.title(f'Симулация на живо: Последните {SIMULATION_DAYS} дни\nТочност: {accuracy:.2f}% | MAE: ${mae:,.0f}')
plt.xlabel('Дата')
plt.ylabel('Цена (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = os.path.join(PLOTS_DIR, 'live_simulation_test.png')
plt.savefig(plot_path)
print(f"\n✅ Графиката е запазена в: {plot_path}")
plt.show()