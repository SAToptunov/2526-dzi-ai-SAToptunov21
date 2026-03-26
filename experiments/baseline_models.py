import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Настройки за визуализация
plt.style.use('ggplot')
MODELS_DIR = '../models'
PLOTS_DIR = '../plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. ЗАРЕЖДАНЕ НА ДАННИ
# ==========================================
print("📥 Зареждане на данни...")
X_train_3d = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')

# Зареждаме скалера
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.gz'))

# Преобразуване от 3D (LSTM) към 2D (ML)
nsamples, nx, ny = X_train_3d.shape
X_train_2d = X_train_3d.reshape((nsamples, nx * ny))

# Разделяне за валидация (последните 20%)
split_idx = int(len(X_train_2d) * 0.8)
X_tr, X_val = X_train_2d[:split_idx], X_train_2d[split_idx:]
y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

print(f"Train size: {len(y_tr)}, Test size: {len(y_val)}")

# ==========================================
# 2. ОБУЧЕНИЕ НА БАЗОВИ МОДЕЛИ
# ==========================================

# --- A. NAIVE BENCHMARK ---
print("\n--- 1. NAIVE BENCHMARK ---")
# Взимаме Close цената от предишния ден (последната в прозореца)
y_pred_naive_scaled = X_val[:, -3]

# --- B. LINEAR REGRESSION ---
print("--- 2. LINEAR REGRESSION ---")
lr_model = LinearRegression()
lr_model.fit(X_tr, y_tr)
y_pred_lr_scaled = lr_model.predict(X_val)

# --- C. ARIMA (One-Step-Ahead) ---
print("--- 3. ARIMA (5,1,0) ---")
# ARIMA изисква 1D серия.
try:
    # 1. Обучаваме само върху Train частта
    arima_model = ARIMA(y_tr, order=(5, 1, 0))
    arima_fit = arima_model.fit()

    # 2. Прилагаме модела върху новите данни (Test set), за да получим one-step predictions
    # Това обновява историята, без да преизчислява параметрите (бързо и коректно)
    arima_test_results = arima_fit.apply(y_val)

    # 3. Взимаме прогнозите (fittedvalues) за новия период
    y_pred_arima_scaled = arima_test_results.fittedvalues

except Exception as e:
    print(f"⚠️ Грешка при ARIMA: {e}")
    # Fallback към Naive, ако ARIMA гръмне
    y_pred_arima_scaled = y_pred_naive_scaled

# --- D. RANDOM FOREST ---
print("--- 4. RANDOM FOREST ---")
# Намаляваме малко дълбочината, за да избегнем overfitting
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_tr, y_tr)
y_pred_rf_scaled = rf_model.predict(X_val)


# ==========================================
# 3. ОБРАТНО СКАЛИРАНЕ (ВРЪЩАНЕ В $)
# ==========================================
def inverse_transform(y_scaled):
    # Dummy масив за скалера (очаква 3 колони)
    dummy = np.zeros((len(y_scaled), 3))
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]


y_real = inverse_transform(y_val)
y_naive = inverse_transform(y_pred_naive_scaled)
y_lr = inverse_transform(y_pred_lr_scaled)
y_rf = inverse_transform(y_pred_rf_scaled)
y_arima = inverse_transform(y_pred_arima_scaled)

# ==========================================
# 4. МЕТРИКИ И ТАБЛИЦА
# ==========================================
models_data = {
    'Naive Benchmark': y_naive,
    'Linear Regression': y_lr,
    'ARIMA': y_arima,
    'Random Forest': y_rf
}

results = []
for name, preds in models_data.items():
    mae = mean_absolute_error(y_real, preds)
    rmse = np.sqrt(mean_squared_error(y_real, preds))
    r2 = r2_score(y_real, preds)
    results.append({'Model': name, 'MAE ($)': mae, 'RMSE ($)': rmse, 'R2': r2})

results_df = pd.DataFrame(results)
print("\n=== 🏆 СРАВНИТЕЛНА ТАБЛИЦА (В ДОЛАРИ) ===")
print(results_df)

# ==========================================
# 5. ВИЗУАЛИЗАЦИИ
# ==========================================

# ГРАФИКА 1: Zoom-in (Последните 100 дни)
plt.figure(figsize=(14, 7))
subset = 100
plt.plot(y_real[:subset], label='Real Price', color='black', linewidth=2.5, alpha=0.8)
plt.plot(y_naive[:subset], label='Naive', linestyle=':', color='gray')
plt.plot(y_lr[:subset], label='Linear Reg', color='blue', linestyle='--', linewidth=1.5)
plt.plot(y_arima[:subset], label='ARIMA', color='purple', linestyle='-.', linewidth=1.5)
plt.plot(y_rf[:subset], label='Random Forest', color='green', alpha=0.5)
plt.title('Baseline Models Comparison (One-Step-Ahead Forecast)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, 'baseline_comparison.png'))
print("✅ Графика 1 запазена: baseline_comparison.png")

# ГРАФИКА 2: MAE Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(results_df['Model'], results_df['MAE ($)'], color=['gray', 'blue', 'purple', 'green'])
plt.title('Average Error (MAE) by Model')
plt.ylabel('Error in USD ($)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f"${int(yval)}", ha='center', va='bottom', fontweight='bold')
plt.savefig(os.path.join(PLOTS_DIR, 'baseline_mae_bar.png'))
print("✅ Графика 2 запазена: baseline_mae_bar.png")

# ГРАФИКА 3: Scatter Plot (Linearity)
plt.figure(figsize=(10, 10))
plt.scatter(y_real, y_lr, alpha=0.3, label='Linear Reg', color='blue')
plt.scatter(y_real, y_arima, alpha=0.3, label='ARIMA', color='purple', marker='x')
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'k--', lw=2, label='Perfect Prediction')
plt.title('Scatter Plot: Predictions vs Reality')
plt.xlabel('Real Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'baseline_scatter.png'))
print("✅ Графика 3 запазена: baseline_scatter.png")

joblib.dump(lr_model, os.path.join(MODELS_DIR, 'baseline_lr.pkl'))
print("✅ Linear Regression запазен.")

plt.show()