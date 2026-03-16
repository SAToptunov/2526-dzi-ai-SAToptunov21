import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression

# Импорт на конфигурацията и новия тракер
from ml_engine.config import SYMBOL, START_DATE, LOOK_BACK, MODELS_DIR, PLOTS_DIR
from ml_engine.data_loader import CryptoDataLoader
from ml_engine.tracker import ExperimentTracker


def main():
    print("🔎 Стартиране на подробен сравнителен анализ...")

    # Инициализиране на тракера
    tracker = ExperimentTracker()

    # 1. Зареждане на данни
    loader = CryptoDataLoader(SYMBOL, START_DATE)
    df = loader.fetch_data()
    X, y, scaler = loader.prepare_lstm_data(df, LOOK_BACK)

    # Тестови данни (последните 20%)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    test_dates = df.index[split_idx + LOOK_BACK:]

    # Реални цени (Unscaled)
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ---------------------------------------------------------
    # МОДЕЛ 1: LSTM
    # ---------------------------------------------------------
    print("🤖 Оценка на LSTM модела...")
    try:
        lstm_model = load_model(os.path.join(MODELS_DIR, 'bitcoin_lstm.keras'))
        lstm_preds_scaled = lstm_model.predict(X_test)
        lstm_preds = scaler.inverse_transform(lstm_preds_scaled)

        # ЗАПИСВАНЕ НА МЕТРИКИТЕ
        tracker.log_experiment(
            model_name="LSTM (Neural Net)",
            epochs=50,  # Тук ръчно пишеш колко са били епохите или ги взимаш от config
            y_true=y_real,
            y_pred=lstm_preds,
            notes="Тунингован модел с Huber Loss"
        )
    except Exception as e:
        print(f"Грешка при LSTM: {e}")
        lstm_preds = np.zeros_like(y_real)  # Fallback

    # ---------------------------------------------------------
    # МОДЕЛ 2: Linear Regression
    # ---------------------------------------------------------
    print("📉 Оценка на Linear Regression...")
    X_train_2d = X[:split_idx].reshape(X[:split_idx].shape[0], X[:split_idx].shape[1])
    y_train_2d = y[:split_idx]
    X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[1])

    lr_model = LinearRegression()
    lr_model.fit(X_train_2d, y_train_2d)
    lr_preds_scaled = lr_model.predict(X_test_2d)
    lr_preds = scaler.inverse_transform(lr_preds_scaled.reshape(-1, 1))

    # ЗАПИСВАНЕ НА МЕТРИКИТЕ ЗА РЕГРЕСИЯТА
    tracker.log_experiment(
        model_name="Linear Regression",
        epochs=0,
        y_true=y_real,
        y_pred=lr_preds,
        notes="Baseline модел"
    )

    # ---------------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ (Аномалии)
    # ---------------------------------------------------------
    # Използваме LSTM за аномалии
    errors = np.abs(y_real - lstm_preds)
    threshold = np.mean(errors) + 2.0 * np.std(errors)
    anomaly_indices = np.where(errors > threshold)[0]

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_real, color='black', label='Реална цена', linewidth=1.5)
    plt.plot(test_dates, lr_preds, color='orange', linestyle='--', label='Linear Regression', alpha=0.8)
    plt.plot(test_dates, lstm_preds, color='green', label='LSTM AI', linewidth=2)

    if len(anomaly_indices) > 0:
        plt.scatter(test_dates[anomaly_indices], y_real[anomaly_indices], color='red', s=50, zorder=5, label='Аномалии')

    plt.title(f'Сравнителен анализ: {SYMBOL}')
    plt.xlabel('Дата')
    plt.ylabel('Цена (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(PLOTS_DIR, 'detailed_comparison.png'))
    plt.show()


if __name__ == "__main__":
    main()