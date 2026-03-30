import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
import subprocess
from .models import db, PredictionLog

# Пътища
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bitcoin_lstm.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.gz')

# Глобално зареждане на модела (Singleton pattern)
model = None
scaler = None


def load_ai_models():
    global model, scaler
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ AI Моделите са заредени в Services слоя!")
    except Exception as e:
        print(f"⚠️ Грешка при зареждане на моделите: {e}")


# Зареждаме ги при стартиране на файла
load_ai_models()


def generate_live_prediction(user_id):
    """Слой Бизнес Логика: Извлича данни, прави инференция и записва в базата"""
    if model is None or scaler is None:
        raise Exception("AI моделът не е инициализиран.")

    # 1. Извличане на данни
    df = yf.download('BTC-USD', period='3mo', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Feature Engineering
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df.dropna(inplace=True)

    last_60 = df.iloc[-60:][['Close', 'SMA_7', 'Volatility']].values
    current_price = float(df['Close'].iloc[-1])

    # 3. AI Предсказание
    last_60_scaled = scaler.transform(last_60)
    X_input = last_60_scaled.reshape(1, 60, 3)
    pred_scaled = model.predict(X_input, verbose=0)

    # 4. Обратно скалиране
    dummy = np.zeros((1, 3))
    dummy[0, 0] = pred_scaled[0, 0]
    predicted_price = float(scaler.inverse_transform(dummy)[0, 0])

    # 5. Логика за аномалия
    diff_pct = abs(predicted_price - current_price) / current_price
    is_anomaly = bool(diff_pct > 0.05)

    # 6. Запис в Data Access слоя (Базата данни)
    new_log = PredictionLog(
        user_id=user_id,
        current_price=current_price,
        predicted_price=predicted_price,
        is_anomaly=is_anomaly
    )
    db.session.add(new_log)
    db.session.commit()

    # Връщаме готовия резултат към слоя за представяне
    return {
        "dates": df.index[-30:].strftime('%Y-%m-%d').tolist() + ["Tomorrow"],
        "history_prices": df['Close'].iloc[-30:].tolist(),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "is_anomaly": is_anomaly
    }


def retrain_model_background():
    """Слой Бизнес Логика: Стартира скрипта за обучение"""
    script_path = os.path.join(BASE_DIR, 'ml_engine', 'train_model.py')
    subprocess.run(['python', script_path], check=True)
    # След пре-обучение презареждаме новия модел в паметта
    load_ai_models()