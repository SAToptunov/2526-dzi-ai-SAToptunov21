import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
import subprocess
from .extensions import db
from .models import PredictionLog
from datetime import datetime

# Пътища до ML моделите (извън web_app папката)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bitcoin_lstm_live.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.gz')

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


load_ai_models()


def generate_live_prediction(user_id):
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

    # ПОПРАВКА 1: Махаме .values, за да запазим DataFrame формата с имената на колоните
    last_window_df = df.iloc[-35:][['Close', 'SMA_7', 'Volatility']]
    current_price = float(df['Close'].iloc[-1])

    # 3. AI Предсказание (Подаваме DataFrame на скалера)
    last_window_scaled = scaler.transform(last_window_df)
    X_input = last_window_scaled.reshape(1, 35, 3)
    pred_scaled = model.predict(X_input, verbose=0)

    # 4. Обратно скалиране
    dummy = np.zeros((1, 3))
    dummy[0, 0] = pred_scaled[0, 0]

    # ПОПРАВКА 2: Превръщаме dummy масива също в DataFrame с точните имена
    dummy_df = pd.DataFrame(dummy, columns=['Close', 'SMA_7', 'Volatility'])
    predicted_price = float(scaler.inverse_transform(dummy_df)[0, 0])

    if current_price != 0:
        diff_pct = abs(predicted_price - current_price) / current_price
    else:
        diff_pct = 0

    is_anomaly = bool(diff_pct > 0.05)  # True ако разликата е над 5%

    # ---------------------------------------------------------
    # 9. ЗАПИС В БАЗАТА ДАННИ (С проверка за днес)
    # ---------------------------------------------------------
    from datetime import datetime  # Гарантираме, че е импортирано

    today = datetime.utcnow().date()
    start_of_day = datetime(today.year, today.month, today.day)

    # Търсим дали потребителят има прогноза от днес
    existing_log = PredictionLog.query.filter(
        PredictionLog.user_id == user_id,
        PredictionLog.timestamp >= start_of_day
    ).first()

    if existing_log:
        # Презаписваме старата прогноза от днес с най-новата
        existing_log.current_price = current_price
        existing_log.predicted_price = predicted_price
        existing_log.is_anomaly = is_anomaly
        existing_log.timestamp = datetime.utcnow()
    else:
        # Създаваме нов запис, ако му е за пръв път днес
        new_log = PredictionLog(
            user_id=user_id,
            symbol='BTC-USD',
            current_price=current_price,
            predicted_price=predicted_price,
            is_anomaly=is_anomaly
        )
        db.session.add(new_log)

    db.session.commit()

    # 10. Връщане на резултата към уебсайта
    return {
        "dates": df.index[-30:].strftime('%Y-%m-%d').tolist() + ["Tomorrow"],
        "history_prices": df['Close'].iloc[-30:].tolist(),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "is_anomaly": is_anomaly
    }


def retrain_model_background():
    """Слой Бизнес Логика: Стартира скрипта за обучение"""

    # 1. ОПРАВЕН ПЪТ: Сочи директно към train_production_live.py в ml_engine
    script_path = os.path.join(BASE_DIR, 'ml_engine', 'train_production_live.py')

    print(f"⚙️ СТАРТИРАНЕ НА ПРЕ-ОБУЧЕНИЕ: {script_path}")

    try:
        # 2. ПРО ТРИК: Използваме sys.executable, за да сме 100% сигурни,
        # че стартираме скрипта с Python-а от твоята .venv среда, а не глобалния!
        subprocess.run([sys.executable, script_path], check=True)

        print("✅ ПРЕ-ОБУЧЕНИЕТО ЗАВЪРШИ УСПЕШНО!")

        # След пре-обучение презареждаме новия модел в паметта на сървъра
        load_ai_models()

    except subprocess.CalledProcessError as e:
        print(f"❌ Грешка по време на изпълнение на скрипта за обучение: {e}")
    except Exception as e:
        print(f"❌ Неочаквана грешка: {e}")