import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import tensorflow as tf
import datetime

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=90)


# Изключване на излишни TensorFlow предупреждения в конзолата
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- КОНФИГУРАЦИЯ ---
SYMBOL = 'BTC-USD'
LOOK_BACK = 60
MODELS_DIR = '../models'

print("==================================================")
print(f"🤖 СТАРТИРАНЕ НА AI ПРОГНОЗА ЗА {SYMBOL} 🤖")
print("==================================================")

# 1. Зареждане на запазените модел и скалер
try:
    print("⏳ Зареждане на модела и скалера...")
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'bitcoin_lstm.keras'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.gz'))
    print("✅ Успешно заредени!")
except Exception as e:
    print(f"❌ Грешка при зареждане: {e}")
    exit()

# 2. Сваляне на данни в реално време
print("\n📡 Сваляне на последните пазарни данни...")
# Теглим последните 3 месеца (за да сме сигурни, че имаме поне 60 дни + 7 дни за SMA)
df = yf.download(SYMBOL, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Взимаме само дните до днес
df = df[['Close']].copy()

# 3. Feature Engineering (Абсолютно същият като при обучението!)
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['Volatility'] = df['Close'].rolling(window=7).std()

# Изтриваме NaN стойностите, създадени от rolling функциите
df.dropna(inplace=True)

# 4. Взимане на последния прозорец (последните 60 дни)
last_60_days_df = df.iloc[-LOOK_BACK:]

if len(last_60_days_df) < LOOK_BACK:
    print(f"❌ Няма достатъчно данни. Нужни са {LOOK_BACK}, налични са {len(last_60_days_df)}.")
    exit()

# Взимаме текущата цена (цената на последното затваряне)
current_price = last_60_days_df['Close'].iloc[-1]
current_date = last_60_days_df.index[-1].strftime('%Y-%m-%d')

# 5. Скалиране
features = ['Close', 'SMA_7', 'Volatility']
last_60_days_values = last_60_days_df[features].values

# Използваме същия скалер от обучението!
last_60_days_scaled = scaler.transform(last_60_days_values)

# Преоформяне за LSTM -> (Samples, TimeSteps, Features) -> (1, 60, 3)
X_input = last_60_days_scaled.reshape(1, LOOK_BACK, 3)

# 6. Прогнозиране
print("🧠 Генериране на прогноза за следващия ден...")
pred_scaled = model.predict(X_input, verbose=0)

# 7. Обратно скалиране
# Правим dummy масив (1 ред, 3 колони), защото скалерът очаква 3 характеристики
dummy = np.zeros((1, 3))
dummy[0, 0] = pred_scaled[0, 0] # Слагаме прогнозата в колоната за 'Close'
predicted_price = scaler.inverse_transform(dummy)[0, 0]

# 8. Изчисляване на очакваната промяна
price_diff = predicted_price - current_price
percent_diff = (price_diff / current_price) * 100

# 9. Принтиране на финалния резултат
print("\n==================================================")
print(f"📊 РЕЗУЛТАТИ ОТ АНАЛИЗА ({current_date}):")
print(f"💵 Текуща цена:       ${current_price:,.2f}")
print(f"🔮 Прогноза за УТРЕ:  ${predicted_price:,.2f}")

if price_diff > 0:
    print(f"📈 Очакван тренд:     ВЪЗХОДЯЩ (+{percent_diff:.2f}%)")
else:
    print(f"📉 Очакван тренд:     НИЗХОДЯЩ ({percent_diff:.2f}%)")

# Проверка за Аномалия (ако очакваме промяна над 5%)
if abs(percent_diff) > 5.0:
    print(f"⚠️ АЛАРМА: Очаква се висока волатилност / Аномалия!")
else:
    print(f"✅ Пазарът се очаква да бъде стабилен.")
print("==================================================")