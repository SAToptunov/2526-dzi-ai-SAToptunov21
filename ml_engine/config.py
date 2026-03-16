import os

# Основни настройки
SYMBOL = 'BTC-USD'       # Криптовалута
START_DATE = '2020-01-01'# Начало на данните
LOOK_BACK = 60           # Времеви прозорец (колко дни назад гледаме)
PREDICTION_DAYS = 1      # Прогноза за утре
TEST_SIZE = 0.2          # 20% от данните са за тест

# Пътища към файловете
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Създаване на папките, ако ги няма
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)