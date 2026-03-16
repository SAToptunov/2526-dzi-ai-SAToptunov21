import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройки
sns.set(style="whitegrid")
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("📥 Сваляне на данни за EDA графиките...")
df = yf.download(SYMBOL, start=START_DATE, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Изчисляване на волатилност (Дневна възвръщаемост)
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()

print("📈 Генериране на Фигура 1 (Хистограма)...")
plt.figure(figsize=(10, 6))
sns.histplot(df['Close'].dropna(), kde=True, bins=50, color='blue')
plt.title('Разпределение на цената на Bitcoin (2020-2026)')
plt.xlabel('Цена (USD)')
plt.ylabel('Честота (брой дни)')
plt.savefig(os.path.join(PLOTS_DIR, 'eda_1_histogram.png'))
plt.close()

print("📈 Генериране на Фигура 2 (Волатилност)...")
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Daily_Return'], color='purple', alpha=0.7, linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title('Дневна възвръщаемост на Bitcoin (Визуализация на екстремна волатилност)')
plt.xlabel('Дата')
plt.ylabel('Процентна промяна')
plt.savefig(os.path.join(PLOTS_DIR, 'eda_2_volatility.png'))
plt.close()

print("📈 Генериране на Фигура 3 (Heatmap корелации)...")
plt.figure(figsize=(8, 6))
# Правим малка корелационна матрица за най-важните неща
corr_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
plt.title('Корелационна матрица на основните атрибути')
plt.savefig(os.path.join(PLOTS_DIR, 'eda_3_heatmap.png'))
plt.close()

print(f"✅ Готово! Твоите 3 графики те чакат в папка: {os.path.abspath(PLOTS_DIR)}")