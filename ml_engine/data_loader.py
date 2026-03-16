import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from .config import DATA_DIR, MODELS_DIR


class CryptoDataLoader:
    def __init__(self, symbol, start_date):
        self.symbol = symbol
        self.start_date = start_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self):
        """Сваля данни от Yahoo Finance и ги пази локално"""
        file_path = os.path.join(DATA_DIR, f"{self.symbol}.csv")

        print(f"⌛ Сваляне на данни за {self.symbol}...")
        df = yf.download(self.symbol, start=self.start_date)

        # Почистване на MultiIndex (проблем на новия yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Запазване само на колоната Close
        df = df[['Close']].dropna()

        # Кеширане в CSV
        df.to_csv(file_path)
        print(f"✅ Данните са записани в {file_path}")
        return df

    def prepare_lstm_data(self, data, look_back):
        """
        Преобразува данните във формат за LSTM:
        [Samples, TimeSteps, Features]
        """
        # 1. Скалиране на данните (0 до 1)
        scaled_data = self.scaler.fit_transform(data.values)

        # ВАЖНО: Запазваме скалера, за да го ползваме в уеб приложението
        scaler_path = os.path.join(MODELS_DIR, 'scaler.gz')
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Scaler-ът е запазен в {scaler_path}")

        # 2. Създаване на прозорци (Sliding Window)
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # 3. Reshape за LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y, self.scaler