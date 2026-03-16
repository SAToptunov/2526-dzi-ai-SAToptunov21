import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ExperimentTracker:
    def __init__(self, log_file='experiments_log.csv'):
        # Файлът ще се намира в главната папка
        self.log_file = log_file
        self._initialize_log()

    def _initialize_log(self):
        """Създава файла с заглавията, ако не съществува"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Timestamp', 'Model_Type', 'Epochs',
                    'MAE (USD)', 'RMSE (USD)', 'MAPE (%)',
                    'R2 Score', 'Direction_Accuracy (%)', 'Notes'
                ])

    def calculate_metrics(self, y_true, y_pred):
        """Изчислява разширени метрики"""
        # 1. Основни регресионни метрики
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # 2. MAPE (Процентна грешка) - предпазваме се от делене на 0
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # 3. Directional Accuracy (Позната ли е посоката?)
        # Сравняваме знака на промяната: (y_t - y_{t-1})
        # Тъй като y_true и y_pred са масиви, правим следното:
        if len(y_true) > 1:
            true_diff = np.diff(y_true.flatten())
            pred_diff = np.diff(y_pred.flatten())
            # Ако знаците съвпадат, значи сме познали посоката
            correct_direction = np.sign(true_diff) == np.sign(pred_diff)
            direction_acc = np.mean(correct_direction) * 100
        else:
            direction_acc = 0.0

        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'R2': round(r2, 4),
            'Direction': round(direction_acc, 2)
        }

    def log_experiment(self, model_name, epochs, y_true, y_pred, notes=""):
        """Записва резултатите във CSV файла"""
        metrics = self.calculate_metrics(y_true, y_pred)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                model_name,
                epochs,
                metrics['MAE'],
                metrics['RMSE'],
                metrics['MAPE'],
                metrics['R2'],
                metrics['Direction'],
                notes
            ])

        print(f"\n📝 Резултатите са записани в {self.log_file}")
        print(f"   MAE: ${metrics['MAE']} | RMSE: ${metrics['RMSE']} | MAPE: {metrics['MAPE']}%")
        print(f"   R2: {metrics['R2']} | Посока: {metrics['Direction']}% Accuracy")