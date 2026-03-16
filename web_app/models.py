from datetime import datetime
from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash

# 1. User Model (Актьори)
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False) # Sensitive Data!
    role = db.Column(db.String(20), default='trader') # 'admin' or 'trader'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# 2. Prediction Log Model (История)
class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'
    id = db.Column(db.Integer, primary_key=True)
    date_predicted = db.Column(db.DateTime, default=datetime.utcnow)
    symbol = db.Column(db.String(10), nullable=False) # e.g. BTC-USD
    predicted_price = db.Column(db.Float, nullable=False)
    actual_price = db.Column(db.Float, nullable=True) # Попълва се по-късно
    is_anomaly = db.Column(db.Boolean, default=False)