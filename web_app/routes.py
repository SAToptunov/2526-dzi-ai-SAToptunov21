import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, PredictionLog

main_bp = Blueprint('main', __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bitcoin_lstm.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.gz')

try:
    print("⏳ Зареждане на AI модела...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ AI Моделът е зареден успешно!")
except:
    model, scaler = None, None


# ================= AUTH ROUTES =================

@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Потребителското име вече е заето.', 'danger')
            return redirect(url_for('main.register'))

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Регистрацията е успешна! Моля, влезте в профила си.', 'success')
        return redirect(url_for('main.login'))

    return render_template('register.html')


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash(f'Здравейте, {username}!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Грешно потребителско име или парола.', 'danger')

    return render_template('login.html')


@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Излязохте от профила си.', 'info')
    return redirect(url_for('main.index'))


# ================= APP ROUTES =================

@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.route('/dashboard')
@login_required  # ЗАЩИТЕНА СТРАНИЦА!
def dashboard():
    return render_template('dashboard.html')


@main_bp.route('/history')
@login_required
def history():
    # Взимаме само прогнозите на текущия потребител
    logs = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.timestamp.desc()).all()
    return render_template('history.html', logs=logs)


# ================= API ROUTE =================

@main_bp.route('/api/predict', methods=['GET'])
@login_required
def get_prediction():
    if model is None:
        return jsonify({"error": "AI моделът не е зареден."}), 500
    try:
        df = yf.download('BTC-USD', period='3mo', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['Volatility'] = df['Close'].rolling(window=7).std()
        df.dropna(inplace=True)

        last_60 = df.iloc[-60:][['Close', 'SMA_7', 'Volatility']].values
        current_price = float(df['Close'].iloc[-1])

        last_60_scaled = scaler.transform(last_60)
        X_input = last_60_scaled.reshape(1, 60, 3)

        pred_scaled = model.predict(X_input, verbose=0)
        dummy = np.zeros((1, 3))
        dummy[0, 0] = pred_scaled[0, 0]
        predicted_price = float(scaler.inverse_transform(dummy)[0, 0])

        diff_pct = abs(predicted_price - current_price) / current_price
        is_anomaly = bool(diff_pct > 0.05)

        # ЗАПИС С ИД НА ПОТРЕБИТЕЛЯ
        new_log = PredictionLog(
            user_id=current_user.id,
            current_price=current_price,
            predicted_price=predicted_price,
            is_anomaly=is_anomaly
        )
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "dates": df.index[-30:].strftime('%Y-%m-%d').tolist() + ["Tomorrow"],
            "history_prices": df['Close'].iloc[-30:].tolist(),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "is_anomaly": is_anomaly
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500