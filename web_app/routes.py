from flask import Blueprint, render_template, jsonify
from models import PredictionLog

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template('base.html')

@main_bp.route('/dashboard')
def dashboard():
    return "<h1>Dashboard - Chart Loading...</h1>"

@main_bp.route('/api/history')
def history_api():
    logs = PredictionLog.query.order_by(PredictionLog.date_predicted.desc()).limit(10).all()
    results = [{"date": log.date_predicted, "price": log.predicted_price} for log in logs]
    return jsonify(results)