from flask import render_template, redirect, url_for
from flask_login import login_required, current_user
from . import main_bp
from ..models import PredictionLog

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@main_bp.route('/history')
@login_required
def history():
    logs = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.timestamp.desc()).all()
    return render_template('history.html', logs=logs)

@main_bp.route('/admin')
@login_required
def admin_panel():
    if current_user.role != 'admin':
        return redirect(url_for('main.dashboard'))
    return render_template('admin.html', last_trained="Проверете логовете")