import threading
from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from .models import db, User, PredictionLog
import services  # Импортираме бизнес логиката!

main_bp = Blueprint('main', __name__)


# ================= AUTH ROUTES =================
@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Потребителското име вече е заето.', 'danger')
            return redirect(url_for('main.register'))

        new_user = User(username=username)
        new_user.set_password(password)
        if username.lower() == 'admin': new_user.role = 'admin'

        db.session.add(new_user)
        db.session.commit()
        flash('Успешна регистрация!', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html')


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Грешни данни.', 'danger')
    return render_template('login.html')


@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))


# ================= APP ROUTES (PRESENTATION LAYER) =================
@main_bp.route('/')
def index(): return render_template('index.html')


@main_bp.route('/about')
def about(): return render_template('about.html')


@main_bp.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')


@main_bp.route('/history')
@login_required
def history():
    logs = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.timestamp.desc()).all()
    return render_template('history.html', logs=logs)


# ================= API ROUTES (СВЪРЗВАНЕ С БИЗНЕС ЛОГИКАТА) =================
@main_bp.route('/api/predict', methods=['GET'])
@login_required
def get_prediction():
    try:
        # ПРЕДСТАВЯЩИЯТ СЛОЙ ПРОСТО ВИКА БИЗНЕС СЛОЯ
        result = services.generate_live_prediction(current_user.id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route('/admin')
@login_required
def admin_panel():
    if current_user.role != 'admin': return redirect(url_for('main.dashboard'))
    return render_template('admin.html', last_trained="Вижте логовете на сървъра")


@main_bp.route('/api/retrain', methods=['POST'])
@login_required
def api_retrain():
    if current_user.role != 'admin': return jsonify({"error": "Отказан достъп."}), 403
    # Делегираме на бизнес слоя да стартира обучението
    threading.Thread(target=services.retrain_model_background).start()
    return jsonify({"message": "Пре-обучението стартира на заден фон."})