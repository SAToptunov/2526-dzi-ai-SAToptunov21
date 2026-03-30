import os
from flask import Flask
from flask_login import LoginManager
from models import db, User


# Този файл е чисто APPLICATION FACTORY
def create_app():
    """Фабрика за създаване на Flask приложението"""
    app = Flask(__name__)

    # 1. Конфигурация
    basedir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'cads.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'super-secret-cads-key-2026'

    # 2. Инициализация на разширения (Data Access Layer setup)
    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'main.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # 3. Регистриране на маршрутите (Presentation Layer setup)
    from routes import main_bp
    app.register_blueprint(main_bp)

    # 4. Създаване на базите данни
    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    # Стартиране на приложението от фабриката
    app = create_app()
    print("🌐 Сървърът стартира на http://127.0.0.1:5000")
    app.run(debug=True, port=5000)