from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from .config import Config
import os

# 1. Глобални инстанции на разширенията (без да са вързани за приложение още!)
db = SQLAlchemy()
login_manager = LoginManager()


def create_app(config_class=Config):
    """APPLICATION FACTORY: Създава и конфигурира Flask приложението"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Създаваме папка instance, ако я няма
    os.makedirs(os.path.join(app.root_path, 'instance'), exist_ok=True)

    # 2. Инициализиране на разширенията С ТОВА приложение
    db.init_app(app)

    login_manager.login_view = 'main.login'
    login_manager.login_message = "Моля, влезте в профила си."
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    # Зареждане на потребител за Flask-Login
    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # 3. Регистриране на Blueprints (Маршрутите)
    from .routes import main_bp
    app.register_blueprint(main_bp)

    # 4. Създаване на базите данни (ако не съществуват)
    with app.app_context():
        db.create_all()

    return app