import os
from flask import Flask
from .config import Config
from .extensions import db, login_manager


def create_app(config_class=Config):
    """APPLICATION FACTORY: Създава и конфигурира Flask приложението"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Създаваме папка instance, ако я няма
    os.makedirs(os.path.join(app.root_path, 'instance'), exist_ok=True)

    # Инициализиране на разширенията
    db.init_app(app)

    login_manager.login_view = 'main.login'
    login_manager.login_message = "Моля, влезте в профила си, за да достъпите тази страница."
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    # Зареждане на потребител за Flask-Login
    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Регистриране на Blueprints (Маршрутите)
    from .routes import main_bp
    app.register_blueprint(main_bp)

    # Създаване на таблиците в базата данни
    with app.app_context():
        db.create_all()

    return app