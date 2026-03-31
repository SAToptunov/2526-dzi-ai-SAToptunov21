import os
from flask import Flask
from .config import Config
from .extensions import db, login_manager


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.makedirs(os.path.join(app.root_path, 'instance'), exist_ok=True)

    db.init_app(app)

    # ВАЖНО: Вече логин страницата се намира в auth модула!
    login_manager.login_view = 'auth.login'
    login_manager.login_message = "Моля, влезте в профила си."
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # РЕГИСТРИРАНЕ НА ТРИТЕ МОДУЛА (BLUEPRINTS)
    from .auth import auth_bp
    from .main import main_bp
    from .api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.create_all()

    return app