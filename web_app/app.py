import os
from flask import Flask
from flask_login import LoginManager
from models import db, User
from routes import main_bp


def create_app():
    app = Flask(__name__)

    # Системни настройки
    basedir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'cads.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'super-secret-cads-key-2026'  # Задължително за логин сесиите

    db.init_app(app)

    # Настройка на Login Manager
    login_manager = LoginManager()
    login_manager.login_view = 'main.login'  # Накъде да пренасочи, ако не си логнат
    login_manager.login_message = "Моля, влезте в профила си, за да достъпите тази страница."
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()  # Създава базите данни

    return app


if __name__ == '__main__':
    app = create_app()
    print("🌐 Уеб сървърът стартира на http://127.0.0.1:5000")
    app.run(debug=True, port=5000)