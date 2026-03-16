from flask import Flask
from extensions import db
from routes import main_bp


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cads.db'
    app.config['SECRET_KEY'] = 'dev-key-123'

    db.init_app(app)

    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()  # Създава таблиците

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)