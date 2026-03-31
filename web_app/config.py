import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'super-seKret-cads-key-2026'
    # Базата данни ще се създаде в web_app/instance/cads.db
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'instance', 'cads.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False