from flask import Blueprint

# Задаваме префикс '/api', така че всички пътища тук автоматично ще започват с /api
api_bp = Blueprint('api', __name__, url_prefix='/api')

from . import routes