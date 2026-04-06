from flask import Blueprint

# Създаваме Blueprint с име 'auth'
auth_bp = Blueprint('auth', __name__)

from . import routes