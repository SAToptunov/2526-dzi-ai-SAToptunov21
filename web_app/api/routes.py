import threading
from flask import jsonify
from flask_login import login_required, current_user
from . import api_bp
from .. import services  # Взимаме бизнес логиката


@api_bp.route('/predict', methods=['GET'])
@login_required
def get_prediction():
    try:
        result = services.generate_live_prediction(current_user.id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/retrain', methods=['POST'])
@login_required
def api_retrain():
    if current_user.role != 'admin':
        return jsonify({"error": "Отказан достъп."}), 403

    threading.Thread(target=services.retrain_model_background).start()
    return jsonify({"message": "Пре-обучението стартира на заден фон."})