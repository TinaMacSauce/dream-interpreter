from flask import Blueprint, make_response, request

from app.services.interpreter_service import run_interpretation

interpreter_bp = Blueprint("interpreter", __name__)


@interpreter_bp.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return make_response("", 204)
    return run_interpretation()
