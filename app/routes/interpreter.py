from flask import Blueprint, make_response, request

from app.services.interpreter_service import run_interpretation

interpreter_bp = Blueprint("interpreter", __name__)


@interpreter_bp.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    """
    Main dream interpretation endpoint.

    POST:
        Accepts JSON payload with dream text and optional email.

    OPTIONS:
        Handles CORS preflight requests.
    """
    if request.method == "OPTIONS":
        resp = make_response("", 204)
        return resp

    return run_interpretation()
