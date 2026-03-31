from flask import Blueprint, redirect, render_template

from app.config import Config

home_bp = Blueprint("home", __name__)


@home_bp.route("/", methods=["GET"])
def home():
    try:
        return render_template(Config.TEMPLATE_INDEX)
    except Exception:
        return redirect(Config.RETURN_URL, code=302)


@home_bp.route("/upgrade", methods=["GET"])
def upgrade():
    try:
        return render_template(Config.TEMPLATE_UPGRADE)
    except Exception:
        return redirect(Config.RETURN_URL, code=302)
