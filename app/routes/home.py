from __future__ import annotations

from flask import (
    Blueprint,
    make_response,
    redirect,
    render_template,
    request,
)

from app.config import Config

home_bp = Blueprint("home", __name__)


# ============================================================
# HELPERS
# ============================================================

def _safe_return_url(value: str) -> str:
    value = (value or "").strip()

    if not value:
        return Config.RETURN_URL

    allowed = [
        "https://jamaicantruestories.com",
        "https://www.jamaicantruestories.com",
        "https://interpreter.jamaicantruestories.com",
    ]

    if value.startswith("/"):
        return value

    for domain in allowed:
        if value.startswith(domain):
            return value

    return Config.RETURN_URL


def _build_html_response(html: str, status: int = 200):
    resp = make_response(html, status)

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"

    # --------------------------------------------------------
    # SECURITY
    # --------------------------------------------------------

    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    resp.headers["X-Content-Type-Options"] = "nosniff"

    # --------------------------------------------------------
    # SHOPIFY EMBED SUPPORT
    # --------------------------------------------------------

    resp.headers["X-Frame-Options"] = "ALLOWALL"

    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors "
        "'self' "
        "https://jamaicantruestories.com "
        "https://www.jamaicantruestories.com "
        "https://admin.shopify.com "
        "https://*.myshopify.com;"
    )

    return resp


def _render_template_safe(template_name: str):
    try:
        html = render_template(
            template_name,
            APP_NAME="Dream Interpreter",
            RETURN_URL=Config.RETURN_URL,
            APP_DOMAIN="https://interpreter.jamaicantruestories.com",
            REQUEST_HOST=request.host,
            REQUEST_ORIGIN=request.host_url.rstrip("/"),
        )

        return _build_html_response(html)

    except Exception:
        return redirect(Config.RETURN_URL, code=302)


# ============================================================
# HOME
# ============================================================

@home_bp.route("/", methods=["GET"])
def home():
    return _render_template_safe(Config.TEMPLATE_INDEX)


# ============================================================
# UPGRADE
# ============================================================

@home_bp.route("/upgrade", methods=["GET"])
def upgrade():
    return _render_template_safe(Config.TEMPLATE_UPGRADE)


# ============================================================
# INTERPRETER ALIAS
# ============================================================

@home_bp.route("/dream-interpreter", methods=["GET"])
def interpreter_alias():
    return _render_template_safe(Config.TEMPLATE_INDEX)


# ============================================================
# UPGRADE ALIAS
# ============================================================

@home_bp.route("/unlock", methods=["GET"])
def unlock_alias():
    return _render_template_safe(Config.TEMPLATE_UPGRADE)


# ============================================================
# HEALTHY REDIRECT
# ============================================================

@home_bp.route("/go-home", methods=["GET"])
def go_home():
    target = _safe_return_url(
        request.args.get("return")
        or Config.RETURN_URL
    )

    return redirect(target, code=302)
