from __future__ import annotations

import uuid
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # ============================================================
    # Core Config
    # ============================================================
    app.config.from_object(Config)

    app.secret_key = Config.SECRET_KEY

    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    # Keep JSON responses predictable for frontend clients.
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    # Template reload should only be active in local/debug mode.
    app.config["TEMPLATES_AUTO_RELOAD"] = bool(
        getattr(Config, "DEBUG", False)
    )

    # ============================================================
    # Session / Cookie Security
    # ============================================================
    app.config["SESSION_COOKIE_NAME"] = "jts_session"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SECURE"] = Config.SESSION_COOKIE_SECURE
    app.config["SESSION_COOKIE_SAMESITE"] = Config.SESSION_COOKIE_SAMESITE
    app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30

    # ============================================================
    # Reverse Proxy / Render Support
    # ============================================================
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=1,
        x_proto=1,
        x_host=1,
        x_port=1,
    )

    # ============================================================
    # Trusted Hosts
    # ============================================================
    trusted_hosts = [
        "jamaicantruestories.com",
        "www.jamaicantruestories.com",
        "interpreter.jamaicantruestories.com",
        "plqwhd-jm.myshopify.com",
        "localhost",
        "127.0.0.1",
    ]

    app.config["TRUSTED_HOSTS"] = trusted_hosts

    # ============================================================
    # Validate Config
    # ============================================================
    Config.validate()

    # ============================================================
    # CORS
    # ============================================================
    allowed_origins = app.config.get("ALLOWED_ORIGINS", [])

    CORS(
        app,
        resources={
            r"/*": {
                "origins": allowed_origins,
            }
        },
        supports_credentials=True,
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Admin-Key",
            "X-Requested-With",
        ],
        expose_headers=[
            "Content-Type",
            "X-Request-ID",
        ],
        methods=[
            "GET",
            "POST",
            "OPTIONS",
        ],
        max_age=86400,
    )

    # ============================================================
    # Request Lifecycle Helpers
    # ============================================================
    @app.before_request
    def attach_request_id() -> None:
        request.request_id = request.headers.get("X-Request-ID") or str(
            uuid.uuid4()
        )

    @app.after_request
    def apply_response_headers(response):
        request_id = getattr(request, "request_id", "")

        if request_id:
            response.headers["X-Request-ID"] = request_id

        # Security headers.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        # Helpful browser caching behavior for app HTML/API responses.
        if request.path.startswith("/interpret") or request.path.startswith("/check-access"):
            response.headers["Cache-Control"] = "no-store"

        return response

    # ============================================================
    # Error Response Helper
    # ============================================================
    def error_response(
        *,
        status_code: int,
        error: str,
        message: str,
        details: Any | None = None,
    ):
        payload: dict[str, Any] = {
            "error": error,
            "message": message,
            "request_id": getattr(request, "request_id", None),
        }

        if details is not None:
            payload["details"] = details

        return jsonify(payload), status_code

    # ============================================================
    # Error Handlers
    # ============================================================
    @app.errorhandler(400)
    def bad_request(error):
        return error_response(
            status_code=400,
            error="bad_request",
            message="The request could not be processed.",
        )

    @app.errorhandler(401)
    def unauthorized(error):
        return error_response(
            status_code=401,
            error="unauthorized",
            message="Authentication is required.",
        )

    @app.errorhandler(402)
    def payment_required(error):
        return error_response(
            status_code=402,
            error="payment_required",
            message="Payment or active access is required.",
        )

    @app.errorhandler(403)
    def forbidden(error):
        return error_response(
            status_code=403,
            error="forbidden",
            message="You do not have permission to access this resource.",
        )

    @app.errorhandler(404)
    def not_found(error):
        return error_response(
            status_code=404,
            error="not_found",
            message="The requested endpoint was not found.",
        )

    @app.errorhandler(405)
    def method_not_allowed(error):
        return error_response(
            status_code=405,
            error="method_not_allowed",
            message="This method is not allowed for the requested endpoint.",
        )

    @app.errorhandler(413)
    def request_too_large(error):
        return error_response(
            status_code=413,
            error="request_too_large",
            message="The dream entry is too large.",
        )

    @app.errorhandler(429)
    def rate_limited(error):
        return error_response(
            status_code=429,
            error="rate_limited",
            message="Too many requests. Please wait and try again.",
        )

    @app.errorhandler(500)
    def internal_error(error):
        return error_response(
            status_code=500,
            error="internal_server_error",
            message="Something went wrong on the server.",
        )

    # ============================================================
    # Blueprints
    # ============================================================
    from app.routes.admin import admin_bp
    from app.routes.billing import billing_bp
    from app.routes.health import health_bp
    from app.routes.home import home_bp
    from app.routes.interpreter import interpreter_bp
    from app.routes.journal import journal_bp
    from app.routes.tracking import tracking_bp

    blueprints = [
        home_bp,
        health_bp,
        interpreter_bp,
        billing_bp,
        journal_bp,
        admin_bp,
        tracking_bp,
    ]

    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    # ============================================================
    # Startup Log
    # ============================================================
    print("JTS Dream Interpreter initialized.", flush=True)
    print(f"Allowed origins: {allowed_origins}", flush=True)
    print(f"Trusted hosts: {trusted_hosts}", flush=True)
    print(
        "Registered blueprints: "
        + ", ".join(bp.name for bp in blueprints),
        flush=True,
    )

    return app
