from flask import Flask
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
    )

    # ============================================================
    # Core Config
    # ============================================================
    app.config.from_object(Config)

    app.secret_key = Config.SECRET_KEY

    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    # ============================================================
    # Session / Cookie Security
    # ============================================================
    app.config["SESSION_COOKIE_NAME"] = "jts_session"

    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SECURE"] = Config.SESSION_COOKIE_SECURE
    app.config["SESSION_COOKIE_SAMESITE"] = Config.SESSION_COOKIE_SAMESITE

    app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30

    # ============================================================
    # Flask Production Optimizations
    # ============================================================
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    app.config["TEMPLATES_AUTO_RELOAD"] = False

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
        ],
        methods=[
            "GET",
            "POST",
            "OPTIONS",
        ],
        max_age=86400,
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

    app.register_blueprint(home_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(interpreter_bp)
    app.register_blueprint(billing_bp)
    app.register_blueprint(journal_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(tracking_bp)

    # ============================================================
    # Security Headers
    # ============================================================
    @app.after_request
    def apply_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        return response

    # ============================================================
    # Health Startup Log
    # ============================================================
    print("JTS Dream Interpreter initialized.", flush=True)
    print(f"Allowed origins: {allowed_origins}", flush=True)

    return app
