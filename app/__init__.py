from flask import Flask
from flask_cors import CORS

from app.config import Config


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
    )

    app.config.from_object(Config)

    app.secret_key = Config.SECRET_KEY
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["SESSION_COOKIE_SAMESITE"] = Config.SESSION_COOKIE_SAMESITE
    app.config["SESSION_COOKIE_SECURE"] = Config.SESSION_COOKIE_SECURE
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    Config.validate()

    CORS(
        app,
        resources={r"/*": {"origins": app.config["ALLOWED_ORIGINS"]}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-Admin-Key"],
        methods=["GET", "POST", "OPTIONS"],
    )

    from app.routes.home import home_bp
    from app.routes.health import health_bp
    from app.routes.interpreter import interpreter_bp
    from app.routes.billing import billing_bp
    from app.routes.journal import journal_bp
    from app.routes.admin import admin_bp
    from app.routes.tracking import tracking_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(interpreter_bp)
    app.register_blueprint(billing_bp)
    app.register_blueprint(journal_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(tracking_bp)

    return app
