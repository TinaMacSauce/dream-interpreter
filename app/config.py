from __future__ import annotations

import os
from typing import List


# ============================================================
# Environment Helpers
# ============================================================

def _csv_to_list(value: str) -> List[str]:
    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None or str(raw).strip() == "":
        return default

    try:
        return int(str(raw).strip())
    except ValueError:
        raise RuntimeError(f"{name} must be an integer.")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)

    if raw is None or str(raw).strip() == "":
        return default

    try:
        return float(str(raw).strip())
    except ValueError:
        raise RuntimeError(f"{name} must be a number.")


def _clean_url(value: str) -> str:
    return (value or "").strip().rstrip("/")


class Config:

    # ============================================================
    # App Environment
    # ============================================================

    APP_ENV = os.getenv(
        "APP_ENV",
        os.getenv("FLASK_ENV", "production"),
    ).strip().lower()

    DEBUG = _env_bool(
        "FLASK_DEBUG",
        APP_ENV in {"dev", "development", "local"},
    )

    SHOW_ERROR_TRACE = _env_bool(
        "SHOW_ERROR_TRACE",
        DEBUG,
    )

    # ============================================================
    # Core Flask
    # ============================================================

    SECRET_KEY = (
        os.getenv("FLASK_SECRET_KEY", "").strip()
        or os.getenv("SECRET_KEY", "").strip()
    )

    if not SECRET_KEY:
        raise RuntimeError("FLASK_SECRET_KEY is required.")

    MAX_CONTENT_LENGTH = _env_int(
        "MAX_CONTENT_LENGTH",
        64 * 1024,
    )

    SESSION_COOKIE_SAMESITE = os.getenv(
        "SESSION_COOKIE_SAMESITE",
        "None",
    ).strip()

    SESSION_COOKIE_SECURE = _env_bool(
        "SESSION_COOKIE_SECURE",
        True,
    )

    # ============================================================
    # Frontend / Domain
    # ============================================================

    FRONTEND_URL = _clean_url(
        os.getenv(
            "FRONTEND_URL",
            "https://interpreter.jamaicantruestories.com",
        )
    )

    RETURN_URL = _clean_url(
        os.getenv(
            "RETURN_URL",
            FRONTEND_URL,
        )
    )

    UPGRADE_URL = _clean_url(
        os.getenv(
            "UPGRADE_URL",
            f"{FRONTEND_URL}/upgrade",
        )
    )

    # ============================================================
    # Allowed Origins / CORS
    # ============================================================

    DEFAULT_ALLOWED_ORIGINS = [
        "https://interpreter.jamaicantruestories.com",
        "https://jamaicantruestories.com",
        "https://www.jamaicantruestories.com",
        "https://plqwhd-jm.myshopify.com",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    ALLOWED_ORIGINS = (
        _csv_to_list(os.getenv("ALLOWED_ORIGINS", ""))
        or DEFAULT_ALLOWED_ORIGINS
    )

    TRUSTED_HOSTS = [
        "jamaicantruestories.com",
        "www.jamaicantruestories.com",
        "interpreter.jamaicantruestories.com",
        "plqwhd-jm.myshopify.com",
        "localhost",
        "127.0.0.1",
    ]

    # ============================================================
    # Templates
    # ============================================================

    TEMPLATE_INDEX = os.getenv(
        "TEMPLATE_INDEX",
        "index.html",
    ).strip()

    TEMPLATE_UPGRADE = os.getenv(
        "TEMPLATE_UPGRADE",
        "upgrade.html",
    ).strip()

    # ============================================================
    # Google Sheets
    # ============================================================

    SPREADSHEET_ID = os.getenv(
        "SPREADSHEET_ID",
        "",
    ).strip()

    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID is required.")

    WORKSHEET_NAME = os.getenv(
        "WORKSHEET_NAME",
        "Sheet1",
    ).strip()

    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "",
    ).strip()

    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "credentials.json",
    ).strip()

    GOOGLE_SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # ============================================================
    # Doctrine Mode
    # ============================================================

    DOCTRINE_MODE = _env_bool(
        "DOCTRINE_MODE",
        True,
    )

    # ============================================================
    # Narration Layer
    # ============================================================

    NARRATIVE_ENABLED = _env_bool(
        "NARRATIVE_ENABLED",
        True,
    )

    NARRATIVE_MAX_SYMBOLS = _env_int(
        "NARRATIVE_MAX_SYMBOLS",
        3,
    )

    NARRATION_ENABLED = _env_bool(
        "NARRATION_ENABLED",
        True,
    )

    NARRATION_MODE = os.getenv(
        "NARRATION_MODE",
        "deterministic_event",
    ).strip().lower()

    NARRATION_MAX_SYMBOLS = _env_int(
        "NARRATION_MAX_SYMBOLS",
        3,
    )

    NARRATION_INCLUDE_PROMPT_PAYLOAD = _env_bool(
        "NARRATION_INCLUDE_PROMPT_PAYLOAD",
        True,
    )

    # ============================================================
    # AI Narration
    # ============================================================

    AI_NARRATION_ENABLED = _env_bool(
        "AI_NARRATION_ENABLED",
        False,
    )

    AI_NARRATION_PROVIDER = os.getenv(
        "AI_NARRATION_PROVIDER",
        "none",
    ).strip().lower()

    AI_NARRATION_MODEL = os.getenv(
        "AI_NARRATION_MODEL",
        "",
    ).strip()

    AI_NARRATION_TIMEOUT_SECONDS = _env_int(
        "AI_NARRATION_TIMEOUT_SECONDS",
        20,
    )

    AI_NARRATION_MAX_INPUT_CHARS = _env_int(
        "AI_NARRATION_MAX_INPUT_CHARS",
        6000,
    )

    AI_NARRATION_MAX_OUTPUT_CHARS = _env_int(
        "AI_NARRATION_MAX_OUTPUT_CHARS",
        2000,
    )

    AI_NARRATION_TEMPERATURE = _env_float(
        "AI_NARRATION_TEMPERATURE",
        0.2,
    )

    AI_NARRATION_STRICT_DOCTRINE = _env_bool(
        "AI_NARRATION_STRICT_DOCTRINE",
        True,
    )

    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    # ============================================================
    # Doctrine Sheets
    # ============================================================

    SHEET_BASE_SYMBOLS = os.getenv(
        "SHEET_BASE_SYMBOLS",
        "BaseSymbols",
    ).strip()

    SHEET_BEHAVIOR_RULES = os.getenv(
        "SHEET_BEHAVIOR_RULES",
        "BehaviorRules",
    ).strip()

    SHEET_SIZE_STATE_RULES = os.getenv(
        "SHEET_SIZE_STATE_RULES",
        "SizeStateRules",
    ).strip()

    SHEET_LOCATION_RULES = os.getenv(
        "SHEET_LOCATION_RULES",
        "LocationRules",
    ).strip()

    SHEET_RELATIONSHIP_RULES = os.getenv(
        "SHEET_RELATIONSHIP_RULES",
        "RelationshipRules",
    ).strip()

    SHEET_ENDING_RULES = os.getenv(
        "SHEET_ENDING_RULES",
        "EndingRules",
    ).strip()

    SHEET_OVERRIDE_RULES = os.getenv(
        "SHEET_OVERRIDE_RULES",
        "OverrideRules",
    ).strip()

    SHEET_OUTPUT_TEMPLATES = os.getenv(
        "SHEET_OUTPUT_TEMPLATES",
        "OutputTemplates",
    ).strip()

    SHEET_DREAM_JOURNAL = os.getenv(
        "SHEET_DREAM_JOURNAL",
        "DreamJournal",
    ).strip()

    DOCTRINE_SHEET_NAMES = [
        SHEET_BASE_SYMBOLS,
        SHEET_BEHAVIOR_RULES,
        SHEET_SIZE_STATE_RULES,
        SHEET_LOCATION_RULES,
        SHEET_RELATIONSHIP_RULES,
        SHEET_ENDING_RULES,
        SHEET_OVERRIDE_RULES,
        SHEET_OUTPUT_TEMPLATES,
    ]

    # ============================================================
    # Expected Headers
    # ============================================================

    EXPECTED_HEADERS = {
        "Sheet1": [
            "Symbol",
            "Meaning",
            "Effects in the Physical Realm",
            "What to Do",
            "Keywords",
        ],
        "BaseSymbols": [
            "symbol",
            "spiritual_meaning",
            "physical_meaning",
            "action",
            "keywords",
            "priority",
            "active",
        ],
        "BehaviorRules": [
            "behavior_name",
            "keywords",
            "life_area_meaning",
            "physical_area_meaning",
            "action_modifier",
            "priority",
            "active",
        ],
        "LocationRules": [
            "location_name",
            "keywords",
            "life_area_meaning",
            "physical_area_meaning",
            "action_modifier",
            "priority",
            "active",
        ],
        "RelationshipRules": [
            "relationship_name",
            "keywords",
            "life_area_meaning",
            "physical_area_meaning",
            "action_modifier",
            "priority",
            "active",
        ],
        "SizeStateRules": [
            "state_name",
            "keywords",
            "life_area_meaning",
            "physical_area_meaning",
            "action_modifier",
            "priority",
            "active",
        ],
        "EndingRules": [
            "ending_name",
            "keywords",
            "meaning",
            "priority",
            "active",
        ],
        "OverrideRules": [
            "override_name",
            "conditions",
            "override_text",
            "priority",
            "active",
        ],
        "OutputTemplates": [
            "template_name",
            "template",
            "priority",
            "active",
        ],
    }

    # ============================================================
    # Matching Engine
    # ============================================================

    CACHE_TTL_SECONDS = _env_int(
        "CACHE_TTL_SECONDS",
        _env_int("SHEET_CACHE_TTL", 120),
    )

    DEBUG_MATCH = _env_bool(
        "DEBUG_MATCH",
        False if APP_ENV == "production" else True,
    )

    KEYWORD_GUARD_ENABLED = _env_bool(
        "KEYWORD_GUARD_ENABLED",
        True,
    )

    MAX_DREAM_LENGTH = _env_int(
        "MAX_DREAM_LENGTH",
        2500,
    )

    MIN_DREAM_LENGTH = _env_int(
        "MIN_DREAM_LENGTH",
        5,
    )

    MAX_RULE_HITS_PER_LAYER = _env_int(
        "MAX_RULE_HITS_PER_LAYER",
        4,
    )

    BASE_MATCH_TOP_K = _env_int(
        "BASE_MATCH_TOP_K",
        3,
    )

    # ============================================================
    # Admin
    # ============================================================

    ADMIN_KEY = (
        os.getenv("JTS_ADMIN_KEY", "").strip()
        or os.getenv("ADMIN_KEY", "").strip()
        or os.getenv("ADMIN_TOKEN", "").strip()
        or os.getenv("JTS_ADMIN", "").strip()
    )

    # ============================================================
    # Free Access / Hybrid Gate
    # ============================================================

    FREE_TRIES = _env_int(
        "FREE_QUOTA",
        _env_int("FREE_TRIES", 3),
    )

    COOKIE_NAME = os.getenv(
        "FREE_TRIES_COOKIE",
        "jts_free_tries_used",
    ).strip()

    SHADOW_WINDOW_HOURS = _env_int(
        "SHADOW_WINDOW_HOURS",
        72,
    )

    COOKIE_MAX_AGE = 60 * 60 * 24 * 365

    COUNTS_FILE = os.getenv(
        "COUNTS_FILE",
        "/data/usage_counts.json",
    ).strip()

    SUBSCRIBERS_FILE = os.getenv(
        "SUBSCRIBERS_FILE",
        "/data/subscribers.json",
    ).strip()

    DREAM_PACKS_FILE = os.getenv(
        "DREAM_PACKS_FILE",
        "/data/dream_packs.json",
    ).strip()

    # ============================================================
    # Stripe
    # ============================================================

    STRIPE_SECRET_KEY = os.getenv(
        "STRIPE_SECRET_KEY",
        "",
    ).strip()

    STRIPE_WEBHOOK_SECRET = os.getenv(
        "STRIPE_WEBHOOK_SECRET",
        "",
    ).strip()

    PRICE_WEEKLY = os.getenv(
        "PRICE_WEEKLY",
        "",
    ).strip()

    PRICE_MONTHLY = os.getenv(
        "PRICE_MONTHLY",
        "",
    ).strip()

    DEFAULT_STRIPE_PRICE_ID = (
        PRICE_WEEKLY
        or PRICE_MONTHLY
    )

    PRICE_DREAM_PACK = os.getenv(
        "PRICE_DREAM_PACK",
        "",
    ).strip()

    DREAM_PACK_USES = _env_int(
        "DREAM_PACK_USES",
        3,
    )

    DREAM_PACK_HOURS = _env_int(
        "DREAM_PACK_HOURS",
        72,
    )

    # ============================================================
    # Billing / Checkout Labels
    # ============================================================

    BILLING_PORTAL_ENABLED = _env_bool(
        "BILLING_PORTAL_ENABLED",
        False,
    )

    SUBSCRIPTION_LABEL = os.getenv(
        "SUBSCRIPTION_LABEL",
        "Subscription",
    ).strip()

    DREAM_PACK_LABEL = os.getenv(
        "DREAM_PACK_LABEL",
        "Dream Pack",
    ).strip()

    # ============================================================
    # Validation
    # ============================================================

    @classmethod
    def validate(cls):
        if cls.SESSION_COOKIE_SAMESITE not in {"Lax", "Strict", "None"}:
            raise RuntimeError(
                "SESSION_COOKIE_SAMESITE must be Lax, Strict, or None."
            )

        if cls.SESSION_COOKIE_SAMESITE == "None" and not cls.SESSION_COOKIE_SECURE:
            raise RuntimeError(
                "SESSION_COOKIE_SECURE must be true when SESSION_COOKIE_SAMESITE=None."
            )

        if not cls.FRONTEND_URL.startswith(("http://", "https://")):
            raise RuntimeError("FRONTEND_URL must be a valid URL.")

        if not cls.RETURN_URL.startswith(("http://", "https://", "/")):
            raise RuntimeError("RETURN_URL must be a valid URL or relative path.")

        if cls.MIN_DREAM_LENGTH < 1:
            raise RuntimeError("MIN_DREAM_LENGTH must be at least 1.")

        if cls.MAX_DREAM_LENGTH <= cls.MIN_DREAM_LENGTH:
            raise RuntimeError(
                "MAX_DREAM_LENGTH must be greater than MIN_DREAM_LENGTH."
            )

        if cls.MAX_CONTENT_LENGTH < cls.MAX_DREAM_LENGTH:
            raise RuntimeError(
                "MAX_CONTENT_LENGTH should be greater than MAX_DREAM_LENGTH."
            )

        if cls.BASE_MATCH_TOP_K < 1:
            raise RuntimeError("BASE_MATCH_TOP_K must be at least 1.")

        if cls.MAX_RULE_HITS_PER_LAYER < 1:
            raise RuntimeError("MAX_RULE_HITS_PER_LAYER must be at least 1.")

        if cls.FREE_TRIES < 0:
            raise RuntimeError("FREE_TRIES cannot be negative.")

        if cls.SHADOW_WINDOW_HOURS < 1:
            raise RuntimeError("SHADOW_WINDOW_HOURS must be at least 1.")

        if cls.NARRATIVE_MAX_SYMBOLS < 1:
            raise RuntimeError("NARRATIVE_MAX_SYMBOLS must be at least 1.")

        if cls.NARRATION_MAX_SYMBOLS < 1:
            raise RuntimeError("NARRATION_MAX_SYMBOLS must be at least 1.")

        if cls.AI_NARRATION_TIMEOUT_SECONDS < 1:
            raise RuntimeError(
                "AI_NARRATION_TIMEOUT_SECONDS must be at least 1."
            )

        if cls.AI_NARRATION_MAX_INPUT_CHARS < 500:
            raise RuntimeError("AI_NARRATION_MAX_INPUT_CHARS is set too low.")

        if cls.AI_NARRATION_MAX_OUTPUT_CHARS < 200:
            raise RuntimeError("AI_NARRATION_MAX_OUTPUT_CHARS is set too low.")

        if (
            cls.AI_NARRATION_TEMPERATURE < 0
            or cls.AI_NARRATION_TEMPERATURE > 1.5
        ):
            raise RuntimeError(
                "AI_NARRATION_TEMPERATURE must be between 0 and 1.5."
            )

        if cls.AI_NARRATION_ENABLED:
            if cls.AI_NARRATION_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is required when AI_NARRATION_PROVIDER=openai."
                )

            if not cls.AI_NARRATION_MODEL:
                raise RuntimeError(
                    "AI_NARRATION_MODEL is required when AI narration is enabled."
                )

        if cls.STRIPE_SECRET_KEY and not cls.DEFAULT_STRIPE_PRICE_ID:
            print(
                "Warning: STRIPE_SECRET_KEY is set but no subscription price ID is configured.",
                flush=True,
            )

        if cls.STRIPE_SECRET_KEY and not cls.PRICE_DREAM_PACK:
            print(
                "Warning: STRIPE_SECRET_KEY is set but PRICE_DREAM_PACK is not configured.",
                flush=True,
            )

        if cls.DREAM_PACK_USES < 1:
            raise RuntimeError("DREAM_PACK_USES must be at least 1.")

        if cls.DREAM_PACK_HOURS < 1:
            raise RuntimeError("DREAM_PACK_HOURS must be at least 1.")
