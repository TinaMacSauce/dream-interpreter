import os
from typing import List


def _csv_to_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # ============================================================
    # Core Flask
    # ============================================================
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "").strip() or os.getenv("SECRET_KEY", "").strip()
    if not SECRET_KEY:
        raise RuntimeError("FLASK_SECRET_KEY is required.")

    MAX_CONTENT_LENGTH = 64 * 1024
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "None").strip()
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", True)

    # ============================================================
    # Allowed Origins / CORS
    # ============================================================
    DEFAULT_ALLOWED_ORIGINS = [
        "https://jamaicantruestories.com",
        "https://www.jamaicantruestories.com",
        "https://interpreter.jamaicantruestories.com",
        "https://plqwhd-jm.myshopify.com",
    ]
    ALLOWED_ORIGINS = _csv_to_list(os.getenv("ALLOWED_ORIGINS", "")) or DEFAULT_ALLOWED_ORIGINS

    # ============================================================
    # App / Templates
    # ============================================================
    RETURN_URL = os.getenv(
        "RETURN_URL",
        "https://jamaicantruestories.com/pages/dream-interpreter",
    ).strip()
    TEMPLATE_INDEX = os.getenv("TEMPLATE_INDEX", "index.html").strip()
    TEMPLATE_UPGRADE = os.getenv("TEMPLATE_UPGRADE", "upgrade.html").strip()

    # ============================================================
    # Google Sheets
    # ============================================================
    SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID is required.")

    WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json").strip()

    GOOGLE_SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # ============================================================
    # Doctrine Mode / Narrative
    # ============================================================
    DOCTRINE_MODE = _env_bool("DOCTRINE_MODE", True)

    NARRATIVE_ENABLED = _env_bool("NARRATIVE_ENABLED", True)
    NARRATIVE_MAX_SYMBOLS = int(os.getenv("NARRATIVE_MAX_SYMBOLS", "3"))

    # ============================================================
    # Narration Layer
    # ============================================================
    NARRATION_ENABLED = _env_bool("NARRATION_ENABLED", True)
    NARRATION_MODE = os.getenv("NARRATION_MODE", "deterministic").strip().lower()
    NARRATION_MAX_SYMBOLS = int(os.getenv("NARRATION_MAX_SYMBOLS", "3"))
    NARRATION_INCLUDE_PROMPT_PAYLOAD = _env_bool("NARRATION_INCLUDE_PROMPT_PAYLOAD", True)

    # ============================================================
    # Future AI Narration Controls
    # ============================================================
    AI_NARRATION_ENABLED = _env_bool("AI_NARRATION_ENABLED", False)
    AI_NARRATION_PROVIDER = os.getenv("AI_NARRATION_PROVIDER", "none").strip().lower()
    AI_NARRATION_MODEL = os.getenv("AI_NARRATION_MODEL", "").strip()
    AI_NARRATION_TIMEOUT_SECONDS = int(os.getenv("AI_NARRATION_TIMEOUT_SECONDS", "20"))
    AI_NARRATION_MAX_INPUT_CHARS = int(os.getenv("AI_NARRATION_MAX_INPUT_CHARS", "6000"))
    AI_NARRATION_MAX_OUTPUT_CHARS = int(os.getenv("AI_NARRATION_MAX_OUTPUT_CHARS", "2000"))
    AI_NARRATION_TEMPERATURE = float(os.getenv("AI_NARRATION_TEMPERATURE", "0.2"))
    AI_NARRATION_STRICT_DOCTRINE = _env_bool("AI_NARRATION_STRICT_DOCTRINE", True)

    # Optional future API key slots
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

    # ============================================================
    # Doctrine Sheets
    # ============================================================
    SHEET_BASE_SYMBOLS = os.getenv("SHEET_BASE_SYMBOLS", "BaseSymbols").strip()
    SHEET_BEHAVIOR_RULES = os.getenv("SHEET_BEHAVIOR_RULES", "BehaviorRules").strip()
    SHEET_SIZE_STATE_RULES = os.getenv("SHEET_SIZE_STATE_RULES", "SizeStateRules").strip()
    SHEET_LOCATION_RULES = os.getenv("SHEET_LOCATION_RULES", "LocationRules").strip()
    SHEET_RELATIONSHIP_RULES = os.getenv("SHEET_RELATIONSHIP_RULES", "RelationshipRules").strip()
    SHEET_OVERRIDE_RULES = os.getenv("SHEET_OVERRIDE_RULES", "OverrideRules").strip()
    SHEET_OUTPUT_TEMPLATES = os.getenv("SHEET_OUTPUT_TEMPLATES", "OutputTemplates").strip()
    SHEET_DREAM_JOURNAL = os.getenv("SHEET_DREAM_JOURNAL", "DreamJournal").strip()

    DOCTRINE_SHEET_NAMES = [
        SHEET_BASE_SYMBOLS,
        SHEET_BEHAVIOR_RULES,
        SHEET_SIZE_STATE_RULES,
        SHEET_LOCATION_RULES,
        SHEET_RELATIONSHIP_RULES,
        SHEET_OVERRIDE_RULES,
        SHEET_OUTPUT_TEMPLATES,
    ]

    # ============================================================
    # Matching / Engine Limits
    # ============================================================
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", os.getenv("SHEET_CACHE_TTL", "120")))
    DEBUG_MATCH = _env_bool("DEBUG_MATCH", False)

    KEYWORD_GUARD_ENABLED = True
    MAX_DREAM_LENGTH = int(os.getenv("MAX_DREAM_LENGTH", "2500"))
    MIN_DREAM_LENGTH = int(os.getenv("MIN_DREAM_LENGTH", "5"))
    MAX_RULE_HITS_PER_LAYER = int(os.getenv("MAX_RULE_HITS_PER_LAYER", "4"))
    BASE_MATCH_TOP_K = int(os.getenv("BASE_MATCH_TOP_K", "3"))

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
    FREE_TRIES = int(os.getenv("FREE_QUOTA", os.getenv("FREE_TRIES", "3")))
    COOKIE_NAME = os.getenv("FREE_TRIES_COOKIE", "jts_free_tries_used").strip()
    SHADOW_WINDOW_HOURS = int(os.getenv("SHADOW_WINDOW_HOURS", "72"))

    COOKIE_MAX_AGE = 60 * 60 * 24 * 365

    COUNTS_FILE = os.getenv("COUNTS_FILE", "/data/usage_counts.json").strip()
    SUBSCRIBERS_FILE = os.getenv("SUBSCRIBERS_FILE", "/data/subscribers.json").strip()

    # ============================================================
    # Stripe
    # ============================================================
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

    PRICE_WEEKLY = os.getenv("PRICE_WEEKLY", "").strip()
    PRICE_MONTHLY = os.getenv("PRICE_MONTHLY", "").strip()
    DEFAULT_STRIPE_PRICE_ID = PRICE_WEEKLY or PRICE_MONTHLY

    PRICE_DREAM_PACK = os.getenv("PRICE_DREAM_PACK", "").strip()
    DREAM_PACK_USES = int(os.getenv("DREAM_PACK_USES", "3"))
    DREAM_PACK_HOURS = int(os.getenv("DREAM_PACK_HOURS", "72"))

    # ============================================================
    # Expected Headers
    # ============================================================
    EXPECTED_HEADERS = {
        WORKSHEET_NAME: [
            "symbol",
            "base_spiritual_meaning",
            "base_physical_effects",
            "base_action",
            "keywords",
            "category",
            "active",
        ],
        SHEET_BASE_SYMBOLS: [
            "symbol",
            "category",
            "base_spiritual_meaning",
            "base_physical_effects",
            "base_action",
            "keywords",
            "active",
        ],
        SHEET_BEHAVIOR_RULES: [
            "behavior_name",
            "keywords",
            "meaning_modifier",
            "physical_modifier",
            "action_modifier",
            "priority",
            "active",
        ],
        SHEET_SIZE_STATE_RULES: [
            "state_name",
            "keywords",
            "meaning_modifier",
            "physical_modifier",
            "action_modifier",
            "priority",
            "active",
        ],
        SHEET_LOCATION_RULES: [
            "location_name",
            "keywords",
            "life_area_meaning",
            "physical_area_meaning",
            "action_modifier",
            "priority",
            "active",
        ],
        SHEET_RELATIONSHIP_RULES: [
            "relationship_name",
            "keywords",
            "meaning_modifier",
            "physical_modifier",
            "action_modifier",
            "priority",
            "active",
        ],
        SHEET_OVERRIDE_RULES: [
            "override_name",
            "condition",
            "final_spiritual_meaning",
            "final_physical_effects",
            "final_action",
            "priority",
            "active",
        ],
        SHEET_OUTPUT_TEMPLATES: [
            "template_type",
            "template_text",
            "active",
        ],
        SHEET_DREAM_JOURNAL: [
            "entry_id",
            "created_at",
            "email",
            "dream_text",
            "spiritual_meaning",
            "effects_in_physical_realm",
            "what_to_do",
            "full_interpretation",
            "receipt_id",
            "top_symbols",
            "seal_status",
            "seal_type",
            "seal_risk",
            "engine_mode",
            "access_type",
            "is_saved",
            "notes",
        ],
    }

    # ============================================================
    # Optional sanity checks
    # ============================================================
    @classmethod
    def validate(cls) -> None:
        if cls.MIN_DREAM_LENGTH < 1:
            raise RuntimeError("MIN_DREAM_LENGTH must be at least 1.")
        if cls.MAX_DREAM_LENGTH <= cls.MIN_DREAM_LENGTH:
            raise RuntimeError("MAX_DREAM_LENGTH must be greater than MIN_DREAM_LENGTH.")
        if cls.BASE_MATCH_TOP_K < 1:
            raise RuntimeError("BASE_MATCH_TOP_K must be at least 1.")
        if cls.MAX_RULE_HITS_PER_LAYER < 1:
            raise RuntimeError("MAX_RULE_HITS_PER_LAYER must be at least 1.")
        if cls.FREE_TRIES < 0:
            raise RuntimeError("FREE_TRIES cannot be negative.")
        if cls.NARRATIVE_MAX_SYMBOLS < 1:
            raise RuntimeError("NARRATIVE_MAX_SYMBOLS must be at least 1.")
        if cls.NARRATION_MAX_SYMBOLS < 1:
            raise RuntimeError("NARRATION_MAX_SYMBOLS must be at least 1.")
        if cls.AI_NARRATION_TIMEOUT_SECONDS < 1:
            raise RuntimeError("AI_NARRATION_TIMEOUT_SECONDS must be at least 1.")
        if cls.AI_NARRATION_MAX_INPUT_CHARS < 500:
            raise RuntimeError("AI_NARRATION_MAX_INPUT_CHARS is set too low.")
        if cls.AI_NARRATION_MAX_OUTPUT_CHARS < 200:
            raise RuntimeError("AI_NARRATION_MAX_OUTPUT_CHARS is set too low.")
        if cls.AI_NARRATION_TEMPERATURE < 0 or cls.AI_NARRATION_TEMPERATURE > 1.5:
            raise RuntimeError("AI_NARRATION_TEMPERATURE must be between 0 and 1.5.")
