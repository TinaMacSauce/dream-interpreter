import os
import json
import time
import re
import secrets
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from flask import (
    Flask,
    request,
    jsonify,
    make_response,
    Response,
    redirect,
    url_for,
    session,
    render_template,
)
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials

try:
    import stripe
except Exception:
    stripe = None


# ============================================================
# App setup
# ============================================================
app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "").strip() or secrets.token_hex(32)
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

DEFAULT_ALLOWED = [
    "https://jamaicantruestories.com",
    "https://www.jamaicantruestories.com",
    "https://interpreter.jamaicantruestories.com",
    "https://plqwhd-jm.myshopify.com",
]

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or DEFAULT_ALLOWED

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Admin-Key"],
    methods=["GET", "POST", "OPTIONS"],
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
RETURN_URL = os.getenv(
    "RETURN_URL",
    "https://jamaicantruestories.com/pages/dream-interpreter",
).strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()
TEMPLATE_INDEX = os.getenv("TEMPLATE_INDEX", "index.html").strip()
TEMPLATE_UPGRADE = os.getenv("TEMPLATE_UPGRADE", "upgrade.html").strip()

ADMIN_KEY = (
    os.getenv("JTS_ADMIN_KEY", "").strip()
    or os.getenv("ADMIN_KEY", "").strip()
    or os.getenv("ADMIN_TOKEN", "").strip()
    or os.getenv("JTS_ADMIN", "").strip()
)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", os.getenv("SHEET_CACHE_TTL", "120")))
DEBUG_MATCH = os.getenv("DEBUG_MATCH", "").strip().lower() in {"1", "true", "yes", "on"}

NARRATIVE_ENABLED = os.getenv("NARRATIVE_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
NARRATIVE_MAX_SYMBOLS = int(os.getenv("NARRATIVE_MAX_SYMBOLS", "3"))

DOCTRINE_MODE = os.getenv("DOCTRINE_MODE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

KEYWORD_GUARD_ENABLED = True

MAX_DREAM_LENGTH = int(os.getenv("MAX_DREAM_LENGTH", "2500"))
MIN_DREAM_LENGTH = int(os.getenv("MIN_DREAM_LENGTH", "5"))
MAX_RULE_HITS_PER_LAYER = int(os.getenv("MAX_RULE_HITS_PER_LAYER", "4"))
BASE_MATCH_TOP_K = int(os.getenv("BASE_MATCH_TOP_K", "3"))

# ============================================================
# Doctrine sheet names
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
# Expected headers for self-healing worksheets
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
# Hybrid Gate Settings
# ============================================================
FREE_TRIES = int(os.getenv("FREE_QUOTA", os.getenv("FREE_TRIES", "3")))
COOKIE_NAME = os.getenv("FREE_TRIES_COOKIE", "jts_free_tries_used")
SHADOW_WINDOW_HOURS = int(os.getenv("SHADOW_WINDOW_HOURS", "72"))

COUNTS_FILE = os.getenv("COUNTS_FILE", "/data/usage_counts.json")
SUBSCRIBERS_FILE = os.getenv("SUBSCRIBERS_FILE", "/data/subscribers.json")

USAGE_COUNTS_PATH = Path(COUNTS_FILE)
SUBSCRIBERS_PATH = Path(SUBSCRIBERS_FILE)

# ============================================================
# Stripe Settings
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
# Cookie settings
# ============================================================
COOKIE_SAMESITE = "None"
COOKIE_SECURE = True
COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# ============================================================
# Caches
# ============================================================
_LEGACY_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "rows": [], "headers": []}
_DOCTRINE_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "sheets": {}, "headers": {}}


# ============================================================
# Helpers
# ============================================================
def _preflight_ok():
    return make_response("", 204)


def _log(*args):
    if DEBUG_MATCH:
        print(*args, flush=True)


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"[^a-z0-9\s_]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _validate_email(email: str) -> bool:
    email = (email or "").strip()
    return bool(email and "@" in email and "." in email.split("@")[-1])


def _extract_email_from_request() -> str:
    data = request.get_json(silent=True) or {}
    return _normalize_email(
        data.get("email")
        or request.args.get("email")
        or request.form.get("email")
        or session.get("subscriber_email")
        or ""
    )


def _persist_email_to_session(email: str):
    email_n = _normalize_email(email)
    if _validate_email(email_n):
        session["subscriber_email"] = email_n


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at",
    "be", "but", "by",
    "for", "from",
    "i", "if", "in", "into", "is", "it",
    "me", "my",
    "of", "on", "or",
    "so",
    "the", "they", "this", "that", "these", "those", "to",
    "was", "we", "were", "with",
    "you", "your",
}


def _is_useless_keyword(token: str) -> bool:
    token = _normalize_text(token)
    if not token:
        return True
    if token.isdigit():
        return True

    words = token.split()
    if not words:
        return True

    if len(words) == 1:
        w = words[0]
        if w in STOP_WORDS:
            return True
        if len(w) < 3:
            return True

    return False


def _split_keywords(s: str) -> List[str]:
    if not s:
        return []

    parts = re.split(r"[,\|;]+", s)
    out = []
    seen = set()

    for p in parts:
        p = _normalize_text(p)
        if not p:
            continue
        if KEYWORD_GUARD_ENABLED and _is_useless_keyword(p):
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    return out


def _sanitize_keywords_for_storage(keywords_raw: str) -> str:
    return ", ".join(_split_keywords(keywords_raw))


def _clean_sentence(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";")
    s = re.sub(r"([.!?]){2,}", r"\1", s)
    return s


def _strip_trailing_punct(s: str) -> str:
    s = _clean_sentence(s)
    return s.rstrip(" .!?\t\r\n")


def _ensure_terminal_punct(s: str) -> str:
    s = _clean_sentence(s)
    if not s:
        return ""
    if s[-1] not in ".!?":
        s += "."
    return s


def _capitalize_first(s: str) -> str:
    s = _clean_sentence(s)
    if not s:
        return ""
    return s[:1].upper() + s[1:]


def _dedupe_words_soft(text: str) -> str:
    text = _clean_sentence(text)
    if not text:
        return ""

    words = text.split()
    out = []
    prev = ""

    for w in words:
        wl = w.lower().strip(".,;:!?")
        if wl and wl == prev:
            continue
        out.append(w)
        prev = wl

    return " ".join(out)


def _normalize_fragment_key(text: str) -> str:
    text = _normalize_text(text)
    if not text:
        return ""

    lead_ins = [
        "this may show up as",
        "in practical terms this may show up as",
        "the actions in the dream suggest",
        "the condition of what appeared points to",
        "the setting connects this message to",
        "the people involved point to",
        "the ending seals this as",
        "the ending confirms",
        "overall this is a",
        "the strongest message in this dream is",
        "the strongest message here points to",
        "this dream points to",
    ]
    for lead in lead_ins:
        if text.startswith(lead):
            text = text[len(lead):].strip()
            break

    text = re.sub(r"\b(points to|represents|shows|suggests|indicates|reveals|confirms)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_semantic_fragments(parts: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for raw in parts:
        cleaned = _strip_trailing_punct(_clean_debug_like_phrase(raw))
        if not cleaned:
            continue

        key = _normalize_fragment_key(cleaned)
        if not key:
            continue

        if key in seen:
            continue

        duplicate = False
        key_words = set(key.split())
        for old in list(seen):
            old_words = set(old.split())
            if not old_words or not key_words:
                continue
            overlap = len(key_words & old_words)
            shortest = max(1, min(len(key_words), len(old_words)))
            if overlap / shortest >= 0.8:
                duplicate = True
                break

        if duplicate:
            continue

        seen.add(key)
        out.append(cleaned)

    return out


def _limit_phrase_list(parts: List[str], max_items: int = 4) -> List[str]:
    parts = _dedupe_semantic_fragments(parts)
    if max_items <= 0:
        return []
    return parts[:max_items]


def _compact_clause(parts: List[str], max_items: int = 3) -> str:
    items = _limit_phrase_list(parts, max_items=max_items)
    if not items:
        return ""
    return _human_join(items)


def _first_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    return _sentence(sentences[0])


def _compress_action_phrases(parts: List[str], max_items: int = 4) -> List[str]:
    cleaned = [_normalize_action_phrase(x) for x in parts if x]
    cleaned = _dedupe_semantic_fragments(cleaned)

    strong = []
    secondary = []
    for item in cleaned:
        lowered = _normalize_text(item)
        if any(x in lowered for x in ["pray", "alert", "guard", "discern", "wisdom", "move carefully", "do not move blindly"]):
            strong.append(item)
        else:
            secondary.append(item)

    ordered = strong + secondary
    return ordered[:max_items]


def _compress_effect_phrases(parts: List[str], max_items: int = 5) -> List[str]:
    cleaned = [_normalize_effect_phrase(x) for x in parts if x]
    cleaned = _dedupe_semantic_fragments(cleaned)

    priority_buckets = {"high": [], "mid": [], "low": []}
    for item in cleaned:
        lowered = _normalize_text(item)
        if any(x in lowered for x in ["pressure", "fear", "conflict", "restriction", "delay", "attack", "deception"]):
            priority_buckets["high"].append(item)
        elif any(x in lowered for x in ["relationship", "access", "boundary", "movement", "progress"]):
            priority_buckets["mid"].append(item)
        else:
            priority_buckets["low"].append(item)

    ordered = priority_buckets["high"] + priority_buckets["mid"] + priority_buckets["low"]
    return ordered[:max_items]


def _compact_spiritual_meaning(
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    override_hit: Optional[Dict[str, Any]],
    seal: Dict[str, Any],
) -> str:
    override_spiritual = _strip_trailing_punct((override_hit or {}).get("spiritual", ""))
    if override_spiritual:
        return override_spiritual

    phrases = []
    for row, _score, _hit in base_matches[:2]:
        symbol = _strip_trailing_punct(_get_base_symbol_input(row))
        meaning = _strip_trailing_punct(_get_base_symbol_meaning(row))
        if symbol and meaning:
            phrases.append(f"{symbol} points to {meaning}")
        elif meaning:
            phrases.append(meaning)
        elif symbol:
            phrases.append(symbol)

    compact = _compact_clause(phrases, max_items=2)
    if compact:
        return compact

    seal_message = _strip_trailing_punct(seal.get("message", ""))
    return seal_message


def _scenario_signal_score(
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> int:
    score = 0
    score += len(behaviors) * 4
    score += len(states) * 2
    score += len(locations) * 2
    score += len(relationships) * 2
    if len(behaviors) >= 2:
        score += 6
    if len(behaviors) >= 3:
        score += 4
    if relationships and locations:
        score += 2
    return score


def _rule_names_set(hits: List[Dict[str, Any]]) -> set:
    return {_normalize_text(x.get("name", "")) for x in hits if x.get("name")}


def _compact_rule_meaning_clause(hits: List[Dict[str, Any]], getter, max_items: int = 2) -> str:
    return _compact_clause([getter(x["row"]) for x in hits if getter(x["row"])], max_items=max_items)


def _build_primary_focus(
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
    seal: Dict[str, Any],
) -> Dict[str, str]:
    if override_hit and _strip_trailing_punct((override_hit or {}).get("spiritual", "")):
        return {
            "mode": "override",
            "lead": _strip_trailing_punct((override_hit or {}).get("spiritual", "")),
            "behavior": "",
            "state": "",
            "location": "",
            "relationship": "",
        }

    behavior_clause = _compact_rule_meaning_clause(behaviors, _get_behavior_meaning_modifier, max_items=2)
    state_clause = _compact_rule_meaning_clause(states, _get_state_meaning_modifier, max_items=1)
    location_clause = _compact_rule_meaning_clause(locations, _get_location_life_area_meaning, max_items=1)
    relationship_clause = _compact_rule_meaning_clause(relationships, _get_relationship_meaning_modifier, max_items=1)
    symbol_clause = _compact_spiritual_meaning(base_matches, None, seal)

    scenario_score = _scenario_signal_score(behaviors, states, locations, relationships)

    if scenario_score >= 10 and behavior_clause:
        lead_parts = [behavior_clause]
        if location_clause:
            lead_parts.append(f"around {location_clause}")
        if relationship_clause:
            lead_parts.append(f"involving {relationship_clause}")
        if state_clause:
            lead_parts.append(f"with {state_clause}")
        return {
            "mode": "scenario",
            "lead": _clean_sentence(" ".join(lead_parts)),
            "behavior": behavior_clause,
            "state": state_clause,
            "location": location_clause,
            "relationship": relationship_clause,
        }

    return {
        "mode": "symbol",
        "lead": symbol_clause,
        "behavior": behavior_clause,
        "state": state_clause,
        "location": location_clause,
        "relationship": relationship_clause,
    }


def _sentence(text: str) -> str:
    text = _dedupe_words_soft(text)
    text = _capitalize_first(text)
    return _ensure_terminal_punct(text)


def _clean_debug_like_phrase(text: str) -> str:
    text = _clean_sentence(text)
    if not text:
        return ""

    banned_starts = [
        "logic layer",
        "behavior detected",
        "state detected",
        "location detected",
        "relationship detected",
    ]
    lower = text.lower()
    for b in banned_starts:
        if lower.startswith(b):
            return ""
    return text


def _normalize_effect_phrase(text: str) -> str:
    text = _strip_trailing_punct(text)
    if not text:
        return ""

    lowered = text.lower()
    replacements = {
        "active attack": "an active attack",
        "intensified": "heightened intensity",
        "personal life": "your personal life",
        "enemy or deception": "pressure through deception or opposition",
    }
    return replacements.get(lowered, text)


def _normalize_action_phrase(text: str) -> str:
    text = _strip_trailing_punct(text)
    if not text:
        return ""

    lowered = text.lower()
    if lowered.startswith("what to do:"):
        text = text.split(":", 1)[-1].strip()
        lowered = text.lower()

    elite_map = {
        "pray and be alert": "Stay prayerful and alert. Be mindful of spiritual influence around you",
        "pray for wisdom and confirmation": "Pray for wisdom and confirmation. Stay spiritually grounded as clarity comes",
        "pray": "Pray and remain spiritually steady",
        "be alert": "Stay alert and guard what concerns you",
    }
    return elite_map.get(lowered, text)


def _merge_natural_paragraphs(parts: List[str]) -> str:
    cleaned = []
    seen = set()

    for p in parts:
        p = _clean_debug_like_phrase(p)
        p = _clean_sentence(p)
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    return "\n".join(cleaned).strip()


def _compress_phrase_list(parts: List[str]) -> List[str]:
    out = []
    seen = set()
    for p in parts:
        p = _strip_trailing_punct(p)
        if not p:
            continue
        key = _normalize_text(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _human_join(parts: List[str]) -> str:
    parts = _compress_phrase_list(parts)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _safe_debug_payload_preview(data: Dict[str, Any], max_len: int = 500) -> Dict[str, Any]:
    try:
        preview = {}
        for k, v in (data or {}).items():
            if isinstance(v, str):
                preview[k] = v[:max_len] + ("..." if len(v) > max_len else "")
            else:
                preview[k] = v
        return preview
    except Exception:
        return {"_debug_preview_error": True}


def _compile_boundary_regex(token: str) -> re.Pattern:
    token_n = _normalize_text(token)
    if not token_n:
        return re.compile(r"(?!x)x")
    return re.compile(rf"(?<!\w){re.escape(token_n)}(?!\w)")


def _tokenize_words(s: str) -> List[str]:
    return [w for w in _normalize_text(s).split() if w]


def _contains_phrase(text_norm: str, phrase: str) -> bool:
    if not text_norm or not phrase:
        return False
    return bool(_compile_boundary_regex(phrase).search(text_norm))


def _find_phrase_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    text_n = _normalize_text(text)
    phrase_n = _normalize_text(phrase)
    if not text_n or not phrase_n:
        return []
    rx = re.compile(rf"(?<!\w){re.escape(phrase_n)}(?!\w)")
    return [m.span() for m in rx.finditer(text_n)]


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _is_span_blocked(span: Tuple[int, int], used_spans: List[Tuple[int, int]]) -> bool:
    for u in used_spans:
        if _spans_overlap(span, u):
            return True
    return False


def _normalize_yes_no(v: str) -> bool:
    return str(v or "").strip().lower() in {"yes", "y", "true", "1", "active", "on"}


def _row_get(row: Dict, *keys: str) -> str:
    normalized = {_normalize_header(k): v for k, v in row.items()}
    for key in keys:
        nk = _normalize_header(key)
        if nk in normalized and str(normalized.get(nk, "")).strip():
            return str(normalized.get(nk, "")).strip()
    return ""


def _validate_dream_text(dream: str) -> Optional[str]:
    if not dream:
        return "Missing 'dream' or 'text'."
    if len(dream.strip()) < MIN_DREAM_LENGTH:
        return f"Dream text is too short. Please enter at least {MIN_DREAM_LENGTH} characters."
    if len(dream) > MAX_DREAM_LENGTH:
        return f"Dream text is too long. Maximum allowed is {MAX_DREAM_LENGTH} characters."
    return None


def _normalize_category(value: str) -> str:
    return _normalize_text(value or "") or "unknown"


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [c.strip() for c in chunks if c.strip()]


def _extract_dream_ending_text(dream: str) -> str:
    sentences = _split_sentences(dream)
    if not sentences:
        return dream.strip()
    if len(sentences) == 1:
        return sentences[-1]
    return " ".join(sentences[-2:]).strip()


# ============================================================
# File utilities
# ============================================================
def _read_json_file(path: Path, default: Any):
    try:
        if not path.exists():
            return default
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return default
        return json.loads(raw)
    except Exception:
        return default


def _write_json_file_atomic(path: Path, data: Any):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        _log("JSON write failed:", str(path), str(e))


def _ensure_files_exist():
    if not USAGE_COUNTS_PATH.exists():
        _write_json_file_atomic(USAGE_COUNTS_PATH, {"ip_shadow": {}})
    if not SUBSCRIBERS_PATH.exists():
        _write_json_file_atomic(SUBSCRIBERS_PATH, {})


_ensure_files_exist()

# ============================================================
# Hybrid gate + subscriber helpers
# ============================================================
def _get_client_ip() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "0.0.0.0"


def _get_cookie_tries_used() -> int:
    try:
        return int(request.cookies.get(COOKIE_NAME, "0"))
    except Exception:
        return 0


def _load_usage_counts() -> Dict[str, Any]:
    data = _read_json_file(USAGE_COUNTS_PATH, {})
    if not isinstance(data, dict):
        data = {}
    if "ip_shadow" not in data or not isinstance(data["ip_shadow"], dict):
        data["ip_shadow"] = {}
    return data


def _save_usage_counts(data: Dict[str, Any]):
    _write_json_file_atomic(USAGE_COUNTS_PATH, data)


def _shadow_get(ip: str) -> Dict[str, Any]:
    data = _load_usage_counts()
    shadow = data["ip_shadow"]
    rec = shadow.get(ip) or {"count": 0, "first_seen": None, "last_seen": None}

    now = datetime.utcnow()
    window = timedelta(hours=SHADOW_WINDOW_HOURS)

    if not rec.get("first_seen"):
        rec["first_seen"] = now.isoformat() + "Z"
    if not rec.get("last_seen"):
        rec["last_seen"] = now.isoformat() + "Z"

    try:
        first = datetime.fromisoformat(str(rec["first_seen"]).replace("Z", ""))
        if now - first > window:
            rec["count"] = 0
            rec["first_seen"] = now.isoformat() + "Z"
            rec["last_seen"] = now.isoformat() + "Z"
    except Exception:
        rec["count"] = 0
        rec["first_seen"] = now.isoformat() + "Z"
        rec["last_seen"] = now.isoformat() + "Z"

    shadow[ip] = rec
    _save_usage_counts(data)
    return rec


def _shadow_count(ip: str) -> int:
    rec = _shadow_get(ip)
    try:
        return int(rec.get("count", 0))
    except Exception:
        return 0


def _shadow_increment(ip: str) -> int:
    data = _load_usage_counts()
    shadow = data["ip_shadow"]
    rec = shadow.get(ip) or {"count": 0, "first_seen": None, "last_seen": None}

    now = datetime.utcnow()
    window = timedelta(hours=SHADOW_WINDOW_HOURS)

    try:
        if rec.get("first_seen"):
            first = datetime.fromisoformat(str(rec["first_seen"]).replace("Z", ""))
            if now - first > window:
                rec["count"] = 0
                rec["first_seen"] = now.isoformat() + "Z"
    except Exception:
        rec["count"] = 0
        rec["first_seen"] = now.isoformat() + "Z"

    if not rec.get("first_seen"):
        rec["first_seen"] = now.isoformat() + "Z"

    rec["count"] = int(rec.get("count", 0)) + 1
    rec["last_seen"] = now.isoformat() + "Z"

    shadow[ip] = rec
    _save_usage_counts(data)
    return rec["count"]


def _free_tries_remaining_after_this(effective_used_before: int) -> int:
    return max(0, FREE_TRIES - (effective_used_before + 1))


def _load_subscribers() -> Dict[str, Any]:
    data = _read_json_file(SUBSCRIBERS_PATH, {})
    return data if isinstance(data, dict) else {}


def _save_subscribers(data: Dict[str, Any]):
    _write_json_file_atomic(SUBSCRIBERS_PATH, data)


def _parse_iso_z(dt_str: str) -> Optional[datetime]:
    try:
        if not dt_str:
            return None
        return datetime.fromisoformat(str(dt_str).replace("Z", ""))
    except Exception:
        return None


def _set_buyer_session(email: str):
    email_n = _normalize_email(email)
    if not email_n:
        return
    session["subscriber_email"] = email_n
    session["premium"] = False
    session["buyer_set_at"] = datetime.utcnow().isoformat() + "Z"


def _mark_subscriber(email: str, is_active: bool, stripe_customer_id: str = ""):
    if not email:
        return

    email_n = _normalize_email(email)
    subs = _load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    rec["email"] = email_n
    rec["is_active"] = bool(is_active)
    if stripe_customer_id:
        rec["stripe_customer_id"] = stripe_customer_id
    rec["updated_at"] = datetime.utcnow().isoformat() + "Z"

    subs[email_n] = rec
    _save_subscribers(subs)


def _mark_dream_pack_purchase(email: str, uses: int = DREAM_PACK_USES, hours: int = DREAM_PACK_HOURS):
    if not email:
        return

    email_n = _normalize_email(email)
    subs = _load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    expires_at = datetime.utcnow() + timedelta(hours=hours)

    rec["email"] = email_n
    rec["dream_pack_uses_remaining"] = int(uses)
    rec["dream_pack_expires_at"] = expires_at.isoformat() + "Z"
    rec["updated_at"] = datetime.utcnow().isoformat() + "Z"

    if "is_active" not in rec:
        rec["is_active"] = False
    if "stripe_customer_id" not in rec:
        rec["stripe_customer_id"] = ""

    subs[email_n] = rec
    _save_subscribers(subs)


def _get_dream_pack_status(email: str) -> Dict[str, Any]:
    email_n = _normalize_email(email)
    if not email_n:
        return {"active": False, "uses_remaining": 0, "expires_at": ""}

    subs = _load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    try:
        uses_remaining = int(rec.get("dream_pack_uses_remaining", 0) or 0)
    except Exception:
        uses_remaining = 0

    expires_at = (rec.get("dream_pack_expires_at") or "").strip()
    expires_dt = _parse_iso_z(expires_at)

    active = False
    if uses_remaining > 0 and expires_dt and datetime.utcnow() < expires_dt:
        active = True

    return {
        "active": active,
        "uses_remaining": uses_remaining if active else 0,
        "expires_at": expires_at if active else "",
    }


def _consume_dream_pack_use(email: str) -> Dict[str, Any]:
    email_n = _normalize_email(email)
    if not email_n:
        return {"active": False, "uses_remaining": 0, "expires_at": ""}

    subs = _load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    current = _get_dream_pack_status(email_n)
    if not current["active"]:
        return current

    uses_remaining = max(0, int(current["uses_remaining"]) - 1)
    rec["dream_pack_uses_remaining"] = uses_remaining
    rec["dream_pack_expires_at"] = current["expires_at"]
    rec["updated_at"] = datetime.utcnow().isoformat() + "Z"

    subs[email_n] = rec
    _save_subscribers(subs)

    return {
        "active": uses_remaining > 0 and bool(current["expires_at"]),
        "uses_remaining": uses_remaining,
        "expires_at": current["expires_at"],
    }


# ============================================================
# Access helpers
# ============================================================
def _get_session_email() -> str:
    return _normalize_email(session.get("subscriber_email") or "")


def _is_premium_session() -> bool:
    return bool(session.get("premium") is True and _get_session_email())


def _set_premium_session(email: str):
    session["subscriber_email"] = _normalize_email(email)
    session["premium"] = True
    session["premium_set_at"] = datetime.utcnow().isoformat() + "Z"


def _clear_access_session():
    session.pop("subscriber_email", None)
    session.pop("premium", None)
    session.pop("premium_set_at", None)
    session.pop("buyer_set_at", None)


def _stripe_config_ok() -> bool:
    return bool(stripe and STRIPE_SECRET_KEY and DEFAULT_STRIPE_PRICE_ID)


def _stripe_dream_pack_ok() -> bool:
    return bool(stripe and STRIPE_SECRET_KEY and PRICE_DREAM_PACK)


def _stripe_active_subscription_for_email(email: str) -> Tuple[bool, str]:
    email = _normalize_email(email)
    if not _validate_email(email):
        return False, ""

    if not stripe or not STRIPE_SECRET_KEY:
        return False, ""

    stripe.api_key = STRIPE_SECRET_KEY

    try:
        customers = stripe.Customer.list(email=email, limit=10)
        for c in (customers.data or []):
            cid = c.get("id") or ""
            if not cid:
                continue

            subs = stripe.Subscription.list(customer=cid, status="all", limit=10)
            for s in (subs.data or []):
                status = (s.get("status") or "").lower()
                if status in {"active", "trialing"}:
                    return True, cid

        return False, ""
    except Exception as e:
        _log("Stripe check error:", str(e))
        return False, ""


def _has_active_access(email: str) -> Tuple[bool, Dict[str, Any]]:
    if not _validate_email(email):
        return False, {}

    is_active, customer_id = _stripe_active_subscription_for_email(email)
    if is_active:
        _mark_subscriber(email, True, customer_id)
        return True, {"type": "subscription", "customer_id": customer_id}

    pack = _get_dream_pack_status(email)
    if pack["active"]:
        return True, {"type": "dream_pack", "details": pack}

    return False, {}


# ============================================================
# Google Sheets helpers
# ============================================================
def _get_credentials() -> Credentials:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        raw_fixed = raw.replace("\\n", "\n")
        info = json.loads(raw_fixed)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _get_spreadsheet():
    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID env var is not set.")
    creds = _get_credentials()
    gc = gspread.authorize(creds)
    return gc.open_by_key(SPREADSHEET_ID)


def _worksheet_exists(sh, sheet_name: str) -> bool:
    try:
        sh.worksheet(sheet_name)
        return True
    except Exception:
        return False


def _get_or_create_worksheet(sheet_name: str, rows: int = 2000, cols: int = 30):
    sh = _get_spreadsheet()
    try:
        ws = sh.worksheet(sheet_name)
    except Exception:
        ws = sh.add_worksheet(title=sheet_name, rows=rows, cols=cols)
    _ensure_expected_headers(ws, sheet_name)
    return ws


def _ensure_expected_headers(ws, sheet_name: str):
    wanted = EXPECTED_HEADERS.get(sheet_name)
    if not wanted:
        return

    current = ws.row_values(1)
    current_norm = [_normalize_header(x) for x in current]
    wanted_norm = [_normalize_header(x) for x in wanted]

    if not current:
        end_col = chr(64 + min(len(wanted), 26))
        ws.update(f"A1:{end_col}1", [wanted], value_input_option="RAW")
        return

    if current_norm[:len(wanted_norm)] != wanted_norm:
        end_col = chr(64 + min(len(wanted), 26))
        ws.update(f"A1:{end_col}1", [wanted], value_input_option="RAW")


def _ensure_core_worksheets():
    sh = _get_spreadsheet()
    for name in list(EXPECTED_HEADERS.keys()):
        try:
            ws = sh.worksheet(name)
        except Exception:
            ws = sh.add_worksheet(title=name, rows=2000, cols=30)
        _ensure_expected_headers(ws, name)


def _worksheet_to_rows(ws) -> Tuple[List[str], List[Dict[str, str]]]:
    values = ws.get_all_values()
    if not values:
        return [], []

    raw_headers = values[0]
    headers = []
    seen = {}

    for h in raw_headers:
        nh = _normalize_header(h)
        if not nh:
            nh = "col"
        count = seen.get(nh, 0)
        seen[nh] = count + 1
        headers.append(nh if count == 0 else f"{nh}__{count + 1}")

    rows: List[Dict[str, str]] = []
    for r in values[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        rows.append({headers[i]: (r[i] or "").strip() for i in range(len(headers))})

    return headers, rows


def _invalidate_all_caches():
    _LEGACY_CACHE["loaded_at"] = 0.0
    _LEGACY_CACHE["rows"] = []
    _LEGACY_CACHE["headers"] = []
    _DOCTRINE_CACHE["loaded_at"] = 0.0
    _DOCTRINE_CACHE["sheets"] = {}
    _DOCTRINE_CACHE["headers"] = {}


def _load_legacy_sheet_rows(force: bool = False) -> List[Dict]:
    now = time.time()
    if (not force) and _LEGACY_CACHE["rows"] and (now - _LEGACY_CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _LEGACY_CACHE["rows"]

    ws = _get_or_create_worksheet(WORKSHEET_NAME)
    headers, rows = _worksheet_to_rows(ws)

    _LEGACY_CACHE["rows"] = rows
    _LEGACY_CACHE["headers"] = headers
    _LEGACY_CACHE["loaded_at"] = now
    return rows


def _load_doctrine_sheets(force: bool = False) -> Dict[str, List[Dict]]:
    now = time.time()
    if (not force) and _DOCTRINE_CACHE["sheets"] and (now - _DOCTRINE_CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _DOCTRINE_CACHE["sheets"]

    sheets_data: Dict[str, List[Dict]] = {}
    headers_map: Dict[str, List[str]] = {}

    for name in DOCTRINE_SHEET_NAMES:
        try:
            ws = _get_or_create_worksheet(name)
            headers, rows = _worksheet_to_rows(ws)
            sheets_data[name] = rows
            headers_map[name] = headers
        except Exception as e:
            _log(f"Doctrine sheet load failed for {name!r}: {e}")
            sheets_data[name] = []
            headers_map[name] = []

    _DOCTRINE_CACHE["sheets"] = sheets_data
    _DOCTRINE_CACHE["headers"] = headers_map
    _DOCTRINE_CACHE["loaded_at"] = now
    return sheets_data


def _doctrine_sheets_available() -> bool:
    try:
        sheets = _load_doctrine_sheets(force=False)
        return len(sheets.get(SHEET_BASE_SYMBOLS, [])) > 0
    except Exception:
        return False


# ============================================================
# Dream journal helpers
# ============================================================
def _get_journal_worksheet():
    return _get_or_create_worksheet(SHEET_DREAM_JOURNAL)


def _append_dream_journal_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    ws = _get_journal_worksheet()

    entry_id = payload.get("entry_id") or f"DJ-{secrets.token_hex(6).upper()}"
    created_at = datetime.utcnow().isoformat() + "Z"

    top_symbols = payload.get("top_symbols") or []
    if not isinstance(top_symbols, list):
        top_symbols = []

    row = [
        entry_id,
        created_at,
        _normalize_email(payload.get("email") or ""),
        (payload.get("dream_text") or "").strip(),
        (payload.get("spiritual_meaning") or "").strip(),
        (payload.get("effects_in_physical_realm") or "").strip(),
        (payload.get("what_to_do") or "").strip(),
        (payload.get("full_interpretation") or "").strip(),
        (payload.get("receipt_id") or "").strip(),
        ", ".join([str(x).strip() for x in top_symbols if str(x).strip()]),
        (payload.get("seal_status") or "").strip(),
        (payload.get("seal_type") or "").strip(),
        (payload.get("seal_risk") or "").strip(),
        (payload.get("engine_mode") or "").strip(),
        (payload.get("access_type") or "").strip(),
        (payload.get("is_saved") or "yes").strip(),
        (payload.get("notes") or "").strip(),
    ]

    ws.append_row(row, value_input_option="RAW")
    return {"ok": True, "entry_id": entry_id, "created_at": created_at}


# ============================================================
# Field getters
# ============================================================
def _get_symbol_cell(row: Dict) -> str:
    return _row_get(row, "input", "symbol", "symbols", "input symbol", "dream symbol", "symbol name")


def _get_spiritual_meaning_cell(row: Dict) -> str:
    return _row_get(
        row,
        "spiritual meaning", "spiritual_meaning", "spiritual",
        "meaning", "base spiritual meaning", "base_spiritual_meaning",
        "final spiritual meaning", "final_spiritual_meaning",
    )


def _get_effects_cell(row: Dict) -> str:
    return _row_get(
        row,
        "effects in the physical realm", "effects_in_the_physical_realm",
        "physical effects", "physical_effects",
        "effects", "base physical effects", "base_physical_effects",
        "final physical effects", "final_physical_effects",
    )


def _get_what_to_do_cell(row: Dict) -> str:
    return _row_get(
        row,
        "what to do", "what_to_do", "action", "actions",
        "base action", "base_action", "final action", "final_action",
    )


def _get_keywords_cell(row: Dict) -> str:
    return _row_get(row, "keywords", "keyword", "tags")


def _row_is_active(row: Dict) -> bool:
    active = _row_get(row, "active")
    if not active:
        return True
    return _normalize_yes_no(active)


def _get_base_symbol_input(row: Dict) -> str:
    return _row_get(row, "symbol", "input")


def _get_base_symbol_category(row: Dict) -> str:
    return _row_get(row, "category") or "unknown"


def _get_base_symbol_meaning(row: Dict) -> str:
    return _row_get(
        row,
        "base_spiritual_meaning", "base spiritual meaning",
        "spiritual meaning", "meaning",
    )


def _get_base_symbol_effects(row: Dict) -> str:
    return _row_get(
        row,
        "base_physical_effects", "base physical effects",
        "physical effects", "effects",
    )


def _get_base_symbol_action(row: Dict) -> str:
    return _row_get(row, "base_action", "base action", "action", "actions")


def _get_rule_name(row: Dict, *keys: str) -> str:
    return _row_get(row, *keys)


def _get_rule_keywords(row: Dict) -> List[str]:
    return _split_keywords(_row_get(row, "keywords", "keyword", "tags"))


def _get_behavior_name(row: Dict) -> str:
    return _get_rule_name(row, "behavior_name", "behavior")


def _get_behavior_meaning_modifier(row: Dict) -> str:
    return _row_get(row, "meaning_modifier", "meaning modifier", "effect", "effects")


def _get_behavior_physical_modifier(row: Dict) -> str:
    return _row_get(row, "physical_modifier", "physical modifier", "physical_effect", "physical effects", "effect", "effects")


def _get_behavior_action_modifier(row: Dict) -> str:
    return _row_get(row, "action_modifier", "action modifier", "action", "actions")


def _get_state_name(row: Dict) -> str:
    return _get_rule_name(row, "state_name", "state")


def _get_state_meaning_modifier(row: Dict) -> str:
    return _row_get(row, "meaning_modifier", "meaning modifier", "effect", "effects")


def _get_state_physical_modifier(row: Dict) -> str:
    return _row_get(row, "physical_modifier", "physical modifier", "physical_effect", "physical effects", "effect", "effects")


def _get_state_action_modifier(row: Dict) -> str:
    return _row_get(row, "action_modifier", "action modifier", "action", "actions")


def _get_location_name(row: Dict) -> str:
    return _get_rule_name(row, "location_name", "location")


def _get_location_life_area_meaning(row: Dict) -> str:
    return _row_get(row, "life_area_meaning", "life area meaning", "effect", "effects")


def _get_location_physical_area_meaning(row: Dict) -> str:
    return _row_get(row, "physical_area_meaning", "physical area meaning", "effect", "effects")


def _get_location_action_modifier(row: Dict) -> str:
    return _row_get(row, "action_modifier", "action modifier", "action", "actions")


def _get_relationship_name(row: Dict) -> str:
    return _get_rule_name(row, "relationship_name", "relationship")


def _get_relationship_meaning_modifier(row: Dict) -> str:
    return _row_get(row, "meaning_modifier", "meaning modifier", "effect", "effects")


def _get_relationship_physical_modifier(row: Dict) -> str:
    return _row_get(row, "physical_modifier", "physical modifier", "physical_effect", "physical effects", "effect", "effects")


def _get_relationship_action_modifier(row: Dict) -> str:
    return _row_get(row, "action_modifier", "action modifier", "action", "actions")


def _get_output_template(rows: List[Dict], template_type: str, fallback: str) -> str:
    wanted = _normalize_text(template_type)
    for row in rows:
        if not _row_is_active(row):
            continue
        if _normalize_text(_row_get(row, "template_type")) == wanted:
            txt = _row_get(row, "template_text")
            if txt:
                return txt
    return fallback


# ============================================================
# Matching and interpretation logic
# ============================================================
def _symbol_length_penalty(symbol: str) -> int:
    words = [w for w in _normalize_text(symbol).split() if w]
    n = len(words)
    if n <= 2:
        return 0
    if n == 3:
        return 2
    if n == 4:
        return 6
    if n == 5:
        return 12
    return 18


def _score_row_strict(
    dream_norm: str,
    row: Dict,
    used_spans: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    symbol_raw = _get_symbol_cell(row)
    if not symbol_raw:
        return 0, None

    symbol = _normalize_text(symbol_raw)
    if not symbol:
        return 0, None

    keywords = _split_keywords(_get_keywords_cell(row))
    candidates: List[Tuple[str, str, int]] = []

    if symbol:
        if len(_tokenize_words(symbol)) > 1:
            candidates.append(("symbol_phrase", symbol, 100 - _symbol_length_penalty(symbol_raw)))
        else:
            candidates.append(("symbol", symbol, 100))

    for kw in keywords:
        if len(_tokenize_words(kw)) > 1:
            candidates.append(("keyword_phrase", kw, 94))
        else:
            candidates.append(("keyword", kw, 90))

    candidates.sort(key=lambda x: (-len(x[1]), -x[2]))

    for match_type, token, score in candidates:
        spans = _find_phrase_spans(dream_norm, token)
        if not spans:
            continue

        chosen_span = None
        for sp in spans:
            if not used_spans or not _is_span_blocked(sp, used_spans):
                chosen_span = sp
                break

        if chosen_span:
            return score, {
                "type": match_type,
                "token": token,
                "span": chosen_span,
                "token_len": len(token),
            }

    return 0, None


def _match_symbols_legacy(
    dream: str,
    rows: List[Dict],
    top_k: int = 3,
) -> List[Tuple[Dict, int, Optional[Dict[str, Any]]]]:
    dream_norm = _normalize_text(dream)
    if not dream_norm:
        return []

    candidates: List[Tuple[Dict, int, Optional[Dict[str, Any]]]] = []
    for row in rows:
        if not _row_is_active(row):
            continue
        sc, hit = _score_row_strict(dream_norm, row, used_spans=[])
        if sc > 0 and hit:
            candidates.append((row, sc, hit))

    def _sort_key(item: Tuple[Dict, int, Optional[Dict[str, Any]]]):
        row, sc, hit = item
        sym = _get_symbol_cell(row)
        token_len = int((hit or {}).get("token_len", 0))
        sym_len = len(_normalize_text(sym))
        return (-token_len, -sc, -sym_len)

    candidates.sort(key=_sort_key)

    used_spans: List[Tuple[int, int]] = []
    seen_symbols = set()
    out: List[Tuple[Dict, int, Optional[Dict[str, Any]]]] = []

    for row, sc, hit in candidates:
        if not hit:
            continue
        span = hit.get("span")
        if not span:
            continue

        sym = _get_symbol_cell(row)
        sym_key = _normalize_text(sym)
        if not sym_key or sym_key in seen_symbols:
            continue
        if _is_span_blocked(span, used_spans):
            continue

        used_spans.append(span)
        seen_symbols.add(sym_key)
        out.append((row, sc, hit))

        if len(out) >= top_k:
            break

    return out


def _category_priority(category: str) -> int:
    c = _normalize_category(category)
    priorities = {
        "ending": 40,
        "death": 36,
        "graveyard": 35,
        "body": 28,
        "person": 26,
        "animal": 24,
        "water": 22,
        "movement": 20,
        "nature": 18,
        "object": 16,
        "location": 14,
        "emotion": 12,
    }
    return priorities.get(c, 10)


def _score_base_candidate(row: Dict, match_type: str, symbol_raw: str) -> int:
    if match_type == "symbol_phrase":
        base = 120
    elif match_type == "symbol":
        base = 112
    elif match_type == "keyword_phrase":
        base = 104
    else:
        base = 96

    category_bonus = _category_priority(_get_base_symbol_category(row))
    length_penalty = _symbol_length_penalty(symbol_raw)
    return base + category_bonus - length_penalty


def _category_conflict_penalty(
    row: Dict,
    already_selected: List[Tuple[Dict, int, Dict[str, Any]]],
) -> int:
    current_cat = _normalize_category(_get_base_symbol_category(row))
    if current_cat == "unknown":
        return 0

    penalty = 0
    for existing_row, _sc, _hit in already_selected:
        ex_cat = _normalize_category(_get_base_symbol_category(existing_row))
        if ex_cat == current_cat:
            penalty += 6
    return penalty


def _ending_bonus_for_symbol(row: Dict, ending_text: str) -> int:
    symbol = _get_base_symbol_input(row)
    if not symbol:
        return 0

    ending_norm = _normalize_text(ending_text)
    symbol_n = _normalize_text(symbol)
    if symbol_n and _contains_phrase(ending_norm, symbol_n):
        return 18

    for kw in _get_rule_keywords(row):
        if _contains_phrase(ending_norm, kw):
            return 10

    return 0


def _dedupe_rule_hits_by_best_row(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_name: Dict[str, Dict[str, Any]] = {}

    for hit in hits:
        key = _normalize_text(hit.get("name", ""))
        if not key:
            continue

        existing = best_by_name.get(key)
        if not existing:
            best_by_name[key] = hit
            continue

        existing_tuple = (existing.get("token_len", 0), existing.get("priority", 0))
        current_tuple = (hit.get("token_len", 0), hit.get("priority", 0))
        if current_tuple > existing_tuple:
            best_by_name[key] = hit

    out = list(best_by_name.values())
    out.sort(key=lambda x: (-x["token_len"], -x["priority"], x["name"].lower()))
    return out


def _select_non_conflicting_rule_hits(
    hits: List[Dict[str, Any]],
    max_hits: int = MAX_RULE_HITS_PER_LAYER,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_tokens = set()

    for hit in hits:
        token = _normalize_text(hit.get("matched_token", ""))
        if token and token in seen_tokens:
            continue
        selected.append(hit)
        if token:
            seen_tokens.add(token)
        if len(selected) >= max_hits:
            break

    return selected


def _match_base_symbols_doctrine(
    dream: str,
    base_rows: List[Dict],
    top_k: int = 3,
    behaviors: Optional[List[Dict[str, Any]]] = None,
    states: Optional[List[Dict[str, Any]]] = None,
    locations: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[Dict, int, Dict[str, Any]]]:
    dream_norm = _normalize_text(dream)
    ending_text = _extract_dream_ending_text(dream)

    if not dream_norm:
        return []

    behaviors = behaviors or []
    states = states or []
    locations = locations or []
    relationships = relationships or []

    scenario_score = _scenario_signal_score(behaviors, states, locations, relationships)
    location_rule_names = _rule_names_set(locations)
    relationship_rule_names = _rule_names_set(relationships)

    candidates: List[Tuple[Dict, int, Dict[str, Any]]] = []

    for row in base_rows:
        if not _row_is_active(row):
            continue

        symbol_raw = _get_base_symbol_input(row)
        if not symbol_raw:
            continue

        symbol = _normalize_text(symbol_raw)
        if not symbol:
            continue

        keywords = _get_rule_keywords(row)
        local_candidates: List[Tuple[str, str]] = []

        if len(_tokenize_words(symbol)) > 1:
            local_candidates.append(("symbol_phrase", symbol))
        else:
            local_candidates.append(("symbol", symbol))

        for kw in keywords:
            if len(_tokenize_words(kw)) > 1:
                local_candidates.append(("keyword_phrase", kw))
            else:
                local_candidates.append(("keyword", kw))

        local_candidates.sort(key=lambda x: -len(x[1]))

        for match_type, token in local_candidates:
            spans = _find_phrase_spans(dream_norm, token)
            if not spans:
                continue

            ending_bonus = _ending_bonus_for_symbol(row, ending_text)
            score = _score_base_candidate(row, match_type, symbol_raw) + ending_bonus

            cat = _normalize_category(_get_base_symbol_category(row))
            if scenario_score >= 10 and cat in {"location", "object"}:
                score -= 14
            if scenario_score >= 10 and symbol in location_rule_names:
                score -= 18
            if scenario_score >= 8 and symbol in relationship_rule_names:
                score -= 8

            candidates.append((row, score, {
                "type": match_type,
                "token": token,
                "span": spans[0],
                "token_len": len(token),
                "ending_bonus": ending_bonus,
            }))
            break

    candidates.sort(
        key=lambda item: (
            -item[1],
            -int(item[2].get("token_len", 0)),
            -_category_priority(_get_base_symbol_category(item[0])),
            -len(_normalize_text(_get_base_symbol_input(item[0]))),
        )
    )

    selected: List[Tuple[Dict, int, Dict[str, Any]]] = []
    used_spans: List[Tuple[int, int]] = []
    seen_symbols = set()
    effective_top_k = 2 if scenario_score >= 10 else top_k

    for row, score, hit in candidates:
        span = hit.get("span")
        if not span:
            continue

        sym_key = _normalize_text(_get_base_symbol_input(row))
        if not sym_key or sym_key in seen_symbols:
            continue
        if _is_span_blocked(span, used_spans):
            continue

        score_after_conflict = score - _category_conflict_penalty(row, selected)
        selected.append((row, score_after_conflict, hit))
        used_spans.append(span)
        seen_symbols.add(sym_key)

        if len(selected) >= effective_top_k:
            break

    selected.sort(key=lambda item: -item[1])
    return selected



def _detect_rule_hits(
    dream: str,
    rows: List[Dict],
    kind: str,
    priority_field: str = "priority",
) -> List[Dict[str, Any]]:
    dream_norm = _normalize_text(dream)
    ending_text = _extract_dream_ending_text(dream)
    ending_norm = _normalize_text(ending_text)

    hits: List[Dict[str, Any]] = []

    for row in rows:
        if not _row_is_active(row):
            continue

        if kind == "behavior":
            name = _get_behavior_name(row)
        elif kind == "state":
            name = _get_state_name(row)
        elif kind == "location":
            name = _get_location_name(row)
        elif kind == "relationship":
            name = _get_relationship_name(row)
        else:
            name = ""

        if not name:
            continue

        rule_keywords = _get_rule_keywords(row)
        if not rule_keywords:
            name_fallback = _normalize_text(name)
            if name_fallback:
                rule_keywords = [name_fallback]
            else:
                continue

        matched_token = ""
        token_len = 0
        matched_in_ending = False

        for kw in sorted(rule_keywords, key=len, reverse=True):
            if _contains_phrase(dream_norm, kw):
                matched_token = kw
                token_len = len(kw)
                matched_in_ending = _contains_phrase(ending_norm, kw)
                break

        if not matched_token:
            continue

        try:
            priority_val = int(str(_row_get(row, priority_field) or "0").strip())
        except Exception:
            priority_val = 0

        if matched_in_ending:
            priority_val += 3

        hits.append({
            "name": name,
            "row": row,
            "matched_token": matched_token,
            "token_len": token_len,
            "priority": priority_val,
            "kind": kind,
            "matched_in_ending": matched_in_ending,
        })

    hits = _dedupe_rule_hits_by_best_row(hits)
    hits = _select_non_conflicting_rule_hits(hits, MAX_RULE_HITS_PER_LAYER)
    return hits


def _override_context(
    dream: str,
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Dict[str, Any]:
    symbol_names = [
        _normalize_text(_get_base_symbol_input(row))
        for row, _sc, _hit in base_matches
        if _get_base_symbol_input(row)
    ]
    categories = [
        _normalize_category(_get_base_symbol_category(row))
        for row, _sc, _hit in base_matches
    ]
    behavior_names = [_normalize_text(x["name"]) for x in behaviors if x.get("name")]
    state_names = [_normalize_text(x["name"]) for x in states if x.get("name")]
    location_names = [_normalize_text(x["name"]) for x in locations if x.get("name")]
    relationship_names = [_normalize_text(x["name"]) for x in relationships if x.get("name")]

    return {
        "symbol": set(symbol_names),
        "category": set(categories),
        "behavior": set(behavior_names),
        "state": set(state_names),
        "location": set(location_names),
        "relationship": set(relationship_names),
        "text": _normalize_text(dream),
        "ending": _normalize_text(_extract_dream_ending_text(dream)),
    }


def _parse_override_groups(condition: str) -> List[List[str]]:
    groups = []
    for raw_group in re.split(r"\|\|", condition or ""):
        tokens = [
            _clean_sentence(x).strip()
            for x in re.split(r"\+", raw_group)
            if _clean_sentence(x).strip()
        ]
        if tokens:
            groups.append(tokens)
    return groups


def _token_matches_context(token: str, ctx: Dict[str, Any]) -> Tuple[bool, int]:
    raw = (token or "").strip()
    if not raw:
        return False, 0

    negative = raw.startswith("!")
    if negative:
        raw = raw[1:].strip()

    raw_norm = _normalize_text(raw)

    field = ""
    value = ""
    op = ""

    if "!=" in raw:
        field, value = raw.split("!=", 1)
        op = "!="
    elif "=" in raw:
        field, value = raw.split("=", 1)
        op = "="

    if field and op:
        field_n = _normalize_text(field)
        value_n = _normalize_text(value)

        if field_n in {"symbol", "category", "behavior", "state", "location", "relationship"}:
            exists = value_n in ctx.get(field_n, set())
        elif field_n in {"text", "ending"}:
            exists = _contains_phrase(ctx.get(field_n, ""), value_n)
        else:
            exists = False

        matched = (not exists) if op == "!=" else exists
        if negative:
            matched = not matched

        specificity = 5 if field_n in {"symbol", "category", "behavior", "state", "location", "relationship"} else 3
        return matched, specificity

    exists_plain = False
    for field_name in ["symbol", "category", "behavior", "state", "location", "relationship"]:
        if raw_norm in ctx.get(field_name, set()):
            exists_plain = True
            break

    if not exists_plain:
        if _contains_phrase(ctx.get("text", ""), raw_norm) or _contains_phrase(ctx.get("ending", ""), raw_norm):
            exists_plain = True

    matched_plain = not exists_plain if negative else exists_plain
    return matched_plain, 2


def _apply_override_rules(
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    dream: str,
    override_rows: List[Dict],
) -> Optional[Dict[str, Any]]:
    ctx = _override_context(dream, base_matches, behaviors, states, locations, relationships)

    best: Optional[Dict[str, Any]] = None
    best_score = -10**9

    for row in override_rows:
        if not _row_is_active(row):
            continue

        condition = _row_get(row, "condition")
        if not condition:
            continue

        groups = _parse_override_groups(condition)
        if not groups:
            continue

        matched_any_group = False
        group_score = -10**9

        for group in groups:
            all_ok = True
            specificity = 0

            for token in group:
                ok, spec = _token_matches_context(token, ctx)
                if not ok:
                    all_ok = False
                    break
                specificity += spec

            if all_ok:
                matched_any_group = True
                group_score = max(group_score, specificity + len(group))

        if not matched_any_group:
            continue

        try:
            priority = int(str(_row_get(row, "priority") or "0").strip())
        except Exception:
            priority = 0

        score = (priority * 100) + group_score

        if score > best_score:
            best_score = score
            best = row

    if not best:
        return None

    try:
        pr = int(str(_row_get(best, "priority") or "0").strip())
    except Exception:
        pr = 0

    return {
        "override_name": _row_get(best, "override_name", "condition"),
        "spiritual": _row_get(best, "final_spiritual_meaning", "override_effect", "effect", "effects"),
        "physical": _row_get(best, "final_physical_effects"),
        "action": _row_get(best, "final_action"),
        "priority": pr,
        "row": best,
        "condition": _row_get(best, "condition"),
    }


def _compute_doctrine_seal(
    dream: str,
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ending_text = _extract_dream_ending_text(dream)
    ending_norm = _normalize_text(ending_text)

    behavior_names = {_normalize_text(x["name"]) for x in behaviors}
    state_names = {_normalize_text(x["name"]) for x in states}
    location_names = {_normalize_text(x["name"]) for x in locations}
    relationship_names = {_normalize_text(x["name"]) for x in relationships}

    ending_hits = []
    for row, _sc, _hit in base_matches:
        symbol = _get_base_symbol_input(row)
        if symbol and _contains_phrase(ending_norm, symbol):
            ending_hits.append(symbol)

    status = "Processing"
    seal_type = "Layered"
    risk = "Medium"
    message = "The ending suggests the dream should be taken seriously in prayer."

    override_name = _normalize_text((override_hit or {}).get("override_name", ""))
    override_spiritual = _normalize_text((override_hit or {}).get("spiritual", ""))

    if override_hit and any(x in f"{override_name} {override_spiritual}" for x in ["death", "grave", "death omen"]):
        status = "Sealed"
        seal_type = "Death Omen"
        risk = "High"
        message = "The ending and override logic point to a sealed warning."
    elif {"escaping", "crossing", "finding"} & behavior_names:
        status = "Open"
        seal_type = "Breakthrough"
        risk = "Low"
        message = "The ending shows movement toward escape, transition, or release."
    elif {"being attacked", "being bitten", "being chased", "fighting"} & behavior_names:
        status = "Contested"
        seal_type = "Warfare"
        risk = "High"
        message = "The ending shows active contention around the message."
    elif {"dirty", "murky", "broken", "bleeding", "dark"} & state_names:
        status = "Warning"
        seal_type = "Corrupted"
        risk = "High"
        message = "The ending carries warning signs of contamination, confusion, or damage."
    elif {"graveyard", "prison", "darkness"} & location_names:
        status = "Warning"
        seal_type = "Bound"
        risk = "High"
        message = "The ending places the message in a heavy or restrictive spiritual environment."
    elif {"mother", "father", "child", "spouse", "friend"} & relationship_names:
        status = "Focused"
        seal_type = "Relational"
        risk = "Medium"
        message = "The dream ending points toward a relationship-centered message."
    elif ending_hits:
        status = "Sealed"
        seal_type = "Symbol Confirmed"
        risk = "Medium"
        message = "The ending repeated or confirmed a major symbol from the dream."
    elif len(base_matches) == 1:
        status = "Live"
        seal_type = "Focused"
        risk = "Low"
        message = "The dream is centered and narrow, with one main symbolic message."
    elif len(base_matches) >= 3:
        status = "Layered"
        seal_type = "Complex"
        risk = "Medium"
        message = "The dream is layered and should be read carefully, not rushed."

    return {
        "status": status,
        "type": seal_type,
        "risk": risk,
        "message": message,
        "ending_text": ending_text,
        "ending_hits": ending_hits,
    }


def _choose_template_type(
    override_hit: Optional[Dict[str, Any]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    interpretation: Dict[str, str],
    seal: Dict[str, Any],
) -> str:
    if override_hit:
        name = _normalize_text(override_hit.get("override_name", ""))
        spiritual = _normalize_text(override_hit.get("spiritual", ""))
        combined = f"{name} {spiritual}"

        if any(x in combined for x in ["death", "grave", "teeth falling out", "death omen"]):
            return "death_omen"
        if any(x in combined for x in ["monitor", "watch", "spy"]):
            return "monitoring"
        if any(x in combined for x in ["escape", "deliver", "freedom", "victory"]):
            return "deliverance"
        if any(x in combined for x in ["clean", "wash", "purif", "release"]):
            return "cleansing"
        if any(x in combined for x in ["promotion", "elevation", "lifted", "favor"]):
            return "promotion"
        if any(x in combined for x in ["warning", "danger", "attack", "disgrace"]):
            return "warning"

    behavior_names = {_normalize_text(x["name"]) for x in behaviors}
    state_names = {_normalize_text(x["name"]) for x in states}
    location_names = {_normalize_text(x["name"]) for x in locations}
    relationship_names = {_normalize_text(x["name"]) for x in relationships}
    interp_text = _normalize_text(" ".join(interpretation.values()))
    seal_type = _normalize_text(seal.get("type", ""))

    if "death omen" in seal_type:
        return "death_omen"
    if "protection" in seal_type or "discernment" in seal_type:
        return "warning"
    if {"being attacked", "being chased", "fighting"} & behavior_names:
        return "warfare"
    if {"escaping", "crossing", "finding"} & behavior_names:
        return "breakthrough"
    if {"dirty", "murky", "broken", "bleeding", "dark"} & state_names:
        return "warning"
    if {"graveyard", "prison", "darkness"} & location_names:
        return "warning"
    if relationship_names:
        return "relational"
    if any(x in interp_text for x in ["monitoring spirit", "being watched", "spiritual spy"]):
        return "monitoring"
    if any(x in interp_text for x in ["cleansing", "washed", "release"]):
        return "cleansing"

    return "default"


# ============================================================
# Interpretation builders
# ============================================================
def _summarize_symbol_meanings(base_matches: List[Tuple[Dict, int, Dict[str, Any]]]) -> str:
    items = []

    for row, _score, _hit in base_matches[:NARRATIVE_MAX_SYMBOLS]:
        symbol = _strip_trailing_punct(_get_base_symbol_input(row))
        meaning = _strip_trailing_punct(_get_base_symbol_meaning(row))

        if symbol and meaning:
            items.append(f"{symbol} points to {meaning}")
        elif symbol:
            items.append(symbol)

    joined = _human_join(items)
    return _sentence(joined) if joined else ""


def _build_core_message(
    dream: str,
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
    seal: Dict[str, Any],
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    focus = _build_primary_focus(dream, base_matches, behaviors, states, locations, relationships, override_hit, seal)
    event_scenario = _build_event_scenario(dream, behaviors, states, locations, relationships)
    seal_type = _strip_trailing_punct(seal.get("type", ""))
    seal_message = _strip_trailing_punct(seal.get("message", ""))

    parts = []
    if event_scenario.get("lead") and _event_layer_strength(dream, states) >= 12:
        parts.append(_sentence(f"This dream points to {event_scenario['lead']}"))
    elif focus.get("lead"):
        parts.append(_sentence(f"This dream points to {focus['lead']}"))

    if event_scenario.get("support"):
        parts.append(_sentence(event_scenario["support"]))

    if seal_type:
        parts.append(_sentence(f"The ending confirms this as {seal_type.lower()}"))

    if _dream_has_escape_cue(dream) and not any("release" in _normalize_text(p) or "escape" in _normalize_text(p) or "came out" in _normalize_text(p) for p in parts):
        parts.append(_sentence("The ending shows that you came out of the danger, so this is a warning with a path of escape"))

    if seal_message and _normalize_text(seal_message) not in _normalize_text(" ".join(parts)):
        parts.append(_sentence(seal_message))

    return _merge_natural_paragraphs(parts), focus, event_scenario


def _build_layered_support_paragraph(
    dream: str,
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    focus: Optional[Dict[str, str]] = None,
    event_scenario: Optional[Dict[str, str]] = None,
) -> str:
    focus = focus or {}
    event_scenario = event_scenario or {}
    behavior_parts = _compact_rule_meaning_clause(behaviors, _get_behavior_meaning_modifier, max_items=2)
    state_parts = _compact_rule_meaning_clause(states, _get_state_meaning_modifier, max_items=1)
    location_parts = _compact_rule_meaning_clause(locations, _get_location_life_area_meaning, max_items=1)
    relationship_parts = _compact_rule_meaning_clause(relationships, _get_relationship_meaning_modifier, max_items=1)

    lines = []
    if event_scenario.get("support"):
        lines.append(_sentence(event_scenario["support"]))

    if behavior_parts and _normalize_text(behavior_parts) != _normalize_text(focus.get("behavior", "")):
        lines.append(_sentence(f"The dream actions suggest {behavior_parts}"))
    if _detect_transition_instability(dream, states):
        lines.append(_sentence("What appeared stable changed suddenly, so the dream is warning about instability hidden beneath a calm surface"))
    elif state_parts and _normalize_text(state_parts) != _normalize_text(focus.get("state", "")):
        lines.append(_sentence(f"The condition of what appeared points to {state_parts}"))
    if location_parts and _normalize_text(location_parts) != _normalize_text(focus.get("location", "")):
        lines.append(_sentence(f"The setting connects this message to {location_parts}"))
    if relationship_parts and _normalize_text(relationship_parts) != _normalize_text(focus.get("relationship", "")):
        lines.append(_sentence(f"The people involved point to {relationship_parts}"))

    return _merge_natural_paragraphs(lines)


def _build_real_world_impact_paragraph(
    dream: str,
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
    focus: Optional[Dict[str, str]] = None,
    event_scenario: Optional[Dict[str, str]] = None,
) -> str:
    effect_parts = []
    event_scenario = event_scenario or {}

    scenario_score = _scenario_signal_score(behaviors, states, locations, relationships, dream_text=dream)
    base_limit = 1 if scenario_score >= 10 else NARRATIVE_MAX_SYMBOLS

    if event_scenario.get("effects"):
        effect_parts.append(event_scenario["effects"])

    for row, _score, _hit in base_matches[:base_limit]:
        effect_parts.append(_get_base_symbol_effects(row))

    effect_parts.extend([_get_behavior_physical_modifier(x["row"]) for x in behaviors])
    effect_parts.extend([_get_state_physical_modifier(x["row"]) for x in states])
    effect_parts.extend([_get_location_physical_area_meaning(x["row"]) for x in locations])
    effect_parts.extend([_get_relationship_physical_modifier(x["row"]) for x in relationships])

    if override_hit:
        effect_parts.append((override_hit or {}).get("physical", ""))

    compressed = _compress_effect_phrases(effect_parts, max_items=3 if scenario_score >= 10 else 4)

    if not compressed:
        return "No clear physical effects were generated."

    return _sentence(f"In practical terms, this may show up as {_human_join(compressed)}")


def _build_action_guidance_paragraph(
    dream: str,
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
    event_scenario: Optional[Dict[str, str]] = None,
) -> str:
    action_parts = []
    event_scenario = event_scenario or {}

    scenario_score = _scenario_signal_score(behaviors, states, locations, relationships, dream_text=dream)
    base_limit = 1 if scenario_score >= 10 else NARRATIVE_MAX_SYMBOLS

    if event_scenario.get("action"):
        action_parts.append(event_scenario["action"])

    for row, _score, _hit in base_matches[:base_limit]:
        action_parts.append(_get_base_symbol_action(row))

    action_parts.extend([_get_behavior_action_modifier(x["row"]) for x in behaviors])
    action_parts.extend([_get_state_action_modifier(x["row"]) for x in states])
    action_parts.extend([_get_location_action_modifier(x["row"]) for x in locations])
    action_parts.extend([_get_relationship_action_modifier(x["row"]) for x in relationships])

    if override_hit:
        action_parts.append((override_hit or {}).get("action", ""))

    compressed = _compress_action_phrases(action_parts, max_items=2 if scenario_score >= 10 else 3)

    if not compressed:
        return "Pray for wisdom and confirmation."

    if len(compressed) == 1:
        return _sentence(compressed[0])

    return _merge_natural_paragraphs([
        _sentence(compressed[0]),
        _sentence(compressed[1]),
    ])


def _build_final_summary_paragraph(
    interpretation: Dict[str, str],
    seal: Dict[str, Any],
) -> str:
    seal_type = _strip_trailing_punct(seal.get("type", ""))
    risk = _strip_trailing_punct(seal.get("risk", ""))

    parts = []
    if seal_type and risk:
        parts.append(_sentence(f"Overall, this is a {seal_type.lower()} message with {risk.lower()} risk"))

    parts.append(_sentence(
        "Do not react in panic. Take the message seriously, stay spiritually grounded, and respond with discipline"
    ))

    return _merge_natural_paragraphs(parts)


def _render_template_text(template_text: str, context: Dict[str, str]) -> str:
    text = template_text or ""
    for key, value in context.items():
        text = text.replace("{" + key + "}", _strip_trailing_punct(value or ""))
    return _sentence(text)


def _build_doctrine_interpretation(
    dream: str,
    base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
    templates: List[Dict],
    seal: Dict[str, Any],
) -> Dict[str, Any]:
    top_symbols = [_get_base_symbol_input(row) for row, _sc, _hit in base_matches]

    core_message, focus = _build_core_message(base_matches, behaviors, states, locations, relationships, override_hit, seal)
    support_message = _build_layered_support_paragraph(behaviors, states, locations, relationships, focus)
    physical_message = _build_real_world_impact_paragraph(base_matches, behaviors, states, locations, relationships, override_hit, focus)
    action_message = _build_action_guidance_paragraph(base_matches, behaviors, states, locations, relationships, override_hit)
    summary_message = _build_final_summary_paragraph(
        {
            "spiritual_meaning": core_message,
            "effects_in_physical_realm": physical_message,
            "what_to_do": action_message,
        },
        seal,
    )

    interpretation = {
        "spiritual_meaning": core_message or "No clear spiritual meaning was generated.",
        "effects_in_physical_realm": physical_message or "No clear physical effects were generated.",
        "what_to_do": action_message or "Pray for wisdom and confirmation.",
    }

    template_type = _choose_template_type(
        override_hit, behaviors, states, locations, relationships, interpretation, seal
    )

    opening_tpl = _get_output_template(
        templates,
        "opening",
        "This dream is revealing a spiritual condition that requires discernment, not panic.",
    )
    closing_tpl = _get_output_template(
        templates,
        "closing",
        "Dreams expose what needs attention so you can respond with wisdom, prayer, and obedience.",
    )
    main_tpl = _get_output_template(
        templates,
        template_type,
        "{symbol} carries the central meaning of {meaning}. The dream actions support {behavior_effect}. "
        "The condition of what appeared points to {state_effect}. The setting connects this to {location_effect}. "
        "The people involved point to {relationship_effect}. The ending seals the dream as {seal_type}.",
    )

    compact_symbol = ", ".join([x for x in top_symbols[:2] if x]).strip() or "This dream"
    compact_meaning = focus.get("lead") or _compact_spiritual_meaning(base_matches, override_hit, seal) or "the main spiritual message here"
    compact_behavior = focus.get("behavior") or _compact_rule_meaning_clause(behaviors, _get_behavior_meaning_modifier, max_items=2) or "the active pattern in the dream"
    compact_state = focus.get("state") or _compact_rule_meaning_clause(states, _get_state_meaning_modifier, max_items=1) or "the condition attached to the message"
    compact_location = focus.get("location") or _compact_rule_meaning_clause(locations, _get_location_life_area_meaning, max_items=1) or "the area of life being touched"
    compact_relationship = focus.get("relationship") or _compact_rule_meaning_clause(relationships, _get_relationship_meaning_modifier, max_items=1) or "the people dimension of the dream"

    context = {
        "symbol": compact_symbol,
        "meaning": compact_meaning,
        "behavior_effect": compact_behavior,
        "state_effect": compact_state,
        "location_effect": compact_location,
        "relationship_effect": compact_relationship,
        "seal_type": seal.get("type", "a layered message"),
    }

    rendered_main = _render_template_text(main_tpl, context)

    full_parts = [
        _sentence(opening_tpl),
        rendered_main,
        support_message,
        interpretation["effects_in_physical_realm"],
        interpretation["what_to_do"],
        summary_message,
        _sentence(closing_tpl),
    ]

    TEMP

    return {
        "interpretation": interpretation,
        "full_interpretation": full_interpretation,
        "top_symbols": top_symbols,
        "override_applied": bool(override_hit),
        "override_name": (override_hit or {}).get("override_name", ""),
        "template_type": template_type,
    }



def _build_legacy_interpretation(matches: List[Tuple[Dict, int, Optional[Dict[str, Any]]]]) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, _sc, _hit in matches[:NARRATIVE_MAX_SYMBOLS]:
        symbol = _strip_trailing_punct(_get_symbol_cell(row))
        base_meaning = _strip_trailing_punct(_get_spiritual_meaning_cell(row))
        base_effects = _strip_trailing_punct(_get_effects_cell(row))
        base_action = _strip_trailing_punct(_get_what_to_do_cell(row))

        if symbol and base_meaning:
            spiritual_parts.append(f"{symbol} points to {base_meaning}")
        elif symbol:
            spiritual_parts.append(symbol)

        if base_effects:
            physical_parts.append(_normalize_effect_phrase(base_effects))
        if base_action:
            action_parts.append(_normalize_action_phrase(base_action))

    spiritual_text = _sentence(_human_join(spiritual_parts)) if spiritual_parts else "No clear spiritual meaning was generated."
    physical_text = (
        _sentence(f"In practical terms, this may show up as {_human_join(physical_parts)}")
        if physical_parts else "No clear physical effects were generated."
    )
    action_text = (
        _sentence(" ".join(_compress_phrase_list(action_parts)))
        if action_parts else "Pray for wisdom and confirmation."
    )

    return {
        "spiritual_meaning": spiritual_text,
        "effects_in_physical_realm": physical_text,
        "what_to_do": action_text,
    }


def _compute_seal_from_symbol_count(symbol_count: int) -> Dict[str, str]:
    if symbol_count <= 0:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}
    if symbol_count == 1:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    return {"status": "Delayed", "type": "Processing", "risk": "Medium"}


# ============================================================
# Admin helpers
# ============================================================
def _get_admin_key_from_request() -> str:
    q = (request.args.get("key") or "").strip()
    h = (request.headers.get("X-Admin-Key") or "").strip()
    return q or h


def _require_admin() -> Optional[Response]:
    if not ADMIN_KEY:
        return make_response("Admin is not configured.", 403)
    provided = _get_admin_key_from_request()
    if not provided or provided != ADMIN_KEY:
        return make_response("Forbidden", 403)
    return None


def _build_col_map(header_row: List[str]) -> Dict[str, int]:
    col_map: Dict[str, int] = {}
    for idx, h in enumerate(header_row, start=1):
        col_map[_normalize_header(h)] = idx
    return col_map


def _find_existing_row_index_by_column(ws, lookup_value: str, lookup_col: int) -> Optional[int]:
    target = _normalize_text(lookup_value)
    if not target:
        return None

    col_vals = ws.col_values(lookup_col)
    for i, v in enumerate(col_vals[1:], start=2):
        if _normalize_text(v) == target:
            return i
    return None


def _sheet_primary_lookup_headers(sheet_name: str) -> List[str]:
    if sheet_name == SHEET_BASE_SYMBOLS:
        return ["symbol", "input"]
    if sheet_name == SHEET_BEHAVIOR_RULES:
        return ["behavior_name", "behavior"]
    if sheet_name == SHEET_SIZE_STATE_RULES:
        return ["state_name", "state"]
    if sheet_name == SHEET_LOCATION_RULES:
        return ["location_name", "location"]
    if sheet_name == SHEET_RELATIONSHIP_RULES:
        return ["relationship_name", "relationship"]
    if sheet_name == SHEET_OVERRIDE_RULES:
        return ["override_name", "condition"]
    if sheet_name == SHEET_OUTPUT_TEMPLATES:
        return ["template_type"]
    return ["input", "symbol"]


def _admin_upsert_generic_sheet(sheet_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    ws = _get_or_create_worksheet(sheet_name, rows=2000, cols=30)
    header_row = ws.row_values(1)
    if not header_row:
        raise RuntimeError(f"Sheet {sheet_name} has no header row.")

    col_map = _build_col_map(header_row)
    mode = (payload.get("mode") or "upsert").strip().lower()

    lookup_col = None
    lookup_value = ""

    for key in _sheet_primary_lookup_headers(sheet_name):
        nk = _normalize_header(key)
        if col_map.get(nk) and str(payload.get(key, "")).strip():
            lookup_col = col_map[nk]
            lookup_value = str(payload.get(key, "")).strip()
            break

    if not lookup_col or not lookup_value:
        raise RuntimeError(f"Missing primary lookup field for sheet {sheet_name}.")

    existing_row = None
    if mode != "add":
        existing_row = _find_existing_row_index_by_column(ws, lookup_value, lookup_col)

    row_index = existing_row if existing_row else len(ws.get_all_values()) + 1
    op = "updated" if existing_row else "added"

    updates = []
    for raw_header in header_row:
        nh = _normalize_header(raw_header)
        chosen_value = None

        for payload_key, payload_val in payload.items():
            if _normalize_header(payload_key) == nh:
                chosen_value = payload_val
                break

        if chosen_value is None:
            continue

        chosen_value = _sanitize_keywords_for_storage(str(chosen_value)) if nh == "keywords" else str(chosen_value).strip()
        col_index = col_map.get(nh)
        if col_index:
            updates.append((row_index, col_index, chosen_value))

    if not updates:
        raise RuntimeError("No matching payload fields found for target sheet headers.")

    ws.update_cells([gspread.Cell(r, c, v) for r, c, v in updates], value_input_option="RAW")
    _invalidate_all_caches()

    return {"ok": True, "action": op, "written_row": row_index, "worksheet_name": sheet_name}


def _admin_upsert_to_sheet(payload: Dict[str, Any]) -> Dict[str, Any]:
    target_sheet = (payload.get("target_sheet") or "legacy").strip()
    if target_sheet == "legacy":
        return _admin_upsert_generic_sheet(WORKSHEET_NAME, payload)

    sheet_alias_map = {
        "BaseSymbols": SHEET_BASE_SYMBOLS,
        "BehaviorRules": SHEET_BEHAVIOR_RULES,
        "SizeStateRules": SHEET_SIZE_STATE_RULES,
        "LocationRules": SHEET_LOCATION_RULES,
        "RelationshipRules": SHEET_RELATIONSHIP_RULES,
        "OverrideRules": SHEET_OVERRIDE_RULES,
        "OutputTemplates": SHEET_OUTPUT_TEMPLATES,
    }
    real_sheet_name = sheet_alias_map.get(target_sheet, target_sheet)
    return _admin_upsert_generic_sheet(real_sheet_name, payload)



# ============================================================
# Template/render helpers
# ============================================================
def _safe_return_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return RETURN_URL
    if value.startswith("http://") or value.startswith("https://") or value.startswith("/"):
        return value
    return RETURN_URL

# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET"])
def home():
    try:
        return render_template(TEMPLATE_INDEX)
    except Exception:
        return redirect(RETURN_URL, code=302)


@app.route("/health", methods=["GET"])
def health():
    doctrine_available = False
    journal_available = False
    self_healing_ready = False

    try:
        _ensure_core_worksheets()
        self_healing_ready = True
        doctrine_available = _doctrine_sheets_available()
        journal_available = True
    except Exception:
        doctrine_available = False
        journal_available = False
        self_healing_ready = False

    return jsonify({
        "ok": True,
        "service": "dream-interpreter",
        "sheet": WORKSHEET_NAME,
        "has_spreadsheet_id": bool(SPREADSHEET_ID),
        "allowed_origins": allowed_origins,
        "match_mode": "strict_word_boundary_with_longest_phrase_priority_and_overlap_guard",
        "brain_layers": [
            "base_symbol_logic",
            "behavior_logic",
            "state_logic",
            "location_logic",
            "relationship_logic",
            "ending_seal_logic",
            "override_logic",
            "template_logic",
        ],
        "doctrine_mode_enabled": DOCTRINE_MODE,
        "doctrine_sheets_available": doctrine_available,
        "doctrine_sheet_names": DOCTRINE_SHEET_NAMES,
        "dream_journal_sheet": SHEET_DREAM_JOURNAL,
        "dream_journal_available": journal_available,
        "self_healing_worksheets_ready": self_healing_ready,
        "free_quota": FREE_TRIES,
        "stripe_configured": bool(_stripe_config_ok()),
        "price_dream_pack_set": bool(PRICE_DREAM_PACK),
        "return_url": RETURN_URL,
        "session_premium": _is_premium_session(),
        "session_email": _get_session_email(),
        "session_dream_pack": _get_dream_pack_status(_get_session_email()),
        "template_index": TEMPLATE_INDEX,
        "template_upgrade": TEMPLATE_UPGRADE,
    })


@app.route("/healthz", methods=["GET"])
def healthz():
    return health()


@app.route("/upgrade", methods=["GET"])
def upgrade():
    try:
        return render_template(TEMPLATE_UPGRADE)
    except Exception:
        return redirect(RETURN_URL, code=302)


@app.route("/check-access", methods=["POST", "OPTIONS"])
def check_access():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    email = _normalize_email(data.get("email") or "")

    if not _validate_email(email):
        return jsonify({"ok": False, "error": "Please enter a valid email."}), 400

    _persist_email_to_session(email)
    access_ok, access_meta = _has_active_access(email)

    if access_ok and access_meta.get("type") == "subscription":
        _set_premium_session(email)
        resp = make_response(jsonify({"ok": True, "access": "active", "return_url": RETURN_URL}))
        resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
        return resp

    if access_ok and access_meta.get("type") == "dream_pack":
        _set_buyer_session(email)
        resp = make_response(jsonify({
            "ok": True,
            "access": "dream_pack_active",
            "return_url": RETURN_URL,
            "dream_pack": access_meta.get("details", {}),
        }))
        resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
        return resp

    return jsonify({"ok": False, "access": "inactive", "message": "No active subscription or dream pack found."}), 402


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not _stripe_config_ok():
        return make_response("Stripe not configured.", 500)

    email = _extract_email_from_request()
    if not _validate_email(email):
        return make_response("Missing email.", 400)

    _persist_email_to_session(email)
    stripe.api_key = STRIPE_SECRET_KEY

    try:
        requested_return = _safe_return_url(
            request.form.get("return")
            or request.args.get("return")
            or request.headers.get("X-Return-Url")
            or RETURN_URL
        )

        checkout = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": DEFAULT_STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=email,
            metadata={"purchase_type": "subscription", "email": email, "return_url": requested_return},
            success_url=url_for("payment_success", _external=True) + f"?email={email}&return={requested_return}",
            cancel_url=url_for("upgrade", _external=True) + f"?email={email}&return={requested_return}",
        )
        return redirect(checkout.url, code=303)
    except Exception as e:
        return make_response(f"Stripe error: {str(e)}", 500)


@app.route("/create-dream-pack-checkout-session", methods=["POST", "GET"])
def create_dream_pack_checkout_session():
    if not _stripe_dream_pack_ok():
        return make_response("Dream pack Stripe config missing.", 500)

    email = _extract_email_from_request()
    if not _validate_email(email):
        return make_response("Missing email for dream pack checkout.", 400)

    _persist_email_to_session(email)
    stripe.api_key = STRIPE_SECRET_KEY

    try:
        requested_return = _safe_return_url(
            request.form.get("return")
            or request.args.get("return")
            or request.headers.get("X-Return-Url")
            or RETURN_URL
        )

        checkout = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": PRICE_DREAM_PACK, "quantity": 1}],
            customer_email=email,
            metadata={
                "purchase_type": "dream_pack",
                "dream_pack_uses": str(DREAM_PACK_USES),
                "dream_pack_hours": str(DREAM_PACK_HOURS),
                "email": email,
                "return_url": requested_return,
            },
            success_url=url_for("dream_pack_success", _external=True) + f"?email={email}&return={requested_return}",
            cancel_url=url_for("upgrade", _external=True) + f"?email={email}&return={requested_return}",
        )
        return redirect(checkout.url, code=303)
    except Exception as e:
        return make_response(f"Stripe dream pack error: {str(e)}", 500)


@app.route("/payment-success", methods=["GET"])
def payment_success():
    email = _normalize_email(request.args.get("email") or "")
    requested_return = _safe_return_url(request.args.get("return") or RETURN_URL)

    if _validate_email(email):
        _persist_email_to_session(email)
        access_ok, access_meta = _has_active_access(email)
        if access_ok and access_meta.get("type") == "subscription":
            _set_premium_session(email)

    resp = redirect(requested_return, code=302)
    resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    return resp


@app.route("/dream-pack-success", methods=["GET"])
def dream_pack_success():
    email = _normalize_email(request.args.get("email") or "")
    requested_return = _safe_return_url(request.args.get("return") or RETURN_URL)

    if _validate_email(email):
        _persist_email_to_session(email)
        _set_buyer_session(email)
        _mark_dream_pack_purchase(email, uses=DREAM_PACK_USES, hours=DREAM_PACK_HOURS)

    resp = redirect(requested_return, code=302)
    resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    return resp


@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    if not stripe or not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET:
        return ("not configured", 400)

    stripe.api_key = STRIPE_SECRET_KEY
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return ("bad signature", 400)

    event_type = event.get("type", "")
    data_obj = (event.get("data") or {}).get("object") or {}

    if event_type == "checkout.session.completed":
        metadata = data_obj.get("metadata") or {}
        email = _normalize_email(metadata.get("email") or data_obj.get("customer_email") or "")
        customer_id = data_obj.get("customer") or ""
        mode = (data_obj.get("mode") or "").strip().lower()
        purchase_type = (metadata.get("purchase_type") or "").strip().lower()

        if _validate_email(email) and mode == "subscription":
            _mark_subscriber(email, True, customer_id)

        if _validate_email(email) and mode == "payment" and purchase_type == "dream_pack":
            _mark_dream_pack_purchase(email, uses=DREAM_PACK_USES, hours=DREAM_PACK_HOURS)

    return ("ok", 200)


@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or data.get("text") or "").strip()

    _log("RAW JSON RECEIVED:", _safe_debug_payload_preview(data))
    _log("RAW DREAM RECEIVED:", repr(dream))

    validation_error = _validate_dream_text(dream)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    request_email = _extract_email_from_request()
    if _validate_email(request_email):
        _persist_email_to_session(request_email)

    session_email = _get_session_email()
    access_ok, access_meta = _has_active_access(session_email)
    access_type = access_meta.get("type", "")
    is_paid = access_ok and access_type == "subscription"

    dream_pack_status_before = _get_dream_pack_status(session_email)
    has_active_dream_pack = access_ok and access_type == "dream_pack"

    if not access_ok:
        ip = _get_client_ip()
        cookie_used = _get_cookie_tries_used()
        ip_used = _shadow_count(ip)
        effective_used = max(cookie_used, ip_used)

        if effective_used >= FREE_TRIES:
            return jsonify({
                "blocked": True,
                "reason": "free_limit_reached",
                "message": f"You’ve used your {FREE_TRIES} free tries.",
                "free_uses_left": 0,
                "access": "blocked",
                "is_paid": False,
                "email": session_email,
                "dream_pack_available": bool(PRICE_DREAM_PACK),
            }), 402

    free_uses_left = 0
    dream_pack_status_after = dream_pack_status_before

    if not access_ok:
        ip = _get_client_ip()
        effective_used = max(_get_cookie_tries_used(), _shadow_count(ip))
        _shadow_increment(ip)
        free_uses_left = _free_tries_remaining_after_this(effective_used)
    elif has_active_dream_pack:
        dream_pack_status_after = _consume_dream_pack_use(session_email)

    doctrine_active = DOCTRINE_MODE and _doctrine_sheets_available()

    if doctrine_active:
        try:
            sheets = _load_doctrine_sheets()
            base_rows = sheets.get(SHEET_BASE_SYMBOLS, [])
            behavior_rows = sheets.get(SHEET_BEHAVIOR_RULES, [])
            state_rows = sheets.get(SHEET_SIZE_STATE_RULES, [])
            location_rows = sheets.get(SHEET_LOCATION_RULES, [])
            relationship_rows = sheets.get(SHEET_RELATIONSHIP_RULES, [])
            override_rows = sheets.get(SHEET_OVERRIDE_RULES, [])
            template_rows = sheets.get(SHEET_OUTPUT_TEMPLATES, [])

            behaviors = _detect_rule_hits(dream, behavior_rows, "behavior")
            states = _detect_rule_hits(dream, state_rows, "state")
            locations = _detect_rule_hits(dream, location_rows, "location")
            relationships = _detect_rule_hits(dream, relationship_rows, "relationship")
            base_matches = _match_base_symbols_doctrine(
                dream,
                base_rows,
                top_k=BASE_MATCH_TOP_K,
                behaviors=behaviors,
                states=states,
                locations=locations,
                relationships=relationships,
            )
            override_hit = _apply_override_rules(base_matches, behaviors, states, locations, relationships, dream, override_rows)
            doctrine_seal = _compute_doctrine_seal(dream, base_matches, behaviors, states, locations, relationships, override_hit)

            receipt_id = f"JTS-{secrets.token_hex(4).upper()}"
            built = _build_doctrine_interpretation(
                dream, base_matches, behaviors, states, locations, relationships, override_hit, template_rows, doctrine_seal
            )

            payload = {
                "engine_mode": "doctrine",
                "access": "paid" if is_paid else ("dream_pack" if has_active_dream_pack else "free"),
                "is_paid": bool(is_paid),
                "email": session_email,
                "free_uses_left": free_uses_left,
                "dream_pack": dream_pack_status_after,
                "seal": doctrine_seal,
                "brain": {
                    "behaviors": [b["name"] for b in behaviors],
                    "states": [s["name"] for s in states],
                    "locations": [l["name"] for l in locations],
                    "relationships": [r["name"] for r in relationships],
                    "override_applied": built["override_applied"],
                    "override_name": built["override_name"],
                    "template_type": built.get("template_type", "default"),
                },
                "interpretation": built["interpretation"],
                "full_interpretation": built["full_interpretation"],
                "receipt": {
                    "id": receipt_id,
                    "top_symbols": built["top_symbols"],
                    "share_phrase": "I decoded my dream on Jamaican True Stories.",
                },
            }
            resp = make_response(jsonify(payload))
        except Exception as e:
            return jsonify({"error": "Doctrine engine failed", "details": str(e)}), 500
    else:
        try:
            legacy_rows = _load_legacy_sheet_rows()
        except Exception as e:
            return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

        matches = _match_symbols_legacy(dream, legacy_rows, top_k=BASE_MATCH_TOP_K)
        receipt_id = f"JTS-{secrets.token_hex(4).upper()}"
        seal = _compute_seal_from_symbol_count(len(matches))
        built_legacy = _build_legacy_interpretation(matches)

        payload = {
            "engine_mode": "legacy",
            "access": "paid" if is_paid else ("dream_pack" if has_active_dream_pack else "free"),
            "is_paid": bool(is_paid),
            "email": session_email,
            "free_uses_left": free_uses_left,
            "dream_pack": dream_pack_status_after,
            "seal": seal,
            "brain": {},
            "interpretation": built_legacy,
            "full_interpretation": "\n\n".join([
                _sentence("This dream is revealing symbolic meaning through the matched symbols"),
                built_legacy["spiritual_meaning"],
                built_legacy["effects_in_physical_realm"],
                built_legacy["what_to_do"],
            ]),
            "receipt": {
                "id": receipt_id,
                "top_symbols": [_get_symbol_cell(row) for row, _sc, _hit in matches if _get_symbol_cell(row)],
                "share_phrase": "I decoded my dream on Jamaican True Stories.",
            },
        }
        resp = make_response(jsonify(payload))

    if not access_ok:
        resp.set_cookie(
            COOKIE_NAME,
            str(_get_cookie_tries_used() + 1),
            max_age=COOKIE_MAX_AGE,
            samesite=COOKIE_SAMESITE,
            secure=COOKIE_SECURE,
        )
    else:
        resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)

    return resp


@app.route("/journal/save", methods=["POST", "OPTIONS"])
def journal_save():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    email = _normalize_email(data.get("email") or "")
    dream_text = (data.get("dream_text") or "").strip()

    if not _validate_email(email):
        return jsonify({"ok": False, "error": "Valid email is required."}), 400
    if not dream_text:
        return jsonify({"ok": False, "error": "Dream text is required."}), 400

    try:
        _persist_email_to_session(email)
        result = _append_dream_journal_entry({
            "email": email,
            "dream_text": dream_text,
            "spiritual_meaning": data.get("spiritual_meaning", ""),
            "effects_in_physical_realm": data.get("effects_in_physical_realm", ""),
            "what_to_do": data.get("what_to_do", ""),
            "full_interpretation": data.get("full_interpretation", ""),
            "receipt_id": data.get("receipt_id", ""),
            "top_symbols": data.get("top_symbols", []),
            "seal_status": data.get("seal_status", ""),
            "seal_type": data.get("seal_type", ""),
            "seal_risk": data.get("seal_risk", ""),
            "engine_mode": data.get("engine_mode", ""),
            "access_type": data.get("access_type", ""),
            "is_saved": "yes",
            "notes": data.get("notes", ""),
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/journal/history", methods=["GET", "OPTIONS"])
def journal_history():
    if request.method == "OPTIONS":
        return _preflight_ok()

    email = _normalize_email(request.args.get("email") or "")
    if not _validate_email(email):
        return jsonify({"ok": False, "error": "Valid email is required."}), 400

    try:
        _persist_email_to_session(email)
        ws = _get_journal_worksheet()
        rows = ws.get_all_records()

        matches = []
        for row in rows:
            row_email = _normalize_email(row.get("email") or "")
            if row_email != email:
                continue

            top_symbols_raw = (row.get("top_symbols") or "").strip()
            top_symbols = [x.strip() for x in top_symbols_raw.split(",") if x.strip()]

            matches.append({
                "entry_id": row.get("entry_id", ""),
                "created_at": row.get("created_at", ""),
                "email": row_email,
                "dream_text": row.get("dream_text", ""),
                "spiritual_meaning": row.get("spiritual_meaning", ""),
                "effects_in_physical_realm": row.get("effects_in_physical_realm", ""),
                "what_to_do": row.get("what_to_do", ""),
                "full_interpretation": row.get("full_interpretation", ""),
                "receipt_id": row.get("receipt_id", ""),
                "top_symbols": top_symbols,
                "seal_status": row.get("seal_status", ""),
                "seal_type": row.get("seal_type", ""),
                "seal_risk": row.get("seal_risk", ""),
                "engine_mode": row.get("engine_mode", ""),
                "access_type": row.get("access_type", ""),
                "notes": row.get("notes", ""),
            })

        matches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jsonify({"ok": True, "email": email, "entries": matches[:50]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/track", methods=["POST", "OPTIONS"])
def track():
    if request.method == "OPTIONS":
        return _preflight_ok()

    payload = request.get_json(silent=True) or {}
    event_name = payload.get("event") or payload.get("event_type") or "unknown"
    return jsonify({
        "ok": True,
        "event": event_name,
        "session_email": _get_session_email(),
        "session_premium": _is_premium_session(),
        "dream_pack": _get_dream_pack_status(_get_session_email()),
    })


@app.route("/admin/upsert", methods=["POST", "OPTIONS"])
def admin_upsert():
    if request.method == "OPTIONS":
        return _preflight_ok()

    auth_fail = _require_admin()
    if auth_fail:
        return jsonify({"ok": False, "error": "Forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    try:
        result = _admin_upsert_to_sheet(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/debug/config", methods=["GET"])
def debug_config():
    if not DEBUG_MATCH:
        return jsonify({"error": "Debug disabled"}), 403

    return jsonify({
        "spreadsheet_id": SPREADSHEET_ID,
        "worksheet_name": WORKSHEET_NAME,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "allowed_origins": allowed_origins,
        "admin_configured": bool(ADMIN_KEY),
        "free_quota": FREE_TRIES,
        "shadow_window_hours": SHADOW_WINDOW_HOURS,
        "stripe_has_key": bool(STRIPE_SECRET_KEY),
        "stripe_has_price": bool(DEFAULT_STRIPE_PRICE_ID),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "price_dream_pack_set": bool(PRICE_DREAM_PACK),
        "dream_pack_uses": DREAM_PACK_USES,
        "dream_pack_hours": DREAM_PACK_HOURS,
        "counts_file": str(USAGE_COUNTS_PATH),
        "subscribers_file": str(SUBSCRIBERS_PATH),
        "cookie_samesite": COOKIE_SAMESITE,
        "return_url": RETURN_URL,
        "doctrine_sheet_names": DOCTRINE_SHEET_NAMES,
        "relationship_rules_sheet": SHEET_RELATIONSHIP_RULES,
        "template_index": TEMPLATE_INDEX,
        "template_upgrade": TEMPLATE_UPGRADE,
    })


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_mode,
    )

def _detect_false_call_event(dream: str) -> bool:
    text_n = _normalize_text(dream)
    if not text_n:
        return False
    heard_call = (
        _contains_any_phrase(text_n, [
            "calling my name", "called my name", "heard someone calling",
            "heard my name", "someone calling me", "voice calling me",
        ])
        or ("name" in text_n and "calling" in text_n)
    )
    nobody_there = _contains_any_phrase(text_n, [
        "no one was there", "nobody was there", "there was no one there",
        "nobody there", "no one there",
    ])
    return heard_call and nobody_there


def _detect_breach_event(dream: str) -> bool:
    text_n = _normalize_text(dream)
    if not text_n:
        return False
    breach_phrases = [
        "door was wide open", "door was open", "open even though i had closed it",
        "even though i had closed it", "tried to come inside", "come inside",
        "trying to enter", "gain access", "came inside", "breach",
    ]
    return _contains_any_phrase(text_n, breach_phrases)


def _detect_restored_boundary_event(dream: str) -> bool:
    ending = _extract_dream_ending_text(dream)
    phrases = [
        "shut the door", "closed the door", "locked the door",
        "shut the door tightly", "closed it behind me", "ran inside and shut the door",
    ]
    return _contains_any_phrase(ending, phrases)


def _detect_hidden_danger_event(dream: str, states: List[Dict[str, Any]]) -> bool:
    text_n = _normalize_text(dream)
    calm_like = _contains_any_phrase(text_n, ["looked calm", "looked normal", "at first", "seemed safe", "everything looked normal"])
    danger_like = _contains_any_phrase(text_n, [
        "suddenly became rough", "started pulling me", "out of control",
        "suddenly", "became rough", "turned dark", "changed suddenly",
    ])
    return _detect_transition_instability(dream, states) or (calm_like and danger_like)


def _detect_pull_into_danger_event(dream: str) -> bool:
    text_n = _normalize_text(dream)
    return _contains_any_phrase(text_n, [
        "started pulling me", "pulling me", "dragging me", "out of control",
        "could not stop", "couldnt stop", "could not reach", "couldnt reach",
    ])


def _detect_known_person_distortion_event(dream: str) -> bool:
    text_n = _normalize_text(dream)
    knew_person = _contains_any_phrase(text_n, ["i knew", "someone i knew", "man i knew", "person i knew"])
    distortion = _contains_any_phrase(text_n, ["face looked different", "looked different", "strange face", "looked strange"])
    return knew_person and distortion


def _extract_event_flags(dream: str, states: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {
        "false_call": _detect_false_call_event(dream),
        "breach": _detect_breach_event(dream),
        "restored_boundary": _detect_restored_boundary_event(dream),
        "hidden_danger": _detect_hidden_danger_event(dream, states),
        "pull_into_danger": _detect_pull_into_danger_event(dream),
        "escape": _dream_has_escape_cue(dream),
        "intrusion": _dream_has_intrusion_cue(dream),
        "known_person_distortion": _detect_known_person_distortion_event(dream),
    }


def _event_layer_strength(dream: str, states: List[Dict[str, Any]]) -> int:
    flags = _extract_event_flags(dream, states)
    strength = 0
    if flags["false_call"]:
        strength += 8
    if flags["breach"]:
        strength += 8
    if flags["restored_boundary"]:
        strength += 8
    if flags["hidden_danger"]:
        strength += 7
    if flags["pull_into_danger"]:
        strength += 7
    if flags["escape"]:
        strength += 6
    if flags["intrusion"]:
        strength += 5
    if flags["known_person_distortion"]:
        strength += 6
    return strength


def _build_event_scenario(
    dream: str,
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Dict[str, str]:
    flags = _extract_event_flags(dream, states)
    relationship_clause = _compact_rule_meaning_clause(relationships, _get_relationship_meaning_modifier, max_items=1)
    location_clause = _compact_rule_meaning_clause(locations, _get_location_life_area_meaning, max_items=1)

    if flags["false_call"] and flags["breach"] and flags["restored_boundary"]:
        lead = "deception or distraction trying to gain access into your personal life, but the ending shows you shut it down"
        support = "A false signal pulled your attention outward, then an opening appeared where a boundary should have remained closed"
        action = "Do not respond blindly to every call or signal. Guard your space and keep your spiritual boundaries firm"
        effects = "subtle distraction, boundary pressure, or access being tested around your personal life"
        seal_type = "Protection"
        risk = "Medium"
    elif flags["hidden_danger"] and flags["pull_into_danger"] and flags["escape"]:
        lead = "a situation that looked manageable at first but turned unstable before you came out of it"
        support = "What appeared calm changed suddenly, and the dream highlights danger hidden beneath a normal surface"
        action = "Move carefully in situations that look harmless at first. Pray for discernment before stepping in further"
        effects = "instability, emotional pressure, and being pulled deeper into something that is not as safe as it appears"
        seal_type = "Breakthrough"
        risk = "Medium"
    elif flags["intrusion"] and flags["restored_boundary"]:
        lead = "pressure trying to enter your space, with the ending showing boundary restoration"
        support = "The dream centers on access, resistance, and shutting the opening again"
        action = "Stay alert, guard your boundaries, and do not permit what should remain outside"
        effects = "boundary pressure or unwanted access being tested"
        seal_type = "Protection"
        risk = "Medium"
    elif flags["known_person_distortion"]:
        lead = "a familiar influence appearing with hidden or altered intent"
        support = "The dream warns that something known to you may not be presenting itself truthfully"
        action = "Pray for discernment around familiar people or situations and do not trust appearances too quickly"
        effects = "confusion, deception, or mixed signals tied to a familiar connection"
        seal_type = "Warning"
        risk = "Medium"
    else:
        lead = ""
        support = ""
        action = ""
        effects = ""
        seal_type = ""
        risk = ""

    if lead:
        if relationship_clause and "familiar" not in lead and "connection" not in effects:
            support = _merge_natural_paragraphs([support, f"The people involved connect this to {relationship_clause}."])
        if location_clause and "personal life" not in lead and "personal life" not in support:
            support = _merge_natural_paragraphs([support, f"The setting ties this to {location_clause}."])

    return {
        "lead": _clean_sentence(lead),
        "support": _clean_sentence(support),
        "action": _clean_sentence(action),
        "effects": _clean_sentence(effects),
        "seal_type": seal_type,
        "risk": risk,
    }

