# app.py — Jamaican True Stories Dream Interpreter
# Option A: Stripe-only access, HYBRID FREE GATE
#
# What this version does:
# - Keeps: Hybrid free-tries gate (cookie + IP shadow)
# - Keeps: /interpret, /track, /health, /healthz, /debug/config, /debug/sheet (guarded), /admin + /admin/upsert
# - Keeps: Stripe Checkout + webhook tracking
# - Removes: ALL account/password logic (no /auth/signup, /auth/login, no users.json)
# - Adds: Stripe-only unlock via email
# - Adds: Longest-phrase-first dream symbol matching with overlap blocking
#   so compound phrases like "baby mouse" can win before standalone words like "baby"
#
# Env vars (Render):
# - RETURN_URL=https://jamaicantruestories.com/pages/dream-interpreter
# - ALLOWED_ORIGINS=...
# - SPREADSHEET_ID, WORKSHEET_NAME
# - GOOGLE_SERVICE_ACCOUNT_JSON (or GOOGLE_SERVICE_ACCOUNT_FILE)
# - FREE_QUOTA, FREE_TRIES_COOKIE, SHADOW_WINDOW_HOURS
# - COUNTS_FILE, SUBSCRIBERS_FILE
# - STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
# - PRICE_WEEKLY / PRICE_MONTHLY
# - ADMIN_KEY (or JTS_ADMIN_KEY / ADMIN_TOKEN / JTS_ADMIN)
# - DEBUG_MATCH=1 (optional)

import os
import json
import time
import re
import secrets
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from flask import (
    Flask, request, jsonify, make_response, Response,
    redirect, url_for, session, render_template
)
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials

try:
    import stripe
except Exception:
    stripe = None


# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

# Sessions must work cross-site (Shopify -> interpreter)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "").strip() or secrets.token_hex(32)
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True

DEFAULT_ALLOWED = [
    "https://jamaicantruestories.com",
    "https://www.jamaicantruestories.com",
    "https://interpreter.jamaicantruestories.com",
]
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or DEFAULT_ALLOWED

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

RETURN_URL = os.getenv("RETURN_URL", "https://jamaicantruestories.com/pages/dream-interpreter").strip()

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
_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "rows": [], "headers": []}

DEBUG_MATCH = os.getenv("DEBUG_MATCH", "").strip().lower() in {"1", "true", "yes", "on"}

NARRATIVE_ENABLED = os.getenv("NARRATIVE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
NARRATIVE_MAX_SYMBOLS = int(os.getenv("NARRATIVE_MAX_SYMBOLS", "3"))


# ----------------------------
# Hybrid Gate Settings
# ----------------------------
FREE_TRIES = int(os.getenv("FREE_QUOTA", os.getenv("FREE_TRIES", "3")))
COOKIE_NAME = os.getenv("FREE_TRIES_COOKIE", "jts_free_tries_used")
SHADOW_WINDOW_HOURS = int(os.getenv("SHADOW_WINDOW_HOURS", "72"))

COUNTS_FILE = os.getenv("COUNTS_FILE", "usage_counts.json")
SUBSCRIBERS_FILE = os.getenv("SUBSCRIBERS_FILE", "subscribers.json")

USAGE_COUNTS_PATH = Path(COUNTS_FILE)
SUBSCRIBERS_PATH = Path(SUBSCRIBERS_FILE)


# ----------------------------
# Stripe Settings
# ----------------------------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

PRICE_WEEKLY = os.getenv("PRICE_WEEKLY", "").strip()
PRICE_MONTHLY = os.getenv("PRICE_MONTHLY", "").strip()
DEFAULT_STRIPE_PRICE_ID = PRICE_WEEKLY or PRICE_MONTHLY


# ----------------------------
# Cookie settings
# ----------------------------
COOKIE_SAMESITE = "None"
COOKIE_SECURE = True
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year


# ----------------------------
# Helpers
# ----------------------------
def _preflight_ok():
    return make_response("", 204)


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


def _log(*args):
    if DEBUG_MATCH:
        print(*args, flush=True)


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


def _title_case_symbol(sym: str) -> str:
    sym = _clean_sentence(sym)
    if not sym:
        return ""
    return sym[:1].upper() + sym[1:]


def _format_label_value(symbol: str, text: str) -> str:
    symbol = _title_case_symbol(symbol)
    text = _clean_sentence(text)
    if not symbol or not text:
        return ""
    return f"{symbol}: {text}"


def _invalidate_cache():
    _CACHE["loaded_at"] = 0.0
    _CACHE["rows"] = []
    _CACHE["headers"] = []


# ----------------------------
# File utilities (atomic JSON)
# ----------------------------
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


# ----------------------------
# Hybrid Gate: cookie + IP shadow
# ----------------------------
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
        first = datetime.fromisoformat(rec["first_seen"].replace("Z", ""))
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
            first = datetime.fromisoformat(rec["first_seen"].replace("Z", ""))
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


# ----------------------------
# Subscribers file helpers
# ----------------------------
def _load_subscribers() -> Dict[str, Any]:
    data = _read_json_file(SUBSCRIBERS_PATH, {})
    if not isinstance(data, dict):
        return {}
    return data


def _save_subscribers(data: Dict[str, Any]):
    _write_json_file_atomic(SUBSCRIBERS_PATH, data)


def _mark_subscriber(email: str, is_active: bool, stripe_customer_id: str = ""):
    if not email:
        return
    email_n = email.strip().lower()
    subs = _load_subscribers()
    subs[email_n] = {
        "email": email_n,
        "is_active": bool(is_active),
        "stripe_customer_id": stripe_customer_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_subscribers(subs)


# ----------------------------
# Stripe-only premium session
# ----------------------------
def _get_session_email() -> str:
    return (session.get("subscriber_email") or "").strip().lower()


def _is_premium_session() -> bool:
    return bool(session.get("premium") is True and _get_session_email())


def _set_premium_session(email: str):
    session["subscriber_email"] = (email or "").strip().lower()
    session["premium"] = True
    session["premium_set_at"] = datetime.utcnow().isoformat() + "Z"


def _clear_premium_session():
    session.pop("subscriber_email", None)
    session.pop("premium", None)
    session.pop("premium_set_at", None)


def _stripe_config_ok() -> bool:
    return bool(stripe and STRIPE_SECRET_KEY and DEFAULT_STRIPE_PRICE_ID)


def _stripe_active_subscription_for_email(email: str) -> Tuple[bool, str]:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return (False, "")

    if not stripe or not STRIPE_SECRET_KEY:
        return (False, "")

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
                    return (True, cid)
        return (False, "")
    except Exception as e:
        _log("Stripe check error:", str(e))
        return (False, "")


# ----------------------------
# Sheet field getters
# ----------------------------
def _get_symbol_cell(row: Dict) -> str:
    return (
        row.get("input")
        or row.get("symbol")
        or row.get("symbols")
        or row.get("input symbol")
        or row.get("dream symbol")
        or row.get("symbol name")
        or ""
    ).strip()


def _get_spiritual_meaning_cell(row: Dict) -> str:
    return (
        row.get("spiritual meaning")
        or row.get("spiritual_meaning")
        or row.get("spiritual")
        or row.get("meaning")
        or ""
    ).strip()


def _get_effects_cell(row: Dict) -> str:
    return (
        row.get("effects in the physical realm")
        or row.get("effects_in_the_physical_realm")
        or row.get("physical effects")
        or row.get("physical_effects")
        or row.get("effects")
        or ""
    ).strip()


def _get_what_to_do_cell(row: Dict) -> str:
    return (
        row.get("what to do")
        or row.get("what_to_do")
        or row.get("action")
        or row.get("actions")
        or ""
    ).strip()


def _get_keywords_cell(row: Dict) -> str:
    return (row.get("keywords") or row.get("keyword") or row.get("tags") or "").strip()


# ----------------------------
# Google credentials + sheet access
# ----------------------------
def _get_credentials() -> Credentials:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        raw_fixed = raw.replace("\\n", "\n")
        info = json.loads(raw_fixed)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
            "or provide credentials.json via GOOGLE_SERVICE_ACCOUNT_FILE."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _get_ws():
    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID env var is not set.")
    creds = _get_credentials()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)
    return ws


def _load_sheet_rows(force: bool = False) -> List[Dict]:
    now = time.time()
    if (not force) and _CACHE["rows"] and (now - _CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _CACHE["rows"]

    ws = _get_ws()
    values = ws.get_all_values()

    if not values or len(values) < 2:
        _CACHE["rows"] = []
        _CACHE["headers"] = []
        _CACHE["loaded_at"] = now
        return []

    headers = [_normalize_header(h) for h in values[0]]
    rows: List[Dict] = []

    for r in values[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        item = {headers[i]: (r[i] or "").strip() for i in range(len(headers))}
        rows.append(item)

    _CACHE["rows"] = rows
    _CACHE["headers"] = headers
    _CACHE["loaded_at"] = now
    return rows


# ----------------------------
# Matching logic
# ----------------------------
def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\|;]+", s)
    out = []
    for p in parts:
        p = _normalize_text(p)
        if p:
            out.append(p)
    return out


def _compile_boundary_regex(token: str) -> re.Pattern:
    token_n = _normalize_text(token)
    if not token_n:
        return re.compile(r"(?!x)x")
    return re.compile(rf"(?<!\w){re.escape(token_n)}(?!\w)")


def _tokenize_words(s: str) -> List[str]:
    s = _normalize_text(s)
    return [w for w in s.split() if w]


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
    used_spans: Optional[List[Tuple[int, int]]] = None
) -> Tuple[int, Optional[Dict[str, Any]]]:
    symbol_raw = _get_symbol_cell(row)
    if not symbol_raw:
        return 0, None

    symbol = _normalize_text(symbol_raw)
    if not symbol:
        return 0, None

    keywords = _split_keywords(_get_keywords_cell(row))
    candidates: List[Tuple[str, str, int]] = []

    # Exact symbol first
    if symbol:
        if len(_tokenize_words(symbol)) > 1:
            candidates.append(("symbol_phrase", symbol, 100 - _symbol_length_penalty(symbol_raw)))
        else:
            candidates.append(("symbol", symbol, 100))

    # Then keywords
    for kw in keywords:
        if not kw:
            continue
        if len(_tokenize_words(kw)) > 1:
            candidates.append(("keyword_phrase", kw, 94))
        else:
            candidates.append(("keyword", kw, 90))

    # Longest / most specific first
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


def _match_symbols_strict(
    dream: str, rows: List[Dict], top_k: int = 3
) -> List[Tuple[Dict, int, Optional[Dict[str, Any]]]]:
    dream_norm = _normalize_text(dream)
    if not dream_norm:
        return []

    _log("\n--- DEBUG_MATCH ON ---")
    _log("DREAM_RAW:", repr(dream))
    _log("DREAM_NORM:", repr(dream_norm))

    candidates: List[Tuple[Dict, int, Optional[Dict[str, Any]]]] = []
    for row in rows:
        symbol_raw = _get_symbol_cell(row)
        if not symbol_raw:
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

    if out:
        _log("MATCHES:")
        for row, sc, hit in out:
            _log(f" - {_get_symbol_cell(row)!r} score={sc} via={hit}")
    else:
        _log("MATCHES: (none)")

    _log("USED_SPANS:", used_spans)
    _log("--- END DEBUG_MATCH ---\n")
    return out


# ----------------------------
# Output building
# ----------------------------
def _combine_fields(matches: List[Tuple[Dict, int, Optional[Dict[str, Any]]]]) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, _sc, _hit in matches:
        symbol = _get_symbol_cell(row)
        sm = _get_spiritual_meaning_cell(row)
        pe = _get_effects_cell(row)
        ac = _get_what_to_do_cell(row)

        line = _format_label_value(symbol, sm)
        if line:
            spiritual_parts.append(line)

        line = _format_label_value(symbol, pe)
        if line:
            physical_parts.append(line)

        line = _format_label_value(symbol, ac)
        if line:
            action_parts.append(line)

    return {
        "spiritual_meaning": "\n".join(spiritual_parts).strip(),
        "effects_in_physical_realm": "\n".join(physical_parts).strip(),
        "what_to_do": "\n".join(action_parts).strip(),
    }


def _make_receipt_id() -> str:
    return f"JTS-{secrets.token_hex(4).upper()}"


def _compute_seal(matches: List[Tuple[Dict, int, Optional[Dict[str, Any]]]]) -> Dict[str, str]:
    if not matches:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}

    avg = sum(sc for _, sc, _ in matches) / max(len(matches), 1)
    if avg >= 95:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    if avg >= 88:
        return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
    return {"status": "Delayed", "type": "Processing", "risk": "High"}


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = _clean_sentence(x)
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _extract_symbol_fields(matches: List[Tuple[Dict, int, Optional[Dict[str, Any]]]], max_n: int = 3):
    symbols: List[str] = []
    meanings: List[str] = []
    effects: List[str] = []
    actions: List[str] = []

    for row, _sc, _hit in matches[:max_n]:
        symbols.append(_get_symbol_cell(row))
        meanings.append(_get_spiritual_meaning_cell(row))
        effects.append(_get_effects_cell(row))
        actions.append(_get_what_to_do_cell(row))

    return symbols, meanings, effects, actions


def build_full_interpretation_from_doctrine(matches: List[Tuple[Dict, int, Optional[Dict[str, Any]]]]) -> str:
    if not matches or not NARRATIVE_ENABLED:
        return ""

    symbols, meanings, effects, actions = _extract_symbol_fields(matches, max_n=NARRATIVE_MAX_SYMBOLS)

    opening = (
        "This dream is revealing a spiritual condition, not predicting physical harm. "
        "Dreams speak in symbols, and meaning is shown through what appears and what happens."
    )

    symbol_sentences: List[str] = []
    for sym, mean in zip(symbols, meanings):
        mean_clean = _strip_trailing_punct(mean)
        if sym and mean_clean:
            symbol_sentences.append(_ensure_terminal_punct(f"{_title_case_symbol(sym)} points to {mean_clean}"))

    effect_sentences: List[str] = []
    for sym, eff in zip(symbols, effects):
        eff_clean = _strip_trailing_punct(eff)
        if sym and eff_clean:
            effect_sentences.append(_ensure_terminal_punct(f"{_title_case_symbol(sym)} can show up as {eff_clean} in the natural realm"))

    action_lines = _dedupe_preserve_order([_strip_trailing_punct(a) for a in actions if a])
    action_text = ""
    if action_lines:
        if len(action_lines) == 1:
            action_text = _ensure_terminal_punct(f"What to do: {action_lines[0]}")
        else:
            joined = ". Then, ".join([_strip_trailing_punct(a) for a in action_lines])
            action_text = _ensure_terminal_punct(f"What to do: {joined}")

    closing = (
        "Dreams expose what needs attention so you can respond. "
        "This is a warning with mercy, not a sentence."
    )

    parts = [
        _ensure_terminal_punct(opening),
        " ".join(symbol_sentences).strip() if symbol_sentences else "",
        " ".join(effect_sentences).strip() if effect_sentences else "",
        action_text.strip() if action_text else "",
        _ensure_terminal_punct(closing),
    ]
    return "\n\n".join([_clean_sentence(p).strip() for p in parts if p and p.strip()]).strip()


# ----------------------------
# Admin auth + HTML
# ----------------------------
def _get_admin_key_from_request() -> str:
    q = (request.args.get("key") or "").strip()
    h = (request.headers.get("X-Admin-Key") or "").strip()
    return q or h


def _require_admin() -> Optional[Response]:
    if not ADMIN_KEY:
        return make_response("Admin is not configured (missing ADMIN_KEY / JTS_ADMIN_KEY).", 403)

    provided = _get_admin_key_from_request()
    if not provided or provided != ADMIN_KEY:
        return make_response("Forbidden", 403)
    return None


ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JTS Admin Panel</title>
  <style>
    :root{--bg:#0b0b0e;--panel:#121218;--ink:#e9e9ef;--muted:#b7b7c2;--gold:#d4af37;--gold2:#b8921e;--edge:#2b2730;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:22px;}
    .wrap{max-width:860px;margin:0 auto;}
    .card{background:linear-gradient(180deg, rgba(212,175,55,.05), transparent 160px), var(--panel);
      border:1px solid rgba(212,175,55,.18);border-radius:18px; padding:18px 16px; box-shadow:0 10px 30px rgba(0,0,0,.45);}
    h1{margin:0 0 8px;font-size:18px}
    .mini{color:var(--muted);font-size:13px;margin-bottom:14px}
    label{display:block;margin-top:10px;margin-bottom:6px;color:var(--muted);font-size:13px}
    input, textarea, select{
      width:100%; background:#0e0e14; color:var(--ink);
      border:1px solid rgba(212,175,55,.2); border-radius:12px;
      padding:10px 12px; outline:none;
    }
    textarea{min-height:74px; resize:vertical}
    .btn{
      margin-top:14px; width:100%;
      border:none; border-radius:999px; padding:12px 16px; font-weight:700; cursor:pointer;
      background:linear-gradient(180deg, var(--gold), var(--gold2)); color:#0b0b0e;
    }
    pre{white-space:pre-wrap;background:rgba(0,0,0,.35);padding:12px;border-radius:12px;border:1px solid rgba(212,175,55,.16);margin-top:12px}
    a{color:var(--gold)}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>JTS Admin Panel</h1>
    <div class="mini">Keep this link private. Health: <a href="/health" target="_blank">/health</a></div>

    <label>Mode</label>
    <select id="mode">
      <option value="upsert">upsert (update if exists)</option>
      <option value="add">add (always append)</option>
    </select>

    <label>Input (symbol phrase)</label>
    <input id="input" placeholder="e.g., teeth falling out" />

    <label>Spiritual Meaning</label>
    <textarea id="spiritual"></textarea>

    <label>Physical Effects</label>
    <textarea id="effects"></textarea>

    <label>What to Do (Action)</label>
    <textarea id="action"></textarea>

    <label>Keywords (comma-separated)</label>
    <textarea id="keywords" placeholder="e.g., teeth, falling, mouth"></textarea>

    <button class="btn" id="save">Save Symbol</button>

    <pre id="out">{}</pre>
  </div>
</div>

<script>
  const out = document.getElementById('out');
  const saveBtn = document.getElementById('save');

  function getKeyFromUrl(){
    const u = new URL(window.location.href);
    return u.searchParams.get('key') || '';
  }

  saveBtn.addEventListener('click', async () => {
    out.textContent = 'Saving...';
    const payload = {
      mode: document.getElementById('mode').value,
      input: document.getElementById('input').value.trim(),
      spiritual_meaning: document.getElementById('spiritual').value.trim(),
      physical_effects: document.getElementById('effects').value.trim(),
      action: document.getElementById('action').value.trim(),
      keywords: document.getElementById('keywords').value.trim(),
    };
    const key = getKeyFromUrl();
    const res = await fetch('/admin/upsert?key=' + encodeURIComponent(key), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({error:'Bad JSON'}));
    out.textContent = JSON.stringify(data, null, 2);
  });
</script>
</body>
</html>
"""


def _build_col_map(header_row: List[str]) -> Dict[str, int]:
    col_map: Dict[str, int] = {}
    for idx, h in enumerate(header_row, start=1):
        col_map[_normalize_header(h)] = idx
    return col_map


def _find_existing_row_index(ws, input_value: str, input_col: int) -> Optional[int]:
    target = _normalize_text(input_value)
    if not target:
        return None

    col_vals = ws.col_values(input_col)
    for i, v in enumerate(col_vals[1:], start=2):
        if _normalize_text(v) == target:
            return i
    return None


def _admin_upsert_to_sheet(payload: Dict[str, Any]) -> Dict[str, Any]:
    ws = _get_ws()
    header_row = ws.row_values(1)
    if not header_row:
        raise RuntimeError("Sheet has no header row.")

    col_map = _build_col_map(header_row)

    input_col = col_map.get("input") or col_map.get("symbol")
    spiritual_col = col_map.get("spiritual meaning") or col_map.get("spiritual_meaning") or col_map.get("spiritual")
    effects_col = col_map.get("physical effects") or col_map.get("physical_effects") or col_map.get("effects in the physical realm")
    action_col = col_map.get("action") or col_map.get("what to do") or col_map.get("what_to_do")
    keywords_col = col_map.get("keywords")

    if not input_col:
        raise RuntimeError("Missing 'input' column in header row.")

    mode = (payload.get("mode") or "upsert").strip().lower()
    input_value = (payload.get("input") or "").strip()
    if not input_value:
        return {"ok": False, "error": "Missing input"}

    spiritual = (payload.get("spiritual_meaning") or payload.get("spiritual") or "").strip()
    effects = (payload.get("physical_effects") or payload.get("effects") or "").strip()
    action = (payload.get("action") or "").strip()
    keywords = (payload.get("keywords") or "").strip()

    existing_row = None
    if mode != "add":
        existing_row = _find_existing_row_index(ws, input_value, input_col)

    if existing_row:
        row_index = existing_row
        op = "updated"
    else:
        row_index = len(ws.get_all_values()) + 1
        op = "added"

    updates = [(row_index, input_col, input_value)]
    if spiritual_col:
        updates.append((row_index, spiritual_col, spiritual))
    if effects_col:
        updates.append((row_index, effects_col, effects))
    if action_col:
        updates.append((row_index, action_col, action))
    if keywords_col:
        updates.append((row_index, keywords_col, keywords))

    cell_list = [gspread.Cell(r, c, v) for r, c, v in updates]
    ws.update_cells(cell_list, value_input_option="RAW")

    _invalidate_cache()

    return {
        "ok": True,
        "action": op,
        "input": input_value,
        "written_row": row_index,
        "spreadsheet_id": SPREADSHEET_ID,
        "worksheet_name": WORKSHEET_NAME,
    }


# ----------------------------
# Upgrade HTML
# ----------------------------
def _upgrade_html_option_a() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Unlock JTS Dream Interpreter</title>
  <style>
    :root{{--bg:#07070a;--panel:#101017;--ink:#f2f2f7;--muted:#b6b6c3;--gold:#d4af37;--gold2:#b8921e;--edge:rgba(212,175,55,.22);}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:radial-gradient(900px 600px at 50% -100px, rgba(212,175,55,.14), transparent 55%), var(--bg);
      color:var(--ink);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:26px;}}
    .wrap{{max-width:980px;margin:0 auto;display:grid;grid-template-columns:1.05fr .95fr;gap:16px}}
    @media (max-width: 920px){{.wrap{{grid-template-columns:1fr}}}}
    .card{{background:linear-gradient(180deg, rgba(212,175,55,.07), transparent 180px), var(--panel);
      border:1px solid var(--edge);border-radius:18px;padding:18px;box-shadow:0 14px 50px rgba(0,0,0,.6);}}
    h1{{margin:0 0 8px;font-size:20px;letter-spacing:.2px}}
    .sub{{color:var(--muted);font-size:13px;line-height:1.4;margin-bottom:14px}}
    .pill{{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(212,175,55,.10);
      border:1px solid rgba(212,175,55,.25);color:var(--gold);font-size:12px;margin-bottom:10px}}
    ul{{margin:0;padding-left:18px;color:var(--muted);font-size:13px}}
    li{{margin:6px 0}}
    label{{display:block;margin:10px 0 6px;color:var(--muted);font-size:12px}}
    input{{width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(212,175,55,.20);background:#0b0b12;color:var(--ink);outline:none}}
    .btn{{width:100%;border:none;border-radius:999px;padding:12px 14px;font-weight:800;cursor:pointer;
      background:linear-gradient(180deg,var(--gold),var(--gold2));color:#09090c;margin-top:12px}}
    .btn2{{width:100%;border:1px solid rgba(212,175,55,.22);border-radius:999px;padding:12px 14px;font-weight:800;cursor:pointer;
      background:transparent;color:var(--ink);margin-top:10px}}
    .small{{margin-top:10px;color:var(--muted);font-size:12px}}
    .err{{margin-top:10px;color:#ffb4b4;font-size:12px;white-space:pre-wrap}}
    .ok{{margin-top:10px;color:#b9ffb9;font-size:12px;white-space:pre-wrap}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="pill">Unlock unlimited access</div>
      <h1>Full Cultural Dream Access</h1>
      <div class="sub">
        Secure checkout powered by Stripe. If you already subscribed, enter the same email and unlock instantly.
      </div>
      <ul>
        <li>Unlimited interpretations</li>
        <li>Advanced Seal of the Dream</li>
        <li>Full spiritual meaning + effects + what to do</li>
        <li>Cancel anytime</li>
      </ul>
      <div class="small">Educational & spiritual insight only. Not medical, psychological, or legal advice.</div>
    </div>

    <div class="card">
      <h1>Enter your email</h1>
      <div class="sub">Use the email you used at checkout.</div>

      <label>Email</label>
      <input id="email" type="email" placeholder="you@example.com" />

      <button class="btn2" id="unlock">Unlock Access</button>

      <form id="checkoutForm" action="/create-checkout-session" method="POST">
        <input type="hidden" name="email" id="emailHidden" value="" />
        <button class="btn" type="submit">Continue to Checkout</button>
      </form>

      <div class="err" id="err"></div>
      <div class="ok" id="ok"></div>
    </div>
  </div>

<script>
  const err = document.getElementById('err');
  const ok = document.getElementById('ok');
  const emailInput = document.getElementById('email');
  const emailHidden = document.getElementById('emailHidden');

  function getEmail(){{
    return (emailInput.value || '').trim();
  }}

  emailInput.addEventListener('input', () => {{
    emailHidden.value = getEmail();
  }});

  document.getElementById('unlock').addEventListener('click', async () => {{
    err.textContent = '';
    ok.textContent = '';
    const email = getEmail();
    if (!email || !email.includes('@')) {{
      err.textContent = 'Please enter a valid email.';
      return;
    }}

    const res = await fetch('/check-access', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{email}}),
      credentials: 'include'
    }});

    const data = await res.json().catch(() => ({{}}));
    if (!res.ok) {{
      err.textContent = data.error || data.message || 'Could not verify access.';
      return;
    }}

    ok.textContent = 'Access confirmed. Sending you back to the interpreter…';
    setTimeout(() => {{
      window.location.href = data.return_url || {json.dumps(RETURN_URL)};
    }}, 700);
  }});

  emailHidden.value = getEmail();
</script>
</body>
</html>"""


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return redirect(RETURN_URL, code=302)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "dream-interpreter",
        "sheet": WORKSHEET_NAME,
        "has_spreadsheet_id": bool(SPREADSHEET_ID),
        "allowed_origins": allowed_origins,
        "match_mode": "strict_word_boundary_with_longest_phrase_priority_and_overlap_guard",
        "debug_match": DEBUG_MATCH,
        "narrative_enabled": NARRATIVE_ENABLED,
        "narrative_max_symbols": NARRATIVE_MAX_SYMBOLS,
        "admin_configured": bool(ADMIN_KEY),
        "free_quota": FREE_TRIES,
        "shadow_window_hours": SHADOW_WINDOW_HOURS,
        "stripe_configured": bool(_stripe_config_ok()),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "counts_file": str(USAGE_COUNTS_PATH),
        "subscribers_file": str(SUBSCRIBERS_PATH),
        "price_weekly_set": bool(PRICE_WEEKLY),
        "price_monthly_set": bool(PRICE_MONTHLY),
        "cookie_samesite": COOKIE_SAMESITE,
        "return_url": RETURN_URL,
        "session_premium": _is_premium_session(),
        "session_email": _get_session_email(),
    })


@app.route("/healthz", methods=["GET"])
def healthz():
    return health()


@app.route("/upgrade", methods=["GET"])
def upgrade():
    try:
        return render_template("upgrade.html")
    except Exception:
        return Response(_upgrade_html_option_a(), mimetype="text/html")


@app.route("/subscribe", methods=["GET"])
def subscribe():
    return redirect(url_for("upgrade"), code=302)


@app.route("/logout", methods=["POST", "GET"])
def logout():
    _clear_premium_session()
    return redirect(url_for("upgrade"), code=302)


@app.route("/check-access", methods=["POST", "OPTIONS"])
def check_access():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email:
        return jsonify({"ok": False, "error": "Please enter a valid email."}), 400

    is_active, customer_id = _stripe_active_subscription_for_email(email)
    if not is_active:
        return jsonify({
            "ok": False,
            "access": "inactive",
            "message": "No active subscription found for that email. If you subscribed with a different email, use that one."
        }), 402

    _set_premium_session(email)
    _mark_subscriber(email, True, customer_id)

    resp = make_response(jsonify({"ok": True, "access": "active", "return_url": RETURN_URL}))
    resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    return resp


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not _stripe_config_ok():
        return make_response(
            "Stripe not configured. Ensure STRIPE_SECRET_KEY and PRICE_WEEKLY (or PRICE_MONTHLY) are set.",
            500,
        )

    email = ""
    if request.form and request.form.get("email"):
        email = (request.form.get("email") or "").strip().lower()
    else:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email:
        return make_response("Missing email. Please enter your email on the upgrade page.", 400)

    stripe.api_key = STRIPE_SECRET_KEY

    try:
        checkout = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": DEFAULT_STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=email,
            success_url=url_for("payment_success", _external=True) + f"?email={email}",
            cancel_url=url_for("upgrade", _external=True),
        )
        return redirect(checkout.url)
    except Exception as e:
        return make_response(f"Stripe error: {str(e)}", 500)


@app.route("/payment-success", methods=["GET"])
def payment_success():
    email = (request.args.get("email") or "").strip().lower()

    if email and "@" in email:
        is_active, customer_id = _stripe_active_subscription_for_email(email)
        if is_active:
            _set_premium_session(email)
            _mark_subscriber(email, True, customer_id)

    resp = Response(
        f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Access Active</title>
        <style>
          body{{margin:0;background:#07070a;color:#f2f2f7;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:26px}}
          .card{{max-width:520px;margin:0 auto;background:#101017;border:1px solid rgba(212,175,55,.22);border-radius:18px;padding:18px}}
          .muted{{color:#b6b6c3;font-size:13px;margin-top:8px}}
        </style></head><body>
        <div class="card">
          <h2 style="margin:0 0 8px">Access Activated</h2>
          <div class="muted">Sending you back to the interpreter…</div>
        </div>
        <script>
          setTimeout(()=>{{ window.location.href={json.dumps(RETURN_URL)}; }}, 1200);
        </script>
        </body></html>""",
        mimetype="text/html"
    )
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

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        email = (session_obj.get("customer_email") or "").strip().lower()
        customer_id = session_obj.get("customer") or ""
        if email:
            _mark_subscriber(email, True, customer_id)

    if event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub.get("customer") or ""
        subs = _load_subscribers()
        changed = False
        for em, rec in subs.items():
            if (rec or {}).get("stripe_customer_id") == customer_id and em:
                rec["is_active"] = False
                rec["updated_at"] = datetime.utcnow().isoformat() + "Z"
                subs[em] = rec
                changed = True
        if changed:
            _save_subscribers(subs)

    return ("ok", 200)


@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or data.get("text") or "").strip()
    if not dream:
        return jsonify({"error": "Missing 'dream' or 'text'"}), 400

    is_paid = _is_premium_session()

    if not is_paid:
        ip = _get_client_ip()
        cookie_used = _get_cookie_tries_used()
        ip_used = _shadow_count(ip)
        effective_used = max(cookie_used, ip_used)

        if effective_used >= FREE_TRIES:
            return jsonify({
                "blocked": True,
                "reason": "free_limit_reached",
                "message": f"You’ve used your {FREE_TRIES} free tries. Subscribe to continue.",
                "redirect": url_for("upgrade"),
                "free_uses_left": 0,
                "access": "blocked",
                "is_paid": False,
            }), 402

    try:
        rows = _load_sheet_rows()
    except Exception as e:
        return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

    matches = _match_symbols_strict(dream, rows, top_k=3)
    receipt_id = _make_receipt_id()
    seal = _compute_seal(matches)

    free_uses_left = 0
    if not is_paid:
        ip = _get_client_ip()
        cookie_used = _get_cookie_tries_used()
        ip_used = _shadow_count(ip)
        effective_used = max(cookie_used, ip_used)

        _shadow_increment(ip)
        free_uses_left = _free_tries_remaining_after_this(effective_used)

    if not matches:
        payload = {
            "access": "paid" if is_paid else "free",
            "is_paid": bool(is_paid),
            "free_uses_left": free_uses_left,
            "seal": seal,
            "receipt": {
                "id": receipt_id,
                "top_symbols": [],
                "share_phrase": "I decoded my dream on Jamaican True Stories.",
            },
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found for the exact words in this dream.",
                "effects_in_physical_realm": "Tip: use clear symbol words or phrases that exist in your Symbols sheet.",
                "what_to_do": "Add 1–2 more key symbols or phrases and try again.",
            },
            "full_interpretation": "",
        }
        resp = make_response(jsonify(payload))
    else:
        interpretation = _combine_fields(matches)
        top_symbols = [_get_symbol_cell(row) for row, _sc, _hit in matches if _get_symbol_cell(row)]
        share_phrase = f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories."
        full_interpretation = build_full_interpretation_from_doctrine(matches)

        payload = {
            "access": "paid" if is_paid else "free",
            "is_paid": bool(is_paid),
            "free_uses_left": free_uses_left,
            "seal": seal,
            "interpretation": interpretation,
            "full_interpretation": full_interpretation,
            "receipt": {
                "id": receipt_id,
                "top_symbols": top_symbols,
                "share_phrase": share_phrase,
            },
        }
        resp = make_response(jsonify(payload))

    if not is_paid:
        cookie_used = _get_cookie_tries_used()
        resp.set_cookie(
            COOKIE_NAME,
            str(cookie_used + 1),
            max_age=COOKIE_MAX_AGE,
            samesite=COOKIE_SAMESITE,
            secure=COOKIE_SECURE,
        )

    return resp


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
    })


# ----------------------------
# Admin routes
# ----------------------------
@app.route("/admin", methods=["GET"])
def admin():
    auth_fail = _require_admin()
    if auth_fail:
        return auth_fail
    return Response(ADMIN_HTML, mimetype="text/html")


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


# ----------------------------
# Debug routes
# ----------------------------
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
        "counts_file": str(USAGE_COUNTS_PATH),
        "subscribers_file": str(SUBSCRIBERS_PATH),
        "cookie_samesite": COOKIE_SAMESITE,
        "return_url": RETURN_URL,
        "session_premium": _is_premium_session(),
        "session_email": _get_session_email(),
    })


@app.route("/debug/sheet", methods=["GET"])
def debug_sheet():
    if not DEBUG_MATCH:
        return jsonify({"error": "Debug disabled"}), 403

    rows = _load_sheet_rows(force=True)
    headers = _CACHE.get("headers", [])
    sample = rows[0] if rows else {}

    return jsonify({
        "worksheet": WORKSHEET_NAME,
        "row_count": len(rows),
        "headers_seen": headers,
        "sample_keys": list(sample.keys()),
        "sample_symbol_cell": _get_symbol_cell(sample),
        "sample_spiritual_meaning": _get_spiritual_meaning_cell(sample),
        "sample_effects": _get_effects_cell(sample),
        "sample_what_to_do": _get_what_to_do_cell(sample),
        "sample_keywords": _get_keywords_cell(sample),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
