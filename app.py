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
RETURN_URL = os.getenv("RETURN_URL", "https://jamaicantruestories.com/pages/dream-interpreter").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

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

NARRATIVE_ENABLED = os.getenv("NARRATIVE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
NARRATIVE_MAX_SYMBOLS = int(os.getenv("NARRATIVE_MAX_SYMBOLS", "3"))

DOCTRINE_MODE = os.getenv("DOCTRINE_MODE", "1").strip().lower() not in {"0", "false", "no", "off"}
KEYWORD_GUARD_ENABLED = True

MAX_DREAM_LENGTH = int(os.getenv("MAX_DREAM_LENGTH", "2500"))
MIN_DREAM_LENGTH = int(os.getenv("MIN_DREAM_LENGTH", "5"))


# ============================================================
# Doctrine sheet names
# ============================================================
SHEET_BASE_SYMBOLS = os.getenv("SHEET_BASE_SYMBOLS", "BaseSymbols").strip()
SHEET_BEHAVIOR_RULES = os.getenv("SHEET_BEHAVIOR_RULES", "BehaviorRules").strip()
SHEET_SIZE_STATE_RULES = os.getenv("SHEET_SIZE_STATE_RULES", "SizeStateRules").strip()
SHEET_LOCATION_RULES = os.getenv("SHEET_LOCATION_RULES", "LocationRules").strip()
SHEET_OVERRIDE_RULES = os.getenv("SHEET_OVERRIDE_RULES", "OverrideRules").strip()
SHEET_OUTPUT_TEMPLATES = os.getenv("SHEET_OUTPUT_TEMPLATES", "OutputTemplates").strip()

DOCTRINE_SHEET_NAMES = [
SHEET_BASE_SYMBOLS,
SHEET_BEHAVIOR_RULES,
SHEET_SIZE_STATE_RULES,
SHEET_LOCATION_RULES,
SHEET_OVERRIDE_RULES,
SHEET_OUTPUT_TEMPLATES,
]


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
"you", "your"
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


def _dedupe_preserve_order(items: List[str]) -> List[str]:
seen = set()
out = []
for x in items:
x = _clean_sentence(x)
if not x:
continue
key = x.lower()
if key in seen:
continue
seen.add(key)
out.append(x)
return out


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
for key in keys:
if key in row and str(row.get(key, "")).strip():
return str(row.get(key, "")).strip()
return ""


def _validate_email(email: str) -> bool:
email = (email or "").strip()
return bool(email and "@" in email and "." in email.split("@")[-1])


def _validate_dream_text(dream: str) -> Optional[str]:
if not dream:
return "Missing 'dream' or 'text'."
if len(dream.strip()) < MIN_DREAM_LENGTH:
return f"Dream text is too short. Please enter at least {MIN_DREAM_LENGTH} characters."
if len(dream) > MAX_DREAM_LENGTH:
return f"Dream text is too long. Maximum allowed is {MAX_DREAM_LENGTH} characters."
return None


# ============================================================
# Elite formatting helpers
# ============================================================
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


def _sentence(text: str) -> str:
text = _dedupe_words_soft(text)
text = _capitalize_first(text)
return _ensure_terminal_punct(text)


def _join_phrases_natural(parts: List[str]) -> str:
cleaned = [_strip_trailing_punct(x) for x in parts if x and _strip_trailing_punct(x)]
if not cleaned:
return ""
if len(cleaned) == 1:
return cleaned[0]
if len(cleaned) == 2:
return f"{cleaned[0]} and {cleaned[1]}"
return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _clean_debug_like_phrase(text: str) -> str:
text = _clean_sentence(text)
if not text:
return ""

banned_starts = [
"logic layer",
"behavior detected",
"state detected",
"location detected",
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
"attack or manipulation": "attack or manipulation",
"enemy or deception": "pressure through deception or opposition",
}

if lowered in replacements:
return replacements[lowered]

return text


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

if lowered in elite_map:
return elite_map[lowered]

return text


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


def _elite_behavior_sentence(text: str) -> str:
text = _strip_trailing_punct(text)
if not text:
return ""
return _sentence(f"This suggests {text}")


def _elite_state_sentence(text: str) -> str:
text = _strip_trailing_punct(text)
if not text:
return ""
return _sentence(f"This indicates {text}")


def _elite_location_sentence(text: str) -> str:
text = _strip_trailing_punct(text)
if not text:
return ""
return _sentence(f"This points to {text}")


def _build_spiritual_meaning_paragraph(
symbol: str,
base_meaning: str,
behavior_mod: str,
state_mod: str,
location_mod: str,
override_spiritual: str = "",
) -> str:
symbol_name = _title_case_symbol(symbol)
core = _strip_trailing_punct(override_spiritual or base_meaning)

parts = []

if core:
parts.append(_sentence(f"{symbol_name} represents {core}"))

if behavior_mod:
parts.append(_elite_behavior_sentence(behavior_mod))

if state_mod:
parts.append(_elite_state_sentence(state_mod))

if location_mod:
parts.append(_elite_location_sentence(location_mod))

return " ".join([p for p in parts if p]).strip()


def _build_physical_effect_sentence(
base_effects: str,
behavior_phys_mod: str,
state_phys_mod: str,
location_phys_mod: str,
override_physical: str = "",
) -> str:
parts = []

if override_physical:
parts.append(_normalize_effect_phrase(override_physical))
if base_effects:
parts.append(_normalize_effect_phrase(base_effects))
if behavior_phys_mod:
parts.append(_normalize_effect_phrase(behavior_phys_mod))
if state_phys_mod:
parts.append(_normalize_effect_phrase(state_phys_mod))
if location_phys_mod:
parts.append(_normalize_effect_phrase(location_phys_mod))

joined = _join_phrases_natural(parts)
if not joined:
return "No clear physical effects were generated."

return _sentence(f"This may manifest as {joined}")


def _build_action_sentence(
base_action: str,
behavior_action_mod: str,
state_action_mod: str,
location_action_mod: str,
override_action: str = "",
) -> str:
parts = []

if override_action:
parts.append(_normalize_action_phrase(override_action))
if base_action:
parts.append(_normalize_action_phrase(base_action))
if behavior_action_mod:
parts.append(_normalize_action_phrase(behavior_action_mod))
if state_action_mod:
parts.append(_normalize_action_phrase(state_action_mod))
if location_action_mod:
parts.append(_normalize_action_phrase(location_action_mod))

cleaned_parts = []
seen = set()

for p in parts:
p = _clean_sentence(p)
if not p:
continue
key = p.lower()
if key in seen:
continue
seen.add(key)
cleaned_parts.append(p)

joined = " ".join(cleaned_parts).strip()
if not joined:
return "Pray for wisdom and confirmation."

return _sentence(joined)


def _render_template_text(template_text: str, context: Dict[str, str]) -> str:
text = template_text or ""
for key, value in context.items():
text = text.replace("{" + key + "}", _strip_trailing_punct(value or ""))
return _sentence(text)


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
# Hybrid Gate
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


# ============================================================
# Subscriber helpers
# ============================================================
def _load_subscribers() -> Dict[str, Any]:
data = _read_json_file(SUBSCRIBERS_PATH, {})
return data if isinstance(data, dict) else {}


def _save_subscribers(data: Dict[str, Any]):
_write_json_file_atomic(SUBSCRIBERS_PATH, data)


def _mark_subscriber(email: str, is_active: bool, stripe_customer_id: str = ""):
if not email:
return
email_n = email.strip().lower()
subs = _load_subscribers()
rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}
rec["email"] = email_n
rec["is_active"] = bool(is_active)
rec["stripe_customer_id"] = stripe_customer_id
rec["updated_at"] = datetime.utcnow().isoformat() + "Z"
subs[email_n] = rec
_save_subscribers(subs)


def _parse_iso_z(dt_str: str) -> Optional[datetime]:
try:
if not dt_str:
return None
return datetime.fromisoformat(dt_str.replace("Z", ""))
except Exception:
return None


def _set_buyer_session(email: str):
email_n = (email or "").strip().lower()
if not email_n:
return
session["subscriber_email"] = email_n
session["buyer_set_at"] = datetime.utcnow().isoformat() + "Z"


def _mark_dream_pack_purchase(email: str, uses: int = DREAM_PACK_USES, hours: int = DREAM_PACK_HOURS):
if not email:
return

email_n = email.strip().lower()
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
email_n = (email or "").strip().lower()
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
email_n = (email or "").strip().lower()
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
# Stripe premium session
# ============================================================
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
session.pop("buyer_set_at", None)


def _stripe_config_ok() -> bool:
return bool(stripe and STRIPE_SECRET_KEY and DEFAULT_STRIPE_PRICE_ID)


def _stripe_dream_pack_ok() -> bool:
return bool(stripe and STRIPE_SECRET_KEY and PRICE_DREAM_PACK)


def _stripe_active_subscription_for_email(email: str) -> Tuple[bool, str]:
email = (email or "").strip().lower()
if not _validate_email(email):
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


# ============================================================
# Google Sheets access
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
"Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
"or provide credentials.json via GOOGLE_SERVICE_ACCOUNT_FILE."
)
return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _get_spreadsheet():
if not SPREADSHEET_ID:
raise RuntimeError("SPREADSHEET_ID env var is not set.")
creds = _get_credentials()
gc = gspread.authorize(creds)
return gc.open_by_key(SPREADSHEET_ID)


def _worksheet_to_rows(ws) -> Tuple[List[str], List[Dict[str, str]]]:
values = ws.get_all_values()
if not values or len(values) < 1:
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
if count == 0:
headers.append(nh)
else:
headers.append(f"{nh}__{count+1}")

rows: List[Dict[str, str]] = []
for r in values[1:]:
if len(r) < len(headers):
r = r + [""] * (len(headers) - len(r))
item = {headers[i]: (r[i] or "").strip() for i in range(len(headers))}
rows.append(item)

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

sh = _get_spreadsheet()
ws = sh.worksheet(WORKSHEET_NAME)
headers, rows = _worksheet_to_rows(ws)

_LEGACY_CACHE["rows"] = rows
_LEGACY_CACHE["headers"] = headers
_LEGACY_CACHE["loaded_at"] = now
return rows


def _load_doctrine_sheets(force: bool = False) -> Dict[str, List[Dict]]:
now = time.time()
if (not force) and _DOCTRINE_CACHE["sheets"] and (now - _DOCTRINE_CACHE["loaded_at"] < CACHE_TTL_SECONDS):
return _DOCTRINE_CACHE["sheets"]

sh = _get_spreadsheet()
sheets_data: Dict[str, List[Dict]] = {}
headers_map: Dict[str, List[str]] = {}

for name in DOCTRINE_SHEET_NAMES:
try:
ws = sh.worksheet(name)
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
# Legacy field getters
# ============================================================
def _get_symbol_cell(row: Dict) -> str:
return _row_get(row, "input", "symbol", "symbols", "input symbol", "dream symbol", "symbol name")


def _get_spiritual_meaning_cell(row: Dict) -> str:
return _row_get(
row,
"spiritual meaning", "spiritual_meaning", "spiritual",
"meaning", "base spiritual meaning", "base_spiritual_meaning",
"final spiritual meaning", "final_spiritual_meaning"
)


def _get_effects_cell(row: Dict) -> str:
return _row_get(
row,
"effects in the physical realm", "effects_in_the_physical_realm",
"physical effects", "physical_effects",
"effects", "base physical effects", "base_physical_effects",
"final physical effects", "final_physical_effects"
)


def _get_what_to_do_cell(row: Dict) -> str:
return _row_get(
row,
"what to do", "what_to_do", "action", "actions",
"base action", "base_action", "final action", "final_action"
)


def _get_keywords_cell(row: Dict) -> str:
return _row_get(row, "keywords", "keyword", "tags")


# ============================================================
# Legacy matcher
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
top_k: int = 3
) -> List[Tuple[Dict, int, Optional[Dict[str, Any]]]]:
dream_norm = _normalize_text(dream)
if not dream_norm:
return []

_log("\n--- LEGACY DEBUG_MATCH ON ---")
_log("DREAM_RAW:", repr(dream))
_log("DREAM_NORM:", repr(dream_norm))

candidates: List[Tuple[Dict, int, Optional[Dict[str, Any]]]] = []
for row in rows:
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
_log("LEGACY MATCHES:")
for row, sc, hit in out:
_log(f" - {_get_symbol_cell(row)!r} score={sc} via={hit}")
else:
_log("LEGACY MATCHES: (none)")

_log("USED_SPANS:", used_spans)
_log("--- END LEGACY DEBUG_MATCH ---\n")
return out


# ============================================================
# Doctrine field getters
# ============================================================
def _row_is_active(row: Dict) -> bool:
active = _row_get(row, "active")
if not active:
return True
return _normalize_yes_no(active)


def _get_base_symbol_input(row: Dict) -> str:
return _row_get(row, "input", "symbol")


def _get_base_symbol_category(row: Dict) -> str:
return _row_get(row, "category") or "unknown"


def _get_base_symbol_meaning(row: Dict) -> str:
return _row_get(
row,
"base_spiritual_meaning", "base spiritual meaning",
"spiritual meaning", "meaning"
)


def _get_base_symbol_effects(row: Dict) -> str:
return _row_get(
row,
"base_physical_effects", "base physical effects",
"physical effects", "effects"
)


def _get_base_symbol_action(row: Dict) -> str:
return _row_get(
row,
"base_action", "base action", "action", "actions"
)


def _get_rule_name(row: Dict, *keys: str) -> str:
return _row_get(row, *keys)


def _get_rule_keywords(row: Dict) -> List[str]:
return _split_keywords(_row_get(row, "keywords", "keyword", "tags"))


def _get_behavior_name(row: Dict) -> str:
return _get_rule_name(row, "behavior_name", "behavior")


def _get_behavior_meaning_modifier(row: Dict) -> str:
return _row_get(
row,
"meaning_modifier", "meaning modifier",
"maning_modifier", "maning modifier",
"effect", "effects"
)


def _get_behavior_physical_modifier(row: Dict) -> str:
return _row_get(
row,
"physical_modifier", "physical modifier",
"physicare_modifier", "physicare modifier",
"physical_effect", "physical effects",
"effect", "effects"
)


def _get_behavior_action_modifier(row: Dict) -> str:
return _row_get(row, "action_modifier", "action modifier", "action", "actions")


def _get_state_name(row: Dict) -> str:
return _get_rule_name(row, "state_name", "state")


def _get_state_meaning_modifier(row: Dict) -> str:
return _row_get(row, "meaning_modifier", "meaning modifier", "effect", "effects")


def _get_state_physical_modifier(row: Dict) -> str:
return _row_get(
row,
"physical_modifier", "physical modifier",
"physical_effect", "physical effects",
"effect", "effects"
)


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
# Doctrine engine detection
# ============================================================
def _match_base_symbols_doctrine(
dream: str,
base_rows: List[Dict],
top_k: int = 3
) -> List[Tuple[Dict, int, Dict[str, Any]]]:
dream_norm = _normalize_text(dream)
if not dream_norm:
return []

_log("\n--- DOCTRINE BASE SYMBOL MATCH ---")
_log("DREAM_RAW:", repr(dream))
_log("DREAM_NORM:", repr(dream_norm))

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
local_candidates: List[Tuple[str, str, int]] = []

if len(_tokenize_words(symbol)) > 1:
local_candidates.append(("symbol_phrase", symbol, 100 - _symbol_length_penalty(symbol_raw)))
else:
local_candidates.append(("symbol", symbol, 100))

for kw in keywords:
if len(_tokenize_words(kw)) > 1:
local_candidates.append(("keyword_phrase", kw, 94))
else:
local_candidates.append(("keyword", kw, 90))

local_candidates.sort(key=lambda x: (-len(x[1]), -x[2]))

for match_type, token, score in local_candidates:
spans = _find_phrase_spans(dream_norm, token)
if not spans:
continue
candidates.append((row, score, {
"type": match_type,
"token": token,
"span": spans[0],
"token_len": len(token),
}))
break

candidates.sort(
key=lambda item: (
-int(item[2].get("token_len", 0)),
-item[1],
-len(_normalize_text(_get_base_symbol_input(item[0])))
)
)

out: List[Tuple[Dict, int, Dict[str, Any]]] = []
used_spans: List[Tuple[int, int]] = []
seen_symbols = set()

for row, score, hit in candidates:
span = hit.get("span")
if not span:
continue

sym_key = _normalize_text(_get_base_symbol_input(row))
if not sym_key or sym_key in seen_symbols:
continue
if _is_span_blocked(span, used_spans):
continue

used_spans.append(span)
seen_symbols.add(sym_key)
out.append((row, score, hit))

if len(out) >= top_k:
break

if out:
_log("DOCTRINE BASE MATCHES:")
for row, sc, hit in out:
_log(f" - {_get_base_symbol_input(row)!r} score={sc} via={hit}")
else:
_log("DOCTRINE BASE MATCHES: (none)")
_log("--- END DOCTRINE BASE MATCH ---\n")

return out


def _detect_rule_hits(
dream: str,
rows: List[Dict],
kind: str,
priority_field: str = "priority"
) -> List[Dict[str, Any]]:
dream_norm = _normalize_text(dream)
hits: List[Dict[str, Any]] = []
seen_names = set()

for row in rows:
if not _row_is_active(row):
continue

if kind == "behavior":
name = _get_behavior_name(row)
elif kind == "state":
name = _get_state_name(row)
elif kind == "location":
name = _get_location_name(row)
else:
name = ""

if not name:
continue

rule_keywords = _get_rule_keywords(row)
if not rule_keywords:
continue

matched_token = ""
token_len = 0
for kw in sorted(rule_keywords, key=len, reverse=True):
if _contains_phrase(dream_norm, kw):
matched_token = kw
token_len = len(kw)
break

if not matched_token:
continue

name_key = _normalize_text(name)
if name_key in seen_names:
continue
seen_names.add(name_key)

try:
priority_val = int(str(_row_get(row, priority_field) or "0").strip())
except Exception:
priority_val = 0

hits.append({
"name": name,
"row": row,
"matched_token": matched_token,
"token_len": token_len,
"priority": priority_val,
"kind": kind,
})

hits.sort(key=lambda x: (-x["token_len"], -x["priority"], x["name"].lower()))
return hits


# ============================================================
# Doctrine overrides
# ============================================================
def _parse_simple_condition(condition: str) -> List[str]:
return [_normalize_text(x) for x in re.split(r"\+", condition or "") if _normalize_text(x)]


def _apply_override_rules(
base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
behaviors: List[Dict[str, Any]],
states: List[Dict[str, Any]],
locations: List[Dict[str, Any]],
dream: str,
override_rows: List[Dict],
) -> Optional[Dict[str, Any]]:
symbol_names = [_get_base_symbol_input(row) for row, _sc, _hit in base_matches]
behavior_names = [x["name"] for x in behaviors]
state_names = [x["name"] for x in states]
location_names = [x["name"] for x in locations]

combined = set()
combined.update({_normalize_text(x) for x in symbol_names if x})
combined.update({_normalize_text(x) for x in behavior_names if x})
combined.update({_normalize_text(x) for x in state_names if x})
combined.update({_normalize_text(x) for x in location_names if x})

dream_norm = _normalize_text(dream)
best: Optional[Dict[str, Any]] = None
best_score = -10**9

for row in override_rows:
if not _row_is_active(row):
continue

condition = _row_get(row, "condition")
if not condition:
continue

tokens = _parse_simple_condition(condition)
if not tokens:
continue

matched = True
specificity = 0

for token in tokens:
if token in combined:
specificity += 3
continue
if _contains_phrase(dream_norm, token):
specificity += 2
continue
matched = False
break

if not matched:
continue

try:
priority = int(str(_row_get(row, "priority") or "0").strip())
except Exception:
priority = 0

score = (priority * 100) + specificity + len(tokens)

if score > best_score:
best_score = score
best = row

if not best:
return None

return {
"override_name": _row_get(best, "override_name", "condition"),
"spiritual": _row_get(best, "final_spiritual_meaning", "override_effect", "effect", "effects"),
"physical": _row_get(best, "final_physical_effects"),
"action": _row_get(best, "final_action"),
"priority": int(str(_row_get(best, "priority") or "0").strip() or "0"),
"row": best,
}


# ============================================================
# Template selection
# ============================================================
def _choose_template_type(
override_hit: Optional[Dict[str, Any]],
behaviors: List[Dict[str, Any]],
states: List[Dict[str, Any]],
locations: List[Dict[str, Any]],
interpretation: Dict[str, str],
) -> str:
if override_hit:
name = _normalize_text(override_hit.get("override_name", ""))
spiritual = _normalize_text(override_hit.get("spiritual", ""))
combined = f"{name} {spiritual}"

if any(x in combined for x in ["death", "grave", "falling down", "teeth falling out", "death omen"]):
return "death_omen"
if any(x in combined for x in ["monitor", "watch", "spy"]):
return "monitoring"
if any(x in combined for x in ["escape", "deliver", "freedom", "victory"]):
return "deliverance"
if any(x in combined for x in ["clean", "wash", "purif", "release"]):
return "cleansing"
if any(x in combined for x in ["promotion", "elevation", "lifted", "favor"]):
return "promotion"
