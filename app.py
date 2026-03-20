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
if any(x in combined for x in ["warning", "danger", "attack", "disgrace"]):
return "warning"

behavior_names = {_normalize_text(x["name"]) for x in behaviors}
state_names = {_normalize_text(x["name"]) for x in states}
location_names = {_normalize_text(x["name"]) for x in locations}
interp_text = _normalize_text(" ".join(interpretation.values()))

if {"being attacked", "attacking", "being chased", "chasing", "fighting"} & behavior_names:
return "warfare"
if {"escaping", "crossing", "finding"} & behavior_names:
return "breakthrough"
if {"dirty", "murky", "broken", "bleeding", "dark"} & state_names:
return "warning"
if {"graveyard", "prison", "darkness"} & location_names:
return "warning"
if any(x in interp_text for x in ["monitoring spirit", "being watched", "spiritual spy"]):
return "monitoring"
if any(x in interp_text for x in ["cleansing", "washed", "release"]):
return "cleansing"

return "default"


# ============================================================
# Doctrine interpretation builder
# ============================================================
def _join_nonempty(parts: List[str], sep: str = " ") -> str:
return sep.join([_strip_trailing_punct(x) for x in parts if x and _strip_trailing_punct(x)]).strip()


def _build_doctrine_interpretation(
dream: str,
base_matches: List[Tuple[Dict, int, Dict[str, Any]]],
behaviors: List[Dict[str, Any]],
states: List[Dict[str, Any]],
locations: List[Dict[str, Any]],
override_hit: Optional[Dict[str, Any]],
templates: List[Dict],
) -> Dict[str, Any]:
top_symbols = [_get_base_symbol_input(row) for row, _sc, _hit in base_matches]

behavior_mod = _join_nonempty([_get_behavior_meaning_modifier(x["row"]) for x in behaviors], " ")
behavior_phys_mod = _join_nonempty([_get_behavior_physical_modifier(x["row"]) for x in behaviors], " ")
behavior_action_mod = _join_nonempty([_get_behavior_action_modifier(x["row"]) for x in behaviors], " ")

state_mod = _join_nonempty([_get_state_meaning_modifier(x["row"]) for x in states], " ")
state_phys_mod = _join_nonempty([_get_state_physical_modifier(x["row"]) for x in states], " ")
state_action_mod = _join_nonempty([_get_state_action_modifier(x["row"]) for x in states], " ")

location_mod = _join_nonempty([_get_location_life_area_meaning(x["row"]) for x in locations], " ")
location_phys_mod = _join_nonempty([_get_location_physical_area_meaning(x["row"]) for x in locations], " ")
location_action_mod = _join_nonempty([_get_location_action_modifier(x["row"]) for x in locations], " ")

override_spiritual = _strip_trailing_punct((override_hit or {}).get("spiritual", ""))
override_physical = _strip_trailing_punct((override_hit or {}).get("physical", ""))
override_action = _strip_trailing_punct((override_hit or {}).get("action", ""))

spiritual_lines: List[str] = []
physical_lines: List[str] = []
action_lines: List[str] = []

if override_hit and not base_matches:
spiritual_text = _sentence(override_spiritual or "A special doctrine condition applies to this dream.")
physical_text = _build_physical_effect_sentence(
base_effects="",
behavior_phys_mod=behavior_phys_mod,
state_phys_mod=state_phys_mod,
location_phys_mod=location_phys_mod,
override_physical=override_physical,
)
action_text = _build_action_sentence(
base_action="",
behavior_action_mod=behavior_action_mod,
state_action_mod=state_action_mod,
location_action_mod=location_action_mod,
override_action=override_action,
)
else:
for row, _sc, _hit in base_matches[:NARRATIVE_MAX_SYMBOLS]:
symbol = _get_base_symbol_input(row)
base_meaning = _get_base_symbol_meaning(row)
base_effects = _get_base_symbol_effects(row)
base_action = _get_base_symbol_action(row)

spiritual_text_i = _build_spiritual_meaning_paragraph(
symbol=symbol,
base_meaning=base_meaning,
behavior_mod=behavior_mod,
state_mod=state_mod,
location_mod=location_mod,
override_spiritual=override_spiritual,
)
physical_text_i = _build_physical_effect_sentence(
base_effects=base_effects,
behavior_phys_mod=behavior_phys_mod,
state_phys_mod=state_phys_mod,
location_phys_mod=location_phys_mod,
override_physical=override_physical,
)
action_text_i = _build_action_sentence(
base_action=base_action,
behavior_action_mod=behavior_action_mod,
state_action_mod=state_action_mod,
location_action_mod=location_action_mod,
override_action=override_action,
)

if spiritual_text_i:
spiritual_lines.append(spiritual_text_i)
if physical_text_i:
physical_lines.append(physical_text_i)
if action_text_i:
action_lines.append(action_text_i)

spiritual_text = _merge_natural_paragraphs(spiritual_lines) or "No clear spiritual meaning was generated."
physical_text = _merge_natural_paragraphs(physical_lines) or "No clear physical effects were generated."
action_text = _merge_natural_paragraphs(action_lines) or "Pray for wisdom and confirmation."

interpretation = {
"spiritual_meaning": spiritual_text,
"effects_in_physical_realm": physical_text,
"what_to_do": action_text,
}

template_type = _choose_template_type(
override_hit=override_hit,
behaviors=behaviors,
states=states,
locations=locations,
interpretation=interpretation,
)

opening_tpl = _get_output_template(
templates,
"opening",
"This dream is revealing a spiritual condition that requires discernment, not panic."
)
closing_tpl = _get_output_template(
templates,
"closing",
"Dreams expose what needs attention so you can respond with wisdom, prayer, and obedience."
)
main_tpl = _get_output_template(
templates,
template_type,
"{symbol} represents {meaning}. The behavior shows {behavior_effect}. The condition shows {state_effect}. The location points to {location_effect}."
)

context = {
"symbol": ", ".join(top_symbols[:1]) if top_symbols else "This dream",
"meaning": interpretation["spiritual_meaning"],
"behavior_effect": _join_nonempty([_get_behavior_meaning_modifier(x["row"]) for x in behaviors], ", "),
"state_effect": _join_nonempty([_get_state_meaning_modifier(x["row"]) for x in states], ", "),
"location_effect": _join_nonempty([_get_location_life_area_meaning(x["row"]) for x in locations], ", "),
}

rendered_main = _render_template_text(main_tpl, context)

full_parts = [
_sentence(opening_tpl),
rendered_main,
interpretation["effects_in_physical_realm"],
interpretation["what_to_do"],
_sentence(closing_tpl),
]
full_interpretation = "\n\n".join([p for p in full_parts if p and p.strip()])

return {
"interpretation": interpretation,
"full_interpretation": full_interpretation,
"top_symbols": top_symbols,
"override_applied": bool(override_hit),
"override_name": (override_hit or {}).get("override_name", ""),
"template_type": template_type,
}


def _build_legacy_interpretation(
matches: List[Tuple[Dict, int, Optional[Dict[str, Any]]]]
) -> Dict[str, str]:
spiritual_lines: List[str] = []
physical_lines: List[str] = []
action_lines: List[str] = []

for row, _sc, _hit in matches[:NARRATIVE_MAX_SYMBOLS]:
symbol = _get_symbol_cell(row)
base_meaning = _get_spiritual_meaning_cell(row)
base_effects = _get_effects_cell(row)
base_action = _get_what_to_do_cell(row)

spiritual_text = _build_spiritual_meaning_paragraph(
symbol=symbol,
base_meaning=base_meaning,
behavior_mod="",
state_mod="",
location_mod="",
override_spiritual="",
)
physical_text = _build_physical_effect_sentence(
base_effects=base_effects,
behavior_phys_mod="",
state_phys_mod="",
location_phys_mod="",
override_physical="",
)
action_text = _build_action_sentence(
base_action=base_action,
behavior_action_mod="",
state_action_mod="",
location_action_mod="",
override_action="",
)

if spiritual_text:
spiritual_lines.append(spiritual_text)
if physical_text:
physical_lines.append(physical_text)
if action_text:
action_lines.append(action_text)

return {
"spiritual_meaning": _merge_natural_paragraphs(spiritual_lines) or "No clear spiritual meaning was generated.",
"effects_in_physical_realm": _merge_natural_paragraphs(physical_lines) or "No clear physical effects were generated.",
"what_to_do": _merge_natural_paragraphs(action_lines) or "Pray for wisdom and confirmation.",
}


def _compute_seal_from_symbol_count(symbol_count: int) -> Dict[str, str]:
if symbol_count <= 0:
return {"status": "Delayed", "type": "Unclear", "risk": "High"}
if symbol_count == 1:
return {"status": "Live", "type": "Confirmed", "risk": "Low"}
if symbol_count == 2:
return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
return {"status": "Delayed", "type": "Processing", "risk": "Medium"}


# ============================================================
# Admin HTML
# ============================================================
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
.wrap{max-width:980px;margin:0 auto;}
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

<label>Target Sheet</label>
<select id="target_sheet">
<option value="legacy">Legacy Sheet</option>
<option value="BaseSymbols">BaseSymbols</option>
<option value="BehaviorRules">BehaviorRules</option>
<option value="SizeStateRules">SizeStateRules</option>
<option value="LocationRules">LocationRules</option>
<option value="OverrideRules">OverrideRules</option>
<option value="OutputTemplates">OutputTemplates</option>
</select>

<label>Mode</label>
<select id="mode">
<option value="upsert">upsert (update if exists)</option>
<option value="add">add (always append)</option>
</select>

<label>Payload JSON</label>
<textarea id="payload" placeholder='{"symbol":"teeth falling out","category":"body","base_spiritual_meaning":"warning","base_physical_effects":"loss","base_action":"pray immediately","keywords":"teeth falling out, teeth, mouth","active":"yes"}'></textarea>

<button class="btn" id="save">Save Row</button>
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

let payload = {};
try {
payload = JSON.parse(document.getElementById('payload').value || '{}');
} catch (e) {
out.textContent = JSON.stringify({ok:false,error:'Invalid JSON payload'}, null, 2);
return;
}

payload.mode = document.getElementById('mode').value;
payload.target_sheet = document.getElementById('target_sheet').value;

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


# ============================================================
# Admin write helpers
# ============================================================
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
if sheet_name == SHEET_OVERRIDE_RULES:
return ["override_name", "condition"]
if sheet_name == SHEET_OUTPUT_TEMPLATES:
return ["template_type"]
return ["input", "symbol"]


def _admin_upsert_generic_sheet(sheet_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
sh = _get_spreadsheet()
ws = sh.worksheet(sheet_name)
header_row = ws.row_values(1)
if not header_row:
raise RuntimeError(f"Sheet {sheet_name} has no header row.")

col_map = _build_col_map(header_row)
mode = (payload.get("mode") or "upsert").strip().lower()

lookup_col = None
lookup_value = ""

for key in _sheet_primary_lookup_headers(sheet_name):
if col_map.get(_normalize_header(key)) and str(payload.get(key, "")).strip():
lookup_col = col_map[_normalize_header(key)]
lookup_value = str(payload.get(key, "")).strip()
break

if not lookup_col or not lookup_value:
raise RuntimeError(f"Missing primary lookup field for sheet {sheet_name}.")

existing_row = None
if mode != "add":
existing_row = _find_existing_row_index_by_column(ws, lookup_value, lookup_col)

if existing_row:
row_index = existing_row
op = "updated"
else:
row_index = len(ws.get_all_values()) + 1
op = "added"

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

if nh == "keywords":
chosen_value = _sanitize_keywords_for_storage(str(chosen_value))
else:
chosen_value = str(chosen_value).strip()

col_index = col_map.get(nh)
if col_index:
updates.append((row_index, col_index, chosen_value))

if not updates:
raise RuntimeError("No matching payload fields found for target sheet headers.")

cell_list = [gspread.Cell(r, c, v) for r, c, v in updates]
ws.update_cells(cell_list, value_input_option="RAW")

_invalidate_all_caches()

return {
"ok": True,
"action": op,
"written_row": row_index,
"spreadsheet_id": SPREADSHEET_ID,
"worksheet_name": sheet_name,
}


def _admin_upsert_legacy_sheet(payload: Dict[str, Any]) -> Dict[str, Any]:
sh = _get_spreadsheet()
ws = sh.worksheet(WORKSHEET_NAME)
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
input_value = (payload.get("input") or payload.get("symbol") or "").strip()
if not input_value:
return {"ok": False, "error": "Missing input"}

spiritual = (payload.get("spiritual_meaning") or payload.get("spiritual") or "").strip()
effects = (payload.get("physical_effects") or payload.get("effects") or "").strip()
action = (payload.get("action") or "").strip()
keywords = _sanitize_keywords_for_storage((payload.get("keywords") or "").strip())

existing_row = None
if mode != "add":
existing_row = _find_existing_row_index_by_column(ws, input_value, input_col)

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

_invalidate_all_caches()

return {
"ok": True,
"action": op,
"input": input_value,
"written_row": row_index,
"spreadsheet_id": SPREADSHEET_ID,
"worksheet_name": WORKSHEET_NAME,
}


def _admin_upsert_to_sheet(payload: Dict[str, Any]) -> Dict[str, Any]:
target_sheet = (payload.get("target_sheet") or "legacy").strip()

if target_sheet == "legacy":
return _admin_upsert_legacy_sheet(payload)

sheet_alias_map = {
"BaseSymbols": SHEET_BASE_SYMBOLS,
"BehaviorRules": SHEET_BEHAVIOR_RULES,
"SizeStateRules": SHEET_SIZE_STATE_RULES,
"LocationRules": SHEET_LOCATION_RULES,
"OverrideRules": SHEET_OVERRIDE_RULES,
"OutputTemplates": SHEET_OUTPUT_TEMPLATES,
}

real_sheet_name = sheet_alias_map.get(target_sheet, target_sheet)
return _admin_upsert_generic_sheet(real_sheet_name, payload)


# ============================================================
# Upgrade HTML
# ============================================================
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
.wrap{{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1.1fr .9fr;gap:16px}}
@media (max-width: 920px){{.wrap{{grid-template-columns:1fr}}}}
.card{{background:linear-gradient(180deg, rgba(212,175,55,.07), transparent 180px), var(--panel);
border:1px solid var(--edge);border-radius:18px;padding:18px;box-shadow:0 14px 50px rgba(0,0,0,.6);}}
h1{{margin:0 0 8px;font-size:20px;letter-spacing:.2px}}
h2{{margin:0 0 8px;font-size:18px;letter-spacing:.2px}}
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
.stack{{display:grid;gap:14px}}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<div class="pill">Unlock dream access</div>
<h1>JTS Dream Interpreter</h1>
<div class="sub">
Choose the option that fits you best. Use the same email at checkout and later for access verification.
</div>
<ul>
<li>Subscription: ongoing access</li>
<li>Dream Pack: {DREAM_PACK_USES} dream interpretations for a one-time purchase</li>
<li>Secure checkout powered by Stripe</li>
</ul>
<div class="small">Educational and spiritual insight only. Not medical, psychological, legal, or financial advice.</div>
</div>

<div class="stack">
<div class="card">
<h2>Enter your email</h2>
<div class="sub">Use the same email for unlocking after payment.</div>

<label>Email</label>
<input id="email" type="email" placeholder="you@example.com" />

<button class="btn2" id="unlock">Unlock Existing Access</button>

<form id="checkoutForm" action="/create-checkout-session" method="POST">
<input type="hidden" name="email" id="emailHidden" value="" />
<button class="btn" type="submit">Start Subscription</button>
</form>

<form id="dreamPackForm" action="/create-dream-pack-checkout-session" method="POST">
<input type="hidden" name="email" id="emailHiddenPack" value="" />
<button class="btn2" type="submit">Buy {DREAM_PACK_USES} Dreams</button>
</form>

<div class="err" id="err"></div>
<div class="ok" id="ok"></div>
</div>
</div>
</div>

<script>
const err = document.getElementById('err');
const ok = document.getElementById('ok');
const emailInput = document.getElementById('email');
const emailHidden = document.getElementById('emailHidden');
const emailHiddenPack = document.getElementById('emailHiddenPack');

function getEmail(){{
return (emailInput.value || '').trim();
}}

function syncEmail(){{
const value = getEmail();
emailHidden.value = value;
emailHiddenPack.value = value;
}}

emailInput.addEventListener('input', syncEmail);
syncEmail();

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
</script>
</body>
</html>"""


# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET"])
def home():
return redirect(RETURN_URL, code=302)


@app.route("/health", methods=["GET"])
def health():
doctrine_available = False
try:
doctrine_available = _doctrine_sheets_available()
except Exception:
doctrine_available = False

return jsonify({
"ok": True,
"service": "dream-interpreter",
"sheet": WORKSHEET_NAME,
"has_spreadsheet_id": bool(SPREADSHEET_ID),
"allowed_origins": allowed_origins,
"match_mode": "strict_word_boundary_with_longest_phrase_priority_and_overlap_guard",
"brain_layers": [
"symbol_logic",
"behavior_logic",
"state_logic",
"location_logic",
"override_logic",
"template_logic",
],
"debug_match": DEBUG_MATCH,
"narrative_enabled": NARRATIVE_ENABLED,
"narrative_max_symbols": NARRATIVE_MAX_SYMBOLS,
"keyword_guard_enabled": KEYWORD_GUARD_ENABLED,
"doctrine_mode_enabled": DOCTRINE_MODE,
"doctrine_sheets_available": doctrine_available,
"doctrine_sheet_names": DOCTRINE_SHEET_NAMES,
"admin_configured": bool(ADMIN_KEY),
"free_quota": FREE_TRIES,
"shadow_window_hours": SHADOW_WINDOW_HOURS,
"stripe_configured": bool(_stripe_config_ok()),
"webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
"counts_file": str(USAGE_COUNTS_PATH),
"subscribers_file": str(SUBSCRIBERS_PATH),
"price_weekly_set": bool(PRICE_WEEKLY),
"price_monthly_set": bool(PRICE_MONTHLY),
"price_dream_pack_set": bool(PRICE_DREAM_PACK),
"dream_pack_uses": DREAM_PACK_USES,
"dream_pack_hours": DREAM_PACK_HOURS,
"cookie_samesite": COOKIE_SAMESITE,
"return_url": RETURN_URL,
"session_premium": _is_premium_session(),
"session_email": _get_session_email(),
"session_dream_pack": _get_dream_pack_status(_get_session_email()),
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

if not _validate_email(email):
return jsonify({"ok": False, "error": "Please enter a valid email."}), 400

is_active, customer_id = _stripe_active_subscription_for_email(email)
if is_active:
_set_premium_session(email)
_mark_subscriber(email, True, customer_id)

resp = make_response(jsonify({
"ok": True,
"access": "active",
"return_url": RETURN_URL,
}))
resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
return resp

dream_pack_status = _get_dream_pack_status(email)
if dream_pack_status["active"]:
_set_buyer_session(email)
resp = make_response(jsonify({
"ok": True,
"access": "dream_pack_active",
"return_url": RETURN_URL,
"dream_pack": dream_pack_status,
}))
resp.set_cookie(COOKIE_NAME, "0", max_age=0, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
return resp

return jsonify({
"ok": False,
"access": "inactive",
"message": "No active subscription or dream pack found for that email."
}), 402


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
if not _stripe_config_ok():
return make_response(
"Stripe not configured. Ensure STRIPE_SECRET_KEY and PRICE_WEEKLY or PRICE_MONTHLY are set.",
500,
)

email = ""
if request.form and request.form.get("email"):
email = (request.form.get("email") or "").strip().lower()
else:
data = request.get_json(silent=True) or {}
email = (data.get("email") or "").strip().lower()

if not _validate_email(email):
return make_response("Missing email. Please enter a valid email on the upgrade page.", 400)

stripe.api_key = STRIPE_SECRET_KEY

try:
checkout = stripe.checkout.Session.create(
mode="subscription",
line_items=[{"price": DEFAULT_STRIPE_PRICE_ID, "quantity": 1}],
customer_email=email,
metadata={
"purchase_type": "subscription",
"email": email,
},
success_url=url_for("payment_success", _external=True) + f"?email={email}",
cancel_url=url_for("upgrade", _external=True),
)
return redirect(checkout.url, code=303)
except Exception as e:
return make_response(f"Stripe error: {str(e)}", 500)


@app.route("/create-dream-pack-checkout-session", methods=["POST"])
def create_dream_pack_checkout_session():
if not _stripe_dream_pack_ok():
return make_response(
"Dream pack Stripe config missing. Ensure STRIPE_SECRET_KEY and PRICE_DREAM_PACK are set.",
500,
)

email = ""
if request.form and request.form.get("email"):
email = (request.form.get("email") or "").strip().lower()
else:
data = request.get_json(silent=True) or {}
email = (data.get("email") or "").strip().lower()

if not _validate_email(email):
return make_response("Missing email for dream pack checkout.", 400)

stripe.api_key = STRIPE_SECRET_KEY

try:
checkout = stripe.checkout.Session.create(
mode="payment",
line_items=[{"price": PRICE_DREAM_PACK, "quantity": 1}],
customer_email=email,
metadata={
"purchase_type": "dream_pack",
"dream_pack_uses": str(DREAM_PACK_USES),
"dream_pack_hours": str(DREAM_PACK_HOURS),
"email": email,
},
success_url=url_for("dream_pack_success", _external=True) + f"?email={email}",
cancel_url=url_for("upgrade", _external=True),
)
return redirect(checkout.url, code=303)
except Exception as e:
return make_response(f"Stripe dream pack error: {str(e)}", 500)


@app.route("/payment-success", methods=["GET"])
def payment_success():
email = (request.args.get("email") or "").strip().lower()

if _validate_email(email):
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


@app.route("/dream-pack-success", methods=["GET"])
def dream_pack_success():
email = (request.args.get("email") or "").strip().lower()

if _validate_email(email):
_set_buyer_session(email)
_mark_dream_pack_purchase(email, uses=DREAM_PACK_USES, hours=DREAM_PACK_HOURS)

resp = Response(
f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dream Pack Active</title>
<style>
body{{margin:0;background:#07070a;color:#f2f2f7;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:26px}}
.card{{max-width:520px;margin:0 auto;background:#101017;border:1px solid rgba(212,175,55,.22);border-radius:18px;padding:18px}}
.muted{{color:#b6b6c3;font-size:13px;margin-top:8px}}
</style></head><body>
<div class="card">
<h2 style="margin:0 0 8px">{DREAM_PACK_USES} Dream Pack Activated</h2>
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
mode = (session_obj.get("mode") or "").strip().lower()
metadata = session_obj.get("metadata") or {}
purchase_type = (metadata.get("purchase_type") or "").strip().lower()

if _validate_email(email) and mode == "subscription":
_mark_subscriber(email, True, customer_id)

if _validate_email(email) and mode == "payment" and purchase_type == "dream_pack":
_mark_dream_pack_purchase(email, uses=DREAM_PACK_USES, hours=DREAM_PACK_HOURS)

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

_log("RAW JSON RECEIVED:", _safe_debug_payload_preview(data))
_log("RAW DREAM RECEIVED:", repr(dream))
_log("REQUEST CONTENT-TYPE:", request.headers.get("Content-Type", ""))
_log("REQUEST ORIGIN:", request.headers.get("Origin", ""))
_log("REQUEST REFERER:", request.headers.get("Referer", ""))

validation_error = _validate_dream_text(dream)
if validation_error:
return jsonify({"error": validation_error}), 400

is_paid = _is_premium_session()
session_email = _get_session_email()
dream_pack_status_before = _get_dream_pack_status(session_email)
has_active_dream_pack = bool(dream_pack_status_before["active"])

if not is_paid:
ip = _get_client_ip()
cookie_used = _get_cookie_tries_used()
ip_used = _shadow_count(ip)
effective_used = max(cookie_used, ip_used)

if (not has_active_dream_pack) and effective_used >= FREE_TRIES:
return jsonify({
"blocked": True,
"reason": "free_limit_reached",
"message": f"You’ve used your {FREE_TRIES} free tries. Buy {DREAM_PACK_USES} more dreams or subscribe to continue.",
"redirect": url_for("upgrade"),
"free_uses_left": 0,
"access": "blocked",
"is_paid": False,
"dream_pack_available": bool(PRICE_DREAM_PACK),
"dream_pack": {
"active": False,
"uses_remaining": 0,
"expires_at": "",
},
}), 402

free_uses_left = 0
dream_pack_status_after = dream_pack_status_before

if not is_paid:
if has_active_dream_pack:
dream_pack_status_after = _consume_dream_pack_use(session_email)
free_uses_left = 0
else:
ip = _get_client_ip()
cookie_used = _get_cookie_tries_used()
ip_used = _shadow_count(ip)
effective_used = max(cookie_used, ip_used)
_shadow_increment(ip)
free_uses_left = _free_tries_remaining_after_this(effective_used)

doctrine_active = DOCTRINE_MODE and _doctrine_sheets_available()

if doctrine_active:
try:
sheets = _load_doctrine_sheets()
base_rows = sheets.get(SHEET_BASE_SYMBOLS, [])
behavior_rows = sheets.get(SHEET_BEHAVIOR_RULES, [])
state_rows = sheets.get(SHEET_SIZE_STATE_RULES, [])
location_rows = sheets.get(SHEET_LOCATION_RULES, [])
override_rows = sheets.get(SHEET_OVERRIDE_RULES, [])
template_rows = sheets.get(SHEET_OUTPUT_TEMPLATES, [])

base_matches = _match_base_symbols_doctrine(dream, base_rows, top_k=3)
behaviors = _detect_rule_hits(dream, behavior_rows, "behavior")
states = _detect_rule_hits(dream, state_rows, "state")
locations = _detect_rule_hits(dream, location_rows, "location")
override_hit = _apply_override_rules(
base_matches=base_matches,
behaviors=behaviors,
states=states,
locations=locations,
dream=dream,
override_rows=override_rows,
)

receipt_id = f"JTS-{secrets.token_hex(4).upper()}"
seal = _compute_seal_from_symbol_count(len(base_matches))

if not base_matches and not override_hit:
payload = {
"engine_mode": "doctrine",
"access": "paid" if is_paid else "free",
"is_paid": bool(is_paid),
"free_uses_left": free_uses_left,
"dream_pack": {
"active": bool(dream_pack_status_after["active"]),
"uses_remaining": int(dream_pack_status_after["uses_remaining"]),
"expires_at": dream_pack_status_after["expires_at"],
},
"seal": seal,
"brain": {
"behaviors": [b["name"] for b in behaviors],
"states": [s["name"] for s in states],
"locations": [l["name"] for l in locations],
"template_type": "default",
},
"receipt": {
"id": receipt_id,
"top_symbols": [],
"share_phrase": "I decoded my dream on Jamaican True Stories.",
},
"interpretation": {
"spiritual_meaning": "No matching doctrine symbols were found for the exact words in this dream.",
"effects_in_physical_realm": "Tip: use clear dream symbols or phrases that exist in your doctrine system.",
"what_to_do": "Add 1–2 more key symbols or phrases and try again.",
},
"full_interpretation": "",
}
_log("TOP SYMBOLS RETURNED:", [])
_log("INTERPRET RESPONSE MODE:", "doctrine_no_matches")
resp = make_response(jsonify(payload))
else:
built = _build_doctrine_interpretation(
dream=dream,
base_matches=base_matches,
behaviors=behaviors,
states=states,
locations=locations,
override_hit=override_hit,
templates=template_rows,
)

top_symbols = built["top_symbols"]
share_phrase = (
f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories."
if top_symbols else
"I decoded my dream on Jamaican True Stories."
)

payload = {
"engine_mode": "doctrine",
"access": "paid" if is_paid else "free",
"is_paid": bool(is_paid),
"free_uses_left": free_uses_left,
"dream_pack": {
"active": bool(dream_pack_status_after["active"]),
"uses_remaining": int(dream_pack_status_after["uses_remaining"]),
"expires_at": dream_pack_status_after["expires_at"],
},
"seal": seal,
"brain": {
"behaviors": [b["name"] for b in behaviors],
"states": [s["name"] for s in states],
"locations": [l["name"] for l in locations],
"top_symbol_categories": [
{
"symbol": _get_base_symbol_input(row),
"category": _get_base_symbol_category(row)
}
for row, _sc, _hit in base_matches
],
"override_applied": built["override_applied"],
"override_name": built["override_name"],
"template_type": built.get("template_type", "default"),
},
"interpretation": built["interpretation"],
"full_interpretation": built["full_interpretation"],
"receipt": {
"id": receipt_id,
"top_symbols": top_symbols,
"share_phrase": share_phrase,
},
}

_log("TOP SYMBOLS RETURNED:", top_symbols)
_log("MATCHED BEHAVIORS:", [b["name"] for b in behaviors])
_log("MATCHED STATES:", [s["name"] for s in states])
_log("MATCHED LOCATIONS:", [l["name"] for l in locations])
_log("OVERRIDE:", override_hit)
_log("TEMPLATE TYPE:", built.get("template_type", "default"))
_log("INTERPRET RESPONSE MODE:", "doctrine_matched")
resp = make_response(jsonify(payload))

except Exception as e:
return jsonify({
"error": "Doctrine engine failed",
"details": str(e),
}), 500

else:
try:
legacy_rows = _load_legacy_sheet_rows()
except Exception as e:
return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

matches = _match_symbols_legacy(dream, legacy_rows, top_k=3)
receipt_id = f"JTS-{secrets.token_hex(4).upper()}"
seal = _compute_seal_from_symbol_count(len(matches))

if not matches:
payload = {
"engine_mode": "legacy",
"access": "paid" if is_paid else "free",
"is_paid": bool(is_paid),
"free_uses_left": free_uses_left,
"dream_pack": {
"active": bool(dream_pack_status_after["active"]),
"uses_remaining": int(dream_pack_status_after["uses_remaining"]),
"expires_at": dream_pack_status_after["expires_at"],
},
"seal": seal,
"brain": {},
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
_log("TOP SYMBOLS RETURNED:", [])
_log("INTERPRET RESPONSE MODE:", "legacy_no_matches")
resp = make_response(jsonify(payload))
else:
built_legacy = _build_legacy_interpretation(matches)
top_symbols = [_get_symbol_cell(row) for row, _sc, _hit in matches if _get_symbol_cell(row)]

full_parts = [
_sentence("This dream is revealing symbolic meaning through the matched symbols"),
built_legacy["spiritual_meaning"],
built_legacy["effects_in_physical_realm"],
built_legacy["what_to_do"],
_sentence("Dreams expose what needs attention so you can respond with wisdom"),
]
full_interpretation = "\n\n".join([x for x in full_parts if x and x.strip()])

payload = {
"engine_mode": "legacy",
"access": "paid" if is_paid else "free",
"is_paid": bool(is_paid),
"free_uses_left": free_uses_left,
"dream_pack": {
"active": bool(dream_pack_status_after["active"]),
"uses_remaining": int(dream_pack_status_after["uses_remaining"]),
"expires_at": dream_pack_status_after["expires_at"],
},
"seal": seal,
"brain": {},
"interpretation": built_legacy,
"full_interpretation": full_interpretation,
"receipt": {
"id": receipt_id,
"top_symbols": top_symbols,
"share_phrase": f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories.",
},
}

_log("TOP SYMBOLS RETURNED:", top_symbols)
_log("INTERPRET RESPONSE MODE:", "legacy_matched")
resp = make_response(jsonify(payload))

if not is_paid and not has_active_dream_pack:
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
"dream_pack": _get_dream_pack_status(_get_session_email()),
})


# ============================================================
# Admin routes
# ============================================================
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


# ============================================================
# Debug routes
# ============================================================
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
"session_premium": _is_premium_session(),
"session_email": _get_session_email(),
"session_dream_pack": _get_dream_pack_status(_get_session_email()),
"keyword_guard_enabled": KEYWORD_GUARD_ENABLED,
"doctrine_mode_enabled": DOCTRINE_MODE,
"doctrine_sheet_names": DOCTRINE_SHEET_NAMES,
})


@app.route("/debug/sheet", methods=["GET"])
def debug_sheet():
if not DEBUG_MATCH:
return jsonify({"error": "Debug disabled"}), 403

rows = _load_legacy_sheet_rows(force=True)
headers = _LEGACY_CACHE.get("headers", [])
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
"sample_keywords_split": _split_keywords(_get_keywords_cell(sample)),
})


@app.route("/debug/doctrine", methods=["GET"])
def debug_doctrine():
if not DEBUG_MATCH:
return jsonify({"error": "Debug disabled"}), 403

try:
sheets = _load_doctrine_sheets(force=True)
headers = _DOCTRINE_CACHE.get("headers", {})
summary = {}

for name in DOCTRINE_SHEET_NAMES:
rows = sheets.get(name, [])
sample = rows[0] if rows else {}
summary[name] = {
"row_count": len(rows),
"headers_seen": headers.get(name, []),
"sample_keys": list(sample.keys()) if sample else [],
"sample_row": sample,
}

return jsonify({
"doctrine_mode_enabled": DOCTRINE_MODE,
"doctrine_sheets_available": _doctrine_sheets_available(),
"sheets": summary,
})
except Exception as e:
return jsonify({"error": "Doctrine debug failed", "details": str(e)}), 500


if __name__ == "__main__":
app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
