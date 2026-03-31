from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from flask import request, session

from app.config import Config
from app.storage import ensure_json_file, load_dict_file, write_json_file_atomic
from app.utils import normalize_email


USAGE_COUNTS_PATH = Path(Config.COUNTS_FILE)
SUBSCRIBERS_PATH = Path(Config.SUBSCRIBERS_FILE)

ensure_json_file(USAGE_COUNTS_PATH, {"ip_shadow": {}})
ensure_json_file(SUBSCRIBERS_PATH, {})


# ============================================================
# Session helpers
# ============================================================
def get_session_email() -> str:
    return normalize_email(session.get("subscriber_email") or "")


def is_premium_session() -> bool:
    return bool(session.get("premium") is True and get_session_email())


def persist_email_to_session(email: str) -> None:
    email_n = normalize_email(email)
    if email_n:
        session["subscriber_email"] = email_n


def set_premium_session(email: str) -> None:
    session["subscriber_email"] = normalize_email(email)
    session["premium"] = True
    session["premium_set_at"] = datetime.utcnow().isoformat() + "Z"


def set_buyer_session(email: str) -> None:
    session["subscriber_email"] = normalize_email(email)
    session["premium"] = False
    session["buyer_set_at"] = datetime.utcnow().isoformat() + "Z"


def clear_access_session() -> None:
    session.pop("subscriber_email", None)
    session.pop("premium", None)
    session.pop("premium_set_at", None)
    session.pop("buyer_set_at", None)


# ============================================================
# Cookie / IP helpers
# ============================================================
def get_client_ip() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "0.0.0.0"


def get_cookie_tries_used() -> int:
    try:
        return int(request.cookies.get(Config.COOKIE_NAME, "0"))
    except Exception:
        return 0


def free_tries_remaining_after_this(effective_used_before: int) -> int:
    return max(0, Config.FREE_TRIES - (effective_used_before + 1))


# ============================================================
# Usage counts storage
# ============================================================
def load_usage_counts() -> Dict[str, Any]:
    data = load_dict_file(USAGE_COUNTS_PATH)
    if "ip_shadow" not in data or not isinstance(data["ip_shadow"], dict):
        data["ip_shadow"] = {}
    return data


def save_usage_counts(data: Dict[str, Any]) -> None:
    write_json_file_atomic(USAGE_COUNTS_PATH, data)


def shadow_get(ip: str) -> Dict[str, Any]:
    data = load_usage_counts()
    shadow = data["ip_shadow"]

    rec = shadow.get(ip) or {
        "count": 0,
        "first_seen": None,
        "last_seen": None,
    }

    now = datetime.utcnow()
    window = timedelta(hours=Config.SHADOW_WINDOW_HOURS)

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
    save_usage_counts(data)
    return rec


def shadow_count(ip: str) -> int:
    rec = shadow_get(ip)
    try:
        return int(rec.get("count", 0))
    except Exception:
        return 0


def shadow_increment(ip: str) -> int:
    data = load_usage_counts()
    shadow = data["ip_shadow"]

    rec = shadow.get(ip) or {
        "count": 0,
        "first_seen": None,
        "last_seen": None,
    }

    now = datetime.utcnow()
    window = timedelta(hours=Config.SHADOW_WINDOW_HOURS)

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
    save_usage_counts(data)
    return rec["count"]


# ============================================================
# Subscriber storage
# ============================================================
def load_subscribers() -> Dict[str, Any]:
    return load_dict_file(SUBSCRIBERS_PATH)


def save_subscribers(data: Dict[str, Any]) -> None:
    write_json_file_atomic(SUBSCRIBERS_PATH, data)


def parse_iso_z(dt_str: str):
    try:
        if not dt_str:
            return None
        return datetime.fromisoformat(str(dt_str).replace("Z", ""))
    except Exception:
        return None


def mark_subscriber(email: str, is_active: bool, stripe_customer_id: str = "") -> None:
    if not email:
        return

    email_n = normalize_email(email)
    subs = load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    rec["email"] = email_n
    rec["is_active"] = bool(is_active)
    if stripe_customer_id:
        rec["stripe_customer_id"] = stripe_customer_id
    rec["updated_at"] = datetime.utcnow().isoformat() + "Z"

    subs[email_n] = rec
    save_subscribers(subs)


def mark_dream_pack_purchase(
    email: str,
    uses: int = Config.DREAM_PACK_USES,
    hours: int = Config.DREAM_PACK_HOURS,
) -> None:
    if not email:
        return

    email_n = normalize_email(email)
    subs = load_subscribers()
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
    save_subscribers(subs)


def get_dream_pack_status(email: str) -> Dict[str, Any]:
    email_n = normalize_email(email)
    if not email_n:
        return {"active": False, "uses_remaining": 0, "expires_at": ""}

    subs = load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    try:
        uses_remaining = int(rec.get("dream_pack_uses_remaining", 0) or 0)
    except Exception:
        uses_remaining = 0

    expires_at = (rec.get("dream_pack_expires_at") or "").strip()
    expires_dt = parse_iso_z(expires_at)

    active = False
    if uses_remaining > 0 and expires_dt and datetime.utcnow() < expires_dt:
        active = True

    return {
        "active": active,
        "uses_remaining": uses_remaining if active else 0,
        "expires_at": expires_at if active else "",
    }


def consume_dream_pack_use(email: str) -> Dict[str, Any]:
    email_n = normalize_email(email)
    if not email_n:
        return {"active": False, "uses_remaining": 0, "expires_at": ""}

    subs = load_subscribers()
    rec = subs.get(email_n, {}) if isinstance(subs.get(email_n, {}), dict) else {}

    current = get_dream_pack_status(email_n)
    if not current["active"]:
        return current

    uses_remaining = max(0, int(current["uses_remaining"]) - 1)
    rec["dream_pack_uses_remaining"] = uses_remaining
    rec["dream_pack_expires_at"] = current["expires_at"]
    rec["updated_at"] = datetime.utcnow().isoformat() + "Z"

    subs[email_n] = rec
    save_subscribers(subs)

    return {
        "active": uses_remaining > 0 and bool(current["expires_at"]),
        "uses_remaining": uses_remaining,
        "expires_at": current["expires_at"],
    }
