from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator

from flask import request, session

from app.config import Config
from app.storage import (
    ensure_json_file,
    load_dict_file,
    write_json_file_atomic,
)
from app.utils import normalize_email


USAGE_COUNTS_PATH = Path(Config.COUNTS_FILE)
SUBSCRIBERS_PATH = Path(Config.SUBSCRIBERS_FILE)

ensure_json_file(
    USAGE_COUNTS_PATH,
    {"ip_shadow": {}},
)

ensure_json_file(
    SUBSCRIBERS_PATH,
    {},
)


# ============================================================
# CONSTANTS
# ============================================================

EMPTY_PACK = {
    "active": False,
    "uses_remaining": 0,
    "expires_at": "",
}


# ============================================================
# FILE LOCKING
# ============================================================

@contextmanager
def _json_file_lock(path: Path) -> Iterator[None]:
    """
    Cross-platform file lock for JSON read-modify-write operations.

    This protects the full sequence:
    1. read JSON
    2. modify data
    3. write JSON

    Atomic writes alone are not enough because two requests can read the
    same old data and then overwrite each other's updates.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            lock_file.write("0")
            lock_file.flush()
            lock_file.seek(0)

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)

            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ============================================================
# DATETIME HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.utcnow()


def iso_now() -> str:
    return utc_now().isoformat() + "Z"


def parse_iso_z(dt_str: str):
    try:
        if not dt_str:
            return None

        return datetime.fromisoformat(
            str(dt_str).replace("Z", "")
        )

    except Exception:
        return None


# ============================================================
# SESSION HELPERS
# ============================================================

def get_session_email() -> str:
    return normalize_email(
        session.get("subscriber_email") or ""
    )


def is_premium_session() -> bool:
    return bool(
        session.get("premium") is True
        and get_session_email()
    )


def persist_email_to_session(email: str) -> None:
    email_n = normalize_email(email)

    if not email_n:
        return

    session["subscriber_email"] = email_n
    session.permanent = True


def set_premium_session(email: str) -> None:
    email_n = normalize_email(email)

    if not email_n:
        return

    session["subscriber_email"] = email_n
    session["premium"] = True
    session["premium_set_at"] = iso_now()
    session.permanent = True


def set_buyer_session(email: str) -> None:
    email_n = normalize_email(email)

    if not email_n:
        return

    session["subscriber_email"] = email_n
    session["premium"] = False
    session["buyer_set_at"] = iso_now()
    session.permanent = True


def clear_access_session() -> None:
    keys = [
        "subscriber_email",
        "premium",
        "premium_set_at",
        "buyer_set_at",
    ]

    for key in keys:
        session.pop(key, None)


# ============================================================
# REQUEST / IP HELPERS
# ============================================================

def get_client_ip() -> str:
    try:
        xff = request.headers.get(
            "X-Forwarded-For",
            "",
        )

        if xff:
            ip = xff.split(",")[0].strip()
            if ip:
                return ip

        real_ip = (
            request.headers.get("X-Real-IP")
            or ""
        ).strip()

        if real_ip:
            return real_ip

        return (
            request.remote_addr
            or "0.0.0.0"
        )

    except Exception:
        return "0.0.0.0"


def get_cookie_tries_used() -> int:
    try:
        value = request.cookies.get(
            Config.COOKIE_NAME,
            "0",
        )

        return max(0, int(value))

    except Exception:
        return 0


def free_tries_remaining_after_this(
    effective_used_before: int,
) -> int:
    return max(
        0,
        Config.FREE_TRIES - (
            effective_used_before + 1
        ),
    )


# ============================================================
# USAGE COUNTS - UNLOCKED INTERNAL HELPERS
# ============================================================

def _load_usage_counts_unlocked() -> Dict[str, Any]:
    data = load_dict_file(
        USAGE_COUNTS_PATH
    )

    if not isinstance(data, dict):
        data = {}

    if not isinstance(
        data.get("ip_shadow"),
        dict,
    ):
        data["ip_shadow"] = {}

    return data


def _save_usage_counts_unlocked(
    data: Dict[str, Any],
) -> None:
    write_json_file_atomic(
        USAGE_COUNTS_PATH,
        data,
    )


# ============================================================
# USAGE COUNTS - PUBLIC HELPERS
# ============================================================

def load_usage_counts() -> Dict[str, Any]:
    with _json_file_lock(USAGE_COUNTS_PATH):
        return _load_usage_counts_unlocked()


def save_usage_counts(
    data: Dict[str, Any],
) -> None:
    with _json_file_lock(USAGE_COUNTS_PATH):
        _save_usage_counts_unlocked(data)


def _empty_shadow_record() -> Dict[str, Any]:
    return {
        "count": 0,
        "first_seen": None,
        "last_seen": None,
    }


def _normalize_shadow_record(
    rec: Any,
) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        rec = _empty_shadow_record()

    now = utc_now()

    if not rec.get("first_seen"):
        rec["first_seen"] = iso_now()

    if not rec.get("last_seen"):
        rec["last_seen"] = iso_now()

    try:
        first_seen = parse_iso_z(
            rec.get("first_seen")
        )

        if not first_seen:
            raise ValueError()

        window = timedelta(
            hours=Config.SHADOW_WINDOW_HOURS
        )

        if now - first_seen > window:
            rec = _empty_shadow_record()
            rec["first_seen"] = iso_now()
            rec["last_seen"] = iso_now()

    except Exception:
        rec = _empty_shadow_record()
        rec["first_seen"] = iso_now()
        rec["last_seen"] = iso_now()

    try:
        rec["count"] = max(
            0,
            int(rec.get("count", 0)),
        )
    except Exception:
        rec["count"] = 0

    return rec


def shadow_get(ip: str) -> Dict[str, Any]:
    with _json_file_lock(USAGE_COUNTS_PATH):
        data = _load_usage_counts_unlocked()

        shadow = data["ip_shadow"]

        rec = _normalize_shadow_record(
            shadow.get(ip)
        )

        shadow[ip] = rec

        _save_usage_counts_unlocked(data)

        return dict(rec)


def shadow_count(ip: str) -> int:
    rec = shadow_get(ip)

    try:
        return max(
            0,
            int(rec.get("count", 0)),
        )

    except Exception:
        return 0


def shadow_increment(ip: str) -> int:
    with _json_file_lock(USAGE_COUNTS_PATH):
        data = _load_usage_counts_unlocked()

        shadow = data["ip_shadow"]

        rec = _normalize_shadow_record(
            shadow.get(ip)
        )

        rec["count"] = max(
            0,
            int(rec.get("count", 0)),
        ) + 1

        rec["last_seen"] = iso_now()

        shadow[ip] = rec

        _save_usage_counts_unlocked(data)

        return rec["count"]


# ============================================================
# SUBSCRIBERS - UNLOCKED INTERNAL HELPERS
# ============================================================

def _load_subscribers_unlocked() -> Dict[str, Any]:
    data = load_dict_file(
        SUBSCRIBERS_PATH
    )

    return data if isinstance(data, dict) else {}


def _save_subscribers_unlocked(
    data: Dict[str, Any],
) -> None:
    write_json_file_atomic(
        SUBSCRIBERS_PATH,
        data,
    )


# ============================================================
# SUBSCRIBERS - PUBLIC HELPERS
# ============================================================

def load_subscribers() -> Dict[str, Any]:
    with _json_file_lock(SUBSCRIBERS_PATH):
        return _load_subscribers_unlocked()


def save_subscribers(
    data: Dict[str, Any],
) -> None:
    with _json_file_lock(SUBSCRIBERS_PATH):
        _save_subscribers_unlocked(data)


def mark_subscriber(
    email: str,
    is_active: bool,
    stripe_customer_id: str = "",
) -> None:
    email_n = normalize_email(email)

    if not email_n:
        return

    with _json_file_lock(SUBSCRIBERS_PATH):
        subscribers = _load_subscribers_unlocked()

        rec = subscribers.get(email_n)

        if not isinstance(rec, dict):
            rec = {}

        rec["email"] = email_n
        rec["is_active"] = bool(is_active)
        rec["updated_at"] = iso_now()

        if stripe_customer_id:
            rec["stripe_customer_id"] = (
                stripe_customer_id
            )

        subscribers[email_n] = rec

        _save_subscribers_unlocked(subscribers)


def mark_dream_pack_purchase(
    email: str,
    uses: int = Config.DREAM_PACK_USES,
    hours: int = Config.DREAM_PACK_HOURS,
) -> None:
    email_n = normalize_email(email)

    if not email_n:
        return

    with _json_file_lock(SUBSCRIBERS_PATH):
        subscribers = _load_subscribers_unlocked()

        rec = subscribers.get(email_n)

        if not isinstance(rec, dict):
            rec = {}

        expires_at = (
            utc_now()
            + timedelta(hours=hours)
        ).isoformat() + "Z"

        rec["email"] = email_n
        rec["dream_pack_uses_remaining"] = max(
            0,
            int(uses),
        )

        rec["dream_pack_expires_at"] = (
            expires_at
        )

        rec["updated_at"] = iso_now()

        if "is_active" not in rec:
            rec["is_active"] = False

        if "stripe_customer_id" not in rec:
            rec["stripe_customer_id"] = ""

        subscribers[email_n] = rec

        _save_subscribers_unlocked(subscribers)


def _get_dream_pack_status_from_record(
    rec: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        return dict(EMPTY_PACK)

    try:
        uses_remaining = max(
            0,
            int(
                rec.get(
                    "dream_pack_uses_remaining",
                    0,
                )
                or 0
            ),
        )

    except Exception:
        uses_remaining = 0

    expires_at = (
        rec.get("dream_pack_expires_at")
        or ""
    ).strip()

    expires_dt = parse_iso_z(expires_at)

    if (
        not expires_dt
        or utc_now() >= expires_dt
        or uses_remaining <= 0
    ):
        return dict(EMPTY_PACK)

    return {
        "active": True,
        "uses_remaining": uses_remaining,
        "expires_at": expires_at,
    }


def get_dream_pack_status(
    email: str,
) -> Dict[str, Any]:
    email_n = normalize_email(email)

    if not email_n:
        return dict(EMPTY_PACK)

    with _json_file_lock(SUBSCRIBERS_PATH):
        subscribers = _load_subscribers_unlocked()

        rec = subscribers.get(email_n)

        return _get_dream_pack_status_from_record(rec)


def consume_dream_pack_use(
    email: str,
) -> Dict[str, Any]:
    email_n = normalize_email(email)

    if not email_n:
        return dict(EMPTY_PACK)

    with _json_file_lock(SUBSCRIBERS_PATH):
        subscribers = _load_subscribers_unlocked()

        rec = subscribers.get(email_n)

        if not isinstance(rec, dict):
            return dict(EMPTY_PACK)

        current = _get_dream_pack_status_from_record(rec)

        if not current["active"]:
            return current

        uses_remaining = max(
            0,
            int(current["uses_remaining"]) - 1,
        )

        rec["dream_pack_uses_remaining"] = (
            uses_remaining
        )

        rec["dream_pack_expires_at"] = (
            current["expires_at"]
        )

        rec["updated_at"] = iso_now()

        subscribers[email_n] = rec

        _save_subscribers_unlocked(subscribers)

        return {
            "active": uses_remaining > 0,
            "uses_remaining": uses_remaining,
            "expires_at": (
                current["expires_at"]
                if uses_remaining > 0
                else ""
            ),
        }


# ============================================================
# ACCESS CHECK WRAPPER
# ============================================================

def has_active_access(email: str):
    """
    Compatibility wrapper.

    Some routes import has_active_access from app.access.
    The real Stripe/subscription logic lives in app.billing.
    This prevents import errors while keeping billing logic separate.
    """
    from app.billing import has_active_access as billing_has_active_access

    return billing_has_active_access(email)