from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from flask import Request

from app.sheets import get_or_create_worksheet


ADMOB_PUBLIC_KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
REWARDS_SHEET_NAME = "AdMobRewards"

REWARD_HEADERS = [
    "reward_id",
    "status",
    "user_id",
    "email",
    "transaction_id",
    "ad_unit",
    "reward_amount",
    "reward_item",
    "ad_network",
    "callback_timestamp_ms",
    "created_at",
    "verified_at",
    "consumed_at",
    "custom_data",
    "key_id",
    "notes",
]

DEFAULT_PRODUCTION_AD_UNIT_ID = "ca-app-pub-2148520143702642/6436071510"
PUBLIC_KEY_CACHE_SECONDS = 23 * 60 * 60
CALLBACK_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
CALLBACK_FUTURE_TOLERANCE_SECONDS = 5 * 60

_REWARD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@+-]{6,128}$")
_TRANSACTION_ID_PATTERN = re.compile(r"^[A-Fa-f0-9]{8,128}$")

_PUBLIC_KEY_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "keys": {}}
_PUBLIC_KEY_CACHE_LOCK = threading.Lock()
_SHEET_LOCK = threading.Lock()


class AdMobSSVError(Exception):
    """Base error for an AdMob SSV validation failure."""


class AdMobSSVSignatureError(AdMobSSVError):
    """Raised when the Google signature cannot be verified."""


class AdMobSSVConfigurationError(AdMobSSVError):
    """Raised when the server or rewards sheet is misconfigured."""


class AdMobSSVConflictError(AdMobSSVError):
    """Raised when a reward ID is reused with another transaction."""


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _normalize_ad_unit(value: Any) -> str:
    raw = _clean(value)
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _expected_ad_unit() -> str:
    configured = (
        os.getenv("ADMOB_REWARDED_AD_UNIT_ID")
        or DEFAULT_PRODUCTION_AD_UNIT_ID
    )
    return _normalize_ad_unit(configured)


def _urlsafe_b64decode(value: str) -> bytes:
    cleaned = _clean(value)
    if not cleaned:
        raise AdMobSSVSignatureError("Missing callback signature.")

    padded = cleaned + ("=" * (-len(cleaned) % 4))

    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:
        raise AdMobSSVSignatureError(
            "The callback signature is not valid URL-safe Base64."
        ) from exc


def _extract_signed_parts(raw_query_string: str) -> Tuple[bytes, bytes, int]:
    """
    Extract exactly what Google signed.

    Google documents that signature and key_id are the final two query
    parameters, in that order. Everything before &signature= must be
    verified without decoding, reordering, or rebuilding it.
    """
    signature_marker = "&signature="
    key_id_marker = "&key_id="

    signature_position = raw_query_string.find(signature_marker)

    if signature_position <= 0:
        raise AdMobSSVSignatureError(
            "The callback does not contain a correctly positioned signature."
        )

    key_id_position = raw_query_string.find(
        key_id_marker,
        signature_position + len(signature_marker),
    )

    if key_id_position <= signature_position:
        raise AdMobSSVSignatureError(
            "The callback does not contain a correctly positioned key_id."
        )

    key_id_text = raw_query_string[
        key_id_position + len(key_id_marker):
    ]

    if not key_id_text or "&" in key_id_text:
        raise AdMobSSVSignatureError(
            "The callback key_id must be the final query parameter."
        )

    try:
        key_id = int(key_id_text)
    except Exception as exc:
        raise AdMobSSVSignatureError(
            "The callback key_id is invalid."
        ) from exc

    signature_text = raw_query_string[
        signature_position + len(signature_marker):
        key_id_position
    ]

    # Google's official RewardedAdsVerifier parses the callback with
    # Java URI.getQuery(), which percent-decodes escaped octets before
    # verifying the signature. Flask's request.query_string is raw, so
    # decode only the signed prefix while preserving parameter order.
    raw_content_to_verify = raw_query_string[
        :signature_position
    ]
    content_to_verify = unquote(
        raw_content_to_verify
    ).encode("utf-8")

    signature_bytes = _urlsafe_b64decode(signature_text)

    return content_to_verify, signature_bytes, key_id


def _parse_custom_data(value: str) -> Dict[str, Any]:
    raw = _clean(value)

    if not raw:
        raise AdMobSSVError("Missing AdMob custom_data.")

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise AdMobSSVError(
            "AdMob custom_data is not valid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise AdMobSSVError(
            "AdMob custom_data must be a JSON object."
        )

    return parsed


def _validate_callback_fields(request_obj: Request) -> Dict[str, Any]:
    args = request_obj.args

    custom_data_raw = args.get("custom_data", "")
    custom_data = _parse_custom_data(custom_data_raw)

    reward_id = _clean(custom_data.get("reward_id"))
    email = _clean(custom_data.get("email")).lower()
    user_id = _clean(args.get("user_id", ""))
    transaction_id = _clean(args.get("transaction_id", ""))
    ad_unit = _normalize_ad_unit(args.get("ad_unit", ""))
    reward_amount = _safe_int(args.get("reward_amount"), 0)
    reward_item = _clean(args.get("reward_item", ""))
    ad_network = _clean(args.get("ad_network", ""))
    callback_timestamp_ms = _safe_int(args.get("timestamp"), 0)
    key_id = _safe_int(args.get("key_id"), 0)

    if not _REWARD_ID_PATTERN.fullmatch(reward_id):
        raise AdMobSSVError(
            "The reward_id in custom_data is missing or invalid."
        )

    if not _USER_ID_PATTERN.fullmatch(user_id):
        raise AdMobSSVError(
            "The callback user_id is missing or invalid."
        )

    if not _TRANSACTION_ID_PATTERN.fullmatch(transaction_id):
        raise AdMobSSVError(
            "The callback transaction_id is missing or invalid."
        )

    expected_ad_unit = _expected_ad_unit()

    google_verification_test = (
        _env_flag("ADMOB_SSV_ALLOW_TEST_KEYS")
        and ad_unit == "1234567890"
    )

    if (
        not ad_unit
        or (
            ad_unit != expected_ad_unit
            and not google_verification_test
        )
    ):
        raise AdMobSSVError(
            "The callback ad unit does not match the configured rewarded ad unit."
        )

    if reward_amount < 1:
        raise AdMobSSVError(
            "The callback reward amount must be at least 1."
        )

    if not reward_item:
        raise AdMobSSVError(
            "The callback reward item is missing."
        )

    if callback_timestamp_ms <= 0:
        raise AdMobSSVError(
            "The callback timestamp is missing or invalid."
        )

    callback_seconds = callback_timestamp_ms / 1000.0
    now_seconds = time.time()

    if callback_seconds > (
        now_seconds + CALLBACK_FUTURE_TOLERANCE_SECONDS
    ):
        raise AdMobSSVError(
            "The callback timestamp is implausibly far in the future."
        )

    if now_seconds - callback_seconds > CALLBACK_MAX_AGE_SECONDS:
        raise AdMobSSVError(
            "The callback timestamp is too old."
        )

    if key_id <= 0:
        raise AdMobSSVError(
            "The callback key_id is missing or invalid."
        )

    return {
        "reward_id": reward_id,
        "email": email,
        "user_id": user_id,
        "transaction_id": transaction_id,
        "ad_unit": ad_unit,
        "reward_amount": reward_amount,
        "reward_item": reward_item,
        "ad_network": ad_network,
        "callback_timestamp_ms": callback_timestamp_ms,
        "custom_data": custom_data_raw,
        "key_id": key_id,
    }


def _download_public_keys(url: str) -> Dict[int, str]:
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "Accept": "application/json",
                "User-Agent": "JDI-AdMob-SSV/1.0",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise AdMobSSVConfigurationError(
            "Unable to retrieve AdMob verification keys."
        ) from exc

    keys: Dict[int, str] = {}

    for item in payload.get("keys", []) or []:
        key_id = _safe_int(item.get("keyId"), 0)
        pem = _clean(item.get("pem"))

        if key_id > 0 and pem:
            keys[key_id] = pem

    if not keys:
        raise AdMobSSVConfigurationError(
            "The AdMob key server returned no usable public keys."
        )

    return keys


def _fetch_public_keys(
    *,
    force_refresh: bool = False,
) -> Dict[int, str]:
    now = time.time()

    with _PUBLIC_KEY_CACHE_LOCK:
        cached_keys = _PUBLIC_KEY_CACHE.get("keys") or {}
        fetched_at = float(
            _PUBLIC_KEY_CACHE.get("fetched_at") or 0.0
        )

        cache_is_fresh = (
            cached_keys
            and now - fetched_at < PUBLIC_KEY_CACHE_SECONDS
        )

        if cache_is_fresh and not force_refresh:
            return dict(cached_keys)

        keys = _download_public_keys(ADMOB_PUBLIC_KEYS_URL)

        _PUBLIC_KEY_CACHE["keys"] = dict(keys)
        _PUBLIC_KEY_CACHE["fetched_at"] = now

        return dict(keys)


def _verify_signature(
    *,
    content_to_verify: bytes,
    signature: bytes,
    key_id: int,
) -> None:
    keys = _fetch_public_keys()
    pem = keys.get(key_id)

    if not pem:
        keys = _fetch_public_keys(force_refresh=True)
        pem = keys.get(key_id)

    if not pem:
        raise AdMobSSVSignatureError(
            f"No AdMob public key exists for key_id {key_id}."
        )

    try:
        public_key = serialization.load_pem_public_key(
            pem.encode("utf-8")
        )

        if not isinstance(
            public_key,
            ec.EllipticCurvePublicKey,
        ):
            raise AdMobSSVSignatureError(
                "The selected AdMob verification key is not an EC public key."
            )

        public_key.verify(
            signature,
            content_to_verify,
            ec.ECDSA(hashes.SHA256()),
        )

    except InvalidSignature as exc:
        raise AdMobSSVSignatureError(
            "AdMob callback signature verification failed."
        ) from exc
    except AdMobSSVError:
        raise
    except Exception as exc:
        raise AdMobSSVSignatureError(
            "Unable to verify the AdMob callback signature."
        ) from exc


def _get_rewards_worksheet():
    worksheet = get_or_create_worksheet(
        REWARDS_SHEET_NAME,
        rows=5000,
        cols=20,
    )

    current_headers = [
        _clean(value)
        for value in worksheet.row_values(1)
    ]

    if current_headers[:len(REWARD_HEADERS)] != REWARD_HEADERS:
        raise AdMobSSVConfigurationError(
            "The AdMobRewards sheet headings do not match the required order."
        )

    return worksheet


def _row_to_record(values: List[Any]) -> Dict[str, str]:
    padded = [
        _clean(value)
        for value in list(values or [])
    ]

    if len(padded) < len(REWARD_HEADERS):
        padded.extend(
            [""] * (len(REWARD_HEADERS) - len(padded))
        )

    return {
        REWARD_HEADERS[index]: padded[index]
        for index in range(len(REWARD_HEADERS))
    }


def _find_row_by_column_value(
    worksheet,
    *,
    column_number: int,
    target: str,
) -> Tuple[Optional[int], Optional[Dict[str, str]]]:
    wanted = _clean(target)

    if not wanted:
        return None, None

    values = worksheet.col_values(column_number)

    for row_number, value in enumerate(values[1:], start=2):
        if _clean(value) == wanted:
            row = worksheet.row_values(row_number)
            return row_number, _row_to_record(row)

    return None, None


def record_verified_reward(
    callback: Dict[str, Any],
) -> Dict[str, Any]:
    now_iso = _utc_now_iso()

    with _SHEET_LOCK:
        worksheet = _get_rewards_worksheet()

        _, existing_transaction = _find_row_by_column_value(
            worksheet,
            column_number=5,
            target=callback["transaction_id"],
        )

        if existing_transaction:
            return {
                "created": False,
                "duplicate": True,
                "record": existing_transaction,
            }

        _, existing_reward = _find_row_by_column_value(
            worksheet,
            column_number=1,
            target=callback["reward_id"],
        )

        if existing_reward:
            raise AdMobSSVConflictError(
                "This reward_id is already attached to another transaction."
            )

        row = [
            callback["reward_id"],
            "verified",
            callback["user_id"],
            callback["email"],
            callback["transaction_id"],
            callback["ad_unit"],
            str(callback["reward_amount"]),
            callback["reward_item"],
            callback["ad_network"],
            str(callback["callback_timestamp_ms"]),
            now_iso,
            now_iso,
            "",
            callback["custom_data"],
            str(callback["key_id"]),
            "AdMob SSV signature verified.",
        ]

        worksheet.append_row(
            row,
            value_input_option="RAW",
        )

        return {
            "created": True,
            "duplicate": False,
            "record": _row_to_record(row),
        }


def get_reward_record(
    reward_id: str,
) -> Optional[Dict[str, str]]:
    with _SHEET_LOCK:
        worksheet = _get_rewards_worksheet()
        _, record = _find_row_by_column_value(
            worksheet,
            column_number=1,
            target=reward_id,
        )
        return record


def reward_is_available(
    *,
    reward_id: str,
    user_id: str,
) -> bool:
    record = get_reward_record(reward_id)

    if not record:
        return False

    return (
        record.get("status") == "verified"
        and record.get("user_id") == _clean(user_id)
        and not record.get("consumed_at")
    )


def consume_verified_reward(
    *,
    reward_id: str,
    user_id: str,
) -> bool:
    with _SHEET_LOCK:
        worksheet = _get_rewards_worksheet()

        row_number, record = _find_row_by_column_value(
            worksheet,
            column_number=1,
            target=reward_id,
        )

        if not row_number or not record:
            return False

        if record.get("status") != "verified":
            return False

        if record.get("user_id") != _clean(user_id):
            return False

        if record.get("consumed_at"):
            return False

        record["status"] = "consumed"
        record["consumed_at"] = _utc_now_iso()

        replacement_row = [
            record.get(header, "")
            for header in REWARD_HEADERS
        ]

        worksheet.update(
            f"A{row_number}:P{row_number}",
            [replacement_row],
            value_input_option="RAW",
        )

        return True


def verify_and_record_callback(
    request_obj: Request,
) -> Dict[str, Any]:
    raw_query_string = request_obj.query_string.decode(
        "utf-8",
        errors="strict",
    )

    content_to_verify, signature, key_id = (
        _extract_signed_parts(raw_query_string)
    )

    callback = _validate_callback_fields(request_obj)

    if callback["key_id"] != key_id:
        raise AdMobSSVSignatureError(
            "The parsed key_id does not match the signed callback key_id."
        )

    _verify_signature(
        content_to_verify=content_to_verify,
        signature=signature,
        key_id=key_id,
    )

    ledger_result = record_verified_reward(callback)

    return {
        "ok": True,
        "verified": True,
        "reward_id": callback["reward_id"],
        "transaction_id": callback["transaction_id"],
        "duplicate": ledger_result["duplicate"],
    }
