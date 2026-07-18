from __future__ import annotations

import re
from typing import Any, Dict

from flask import Blueprint, jsonify, make_response, request

from app.services.admob_ssv_service import (
    AdMobSSVConfigurationError,
    AdMobSSVConflictError,
    AdMobSSVError,
    AdMobSSVSignatureError,
    get_reward_record,
    verify_and_record_callback,
)


admob_bp = Blueprint("admob", __name__)

_REWARD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@+-]{6,128}$")


def _json_response(
    payload: Dict[str, Any],
    status_code: int = 200,
):
    response = make_response(
        jsonify(payload),
        status_code,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _clean(value: Any) -> str:
    return str(value or "").strip()


@admob_bp.route(
    "/admob-ssv",
    methods=["GET"],
)
def admob_ssv_callback():
    """
    Receive and verify Google AdMob rewarded-ad SSV callbacks.

    A successful or already-recorded transaction returns HTTP 200.
    Invalid, unsigned, or conflicting callbacks are rejected.
    """
    try:
        result = verify_and_record_callback(request)

        return _json_response(
            {
                "ok": True,
                "verified": True,
                "duplicate": bool(
                    result.get("duplicate")
                ),
            },
            200,
        )

    except AdMobSSVConflictError as exc:
        print(
            f"AdMob SSV reward conflict: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "reward_conflict",
                "message": "The reward identifier conflicts with an existing transaction.",
            },
            409,
        )

    except AdMobSSVSignatureError as exc:
        print(
            f"AdMob SSV signature rejected: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "invalid_signature",
                "message": "The AdMob callback signature could not be verified.",
            },
            403,
        )

    except AdMobSSVConfigurationError as exc:
        print(
            f"AdMob SSV configuration error: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "ssv_configuration_error",
                "message": "The reward verification service is temporarily unavailable.",
            },
            503,
        )

    except AdMobSSVError as exc:
        print(
            f"AdMob SSV callback rejected: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "invalid_callback",
                "message": "The AdMob callback data is invalid.",
            },
            400,
        )

    except Exception as exc:
        print(
            f"Unexpected AdMob SSV error: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "ssv_internal_error",
                "message": "The reward verification service encountered an error.",
            },
            500,
        )


@admob_bp.route(
    "/admob-reward-status",
    methods=["POST", "OPTIONS"],
)
def admob_reward_status():
    """
    Let the app check whether Google has verified a specific reward.

    This endpoint does not consume the reward. The interpretation endpoint
    consumes it after the app submits the matching reward_id and user_id.
    """
    if request.method == "OPTIONS":
        response = make_response("", 204)
        response.headers["Cache-Control"] = "no-store"
        return response

    data = request.get_json(
        silent=True,
    ) or {}

    reward_id = _clean(
        data.get("reward_id")
    )
    user_id = _clean(
        data.get("user_id")
    )

    if not _REWARD_ID_PATTERN.fullmatch(reward_id):
        return _json_response(
            {
                "ok": False,
                "error": "invalid_reward_id",
                "status": "invalid",
            },
            400,
        )

    if not _USER_ID_PATTERN.fullmatch(user_id):
        return _json_response(
            {
                "ok": False,
                "error": "invalid_user_id",
                "status": "invalid",
            },
            400,
        )

    try:
        record = get_reward_record(
            reward_id
        )

    except AdMobSSVConfigurationError as exc:
        print(
            f"AdMob reward status configuration error: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "ssv_configuration_error",
                "status": "error",
            },
            503,
        )

    except Exception as exc:
        print(
            f"AdMob reward status lookup failed: {exc}",
            flush=True,
        )

        return _json_response(
            {
                "ok": False,
                "error": "reward_status_failed",
                "status": "error",
            },
            500,
        )

    if not record:
        return _json_response(
            {
                "ok": True,
                "verified": False,
                "status": "pending",
            },
            200,
        )

    if _clean(record.get("user_id")) != user_id:
        return _json_response(
            {
                "ok": True,
                "verified": False,
                "status": "pending",
            },
            200,
        )

    status = _clean(
        record.get("status")
    ).lower()

    consumed_at = _clean(
        record.get("consumed_at")
    )

    if status == "verified" and not consumed_at:
        return _json_response(
            {
                "ok": True,
                "verified": True,
                "status": "verified",
            },
            200,
        )

    if status == "consumed" or consumed_at:
        return _json_response(
            {
                "ok": True,
                "verified": False,
                "status": "consumed",
            },
            200,
        )

    return _json_response(
        {
            "ok": True,
            "verified": False,
            "status": "pending",
        },
        200,
    )
