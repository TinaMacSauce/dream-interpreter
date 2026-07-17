from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, make_response, request

from app.services.admob_ssv_service import (
    AdMobSSVConfigurationError,
    AdMobSSVConflictError,
    AdMobSSVError,
    AdMobSSVSignatureError,
    verify_and_record_callback,
)


admob_bp = Blueprint("admob", __name__)


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
