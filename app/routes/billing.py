from flask import Blueprint, jsonify, make_response, redirect, request, url_for

from app.access import (
    mark_dream_pack_purchase,
    persist_email_to_session,
    set_buyer_session,
    set_premium_session,
)
from app.billing import has_active_access, stripe, stripe_config_ok, stripe_dream_pack_ok
from app.config import Config
from app.utils import normalize_email, validate_email

billing_bp = Blueprint("billing", __name__)


def _extract_email_from_request() -> str:
    data = request.get_json(silent=True) or {}
    return normalize_email(
        data.get("email")
        or request.args.get("email")
        or request.form.get("email")
        or ""
    )


def _safe_return_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return Config.RETURN_URL
    if value.startswith("http://") or value.startswith("https://") or value.startswith("/"):
        return value
    return Config.RETURN_URL


def _empty_dream_pack() -> dict:
    return {
        "active": False,
        "uses_remaining": 0,
        "expires_at": "",
    }


def _clear_free_cookie(resp):
    resp.set_cookie(
        Config.COOKIE_NAME,
        "0",
        max_age=0,
        samesite=Config.SESSION_COOKIE_SAMESITE,
        secure=Config.SESSION_COOKIE_SECURE,
    )
    return resp


@billing_bp.route("/check-access", methods=["POST", "OPTIONS"])
def check_access():
    if request.method == "OPTIONS":
        return make_response("", 204)

    email = _extract_email_from_request()

    if not validate_email(email):
        return jsonify(
            {
                "ok": False,
                "error": "Please enter a valid email.",
                "access": "invalid",
                "is_paid": False,
                "email": "",
                "free_uses_left": None,
                "dream_pack": _empty_dream_pack(),
                "return_url": Config.RETURN_URL,
            }
        ), 400

    persist_email_to_session(email)
    access_ok, access_meta = has_active_access(email)
    access_type = (access_meta.get("type") or "").strip().lower()

    if access_ok and access_type == "subscription":
        set_premium_session(email)

        resp = make_response(
            jsonify(
                {
                    "ok": True,
                    "access": "active",
                    "is_paid": True,
                    "email": email,
                    "free_uses_left": None,
                    "dream_pack": _empty_dream_pack(),
                    "return_url": Config.RETURN_URL,
                }
            )
        )
        return _clear_free_cookie(resp)

    if access_ok and access_type == "dream_pack":
        pack_details = access_meta.get("details") or _empty_dream_pack()
        set_buyer_session(email)

        resp = make_response(
            jsonify(
                {
                    "ok": True,
                    "access": "dream_pack_active",
                    "is_paid": False,
                    "email": email,
                    "free_uses_left": None,
                    "dream_pack": {
                        "active": bool(pack_details.get("active", False)),
                        "uses_remaining": int(pack_details.get("uses_remaining", 0) or 0),
                        "expires_at": pack_details.get("expires_at", "") or "",
                    },
                    "return_url": Config.RETURN_URL,
                }
            )
        )
        return _clear_free_cookie(resp)

    return jsonify(
        {
            "ok": False,
            "access": "inactive",
            "is_paid": False,
            "email": email,
            "free_uses_left": None,
            "dream_pack": _empty_dream_pack(),
            "message": "No active subscription or dream pack found.",
            "return_url": Config.RETURN_URL,
        }
    ), 402


@billing_bp.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not stripe_config_ok():
        return make_response("Stripe not configured.", 500)

    email = _extract_email_from_request()
    if not validate_email(email):
        return make_response("Missing email.", 400)

    persist_email_to_session(email)
    stripe.api_key = Config.STRIPE_SECRET_KEY

    try:
        requested_return = _safe_return_url(
            request.form.get("return")
            or request.args.get("return")
            or request.headers.get("X-Return-Url")
            or Config.RETURN_URL
        )

        checkout = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": Config.DEFAULT_STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=email,
            metadata={
                "purchase_type": "subscription",
                "email": email,
                "return_url": requested_return,
            },
            success_url=(
                url_for("billing.payment_success", _external=True)
                + f"?email={email}&return={requested_return}"
            ),
            cancel_url=(
                url_for("home.upgrade", _external=True)
                + f"?email={email}&return={requested_return}"
            ),
        )
        return redirect(checkout.url, code=303)
    except Exception as e:
        return make_response(f"Stripe error: {str(e)}", 500)


@billing_bp.route("/create-dream-pack-checkout-session", methods=["POST", "GET"])
def create_dream_pack_checkout_session():
    if not stripe_dream_pack_ok():
        return make_response("Dream pack Stripe config missing.", 500)

    email = _extract_email_from_request()
    if not validate_email(email):
        return make_response("Missing email for dream pack checkout.", 400)

    persist_email_to_session(email)
    stripe.api_key = Config.STRIPE_SECRET_KEY

    try:
        requested_return = _safe_return_url(
            request.form.get("return")
            or request.args.get("return")
            or request.headers.get("X-Return-Url")
            or Config.RETURN_URL
        )

        checkout = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": Config.PRICE_DREAM_PACK, "quantity": 1}],
            customer_email=email,
            metadata={
                "purchase_type": "dream_pack",
                "dream_pack_uses": str(Config.DREAM_PACK_USES),
                "dream_pack_hours": str(Config.DREAM_PACK_HOURS),
                "email": email,
                "return_url": requested_return,
            },
            success_url=(
                url_for("billing.dream_pack_success", _external=True)
                + f"?email={email}&return={requested_return}"
            ),
            cancel_url=(
                url_for("home.upgrade", _external=True)
                + f"?email={email}&return={requested_return}"
            ),
        )
        return redirect(checkout.url, code=303)
    except Exception as e:
        return make_response(f"Stripe dream pack error: {str(e)}", 500)


@billing_bp.route("/payment-success", methods=["GET"])
def payment_success():
    email = normalize_email(request.args.get("email") or "")
    requested_return = _safe_return_url(request.args.get("return") or Config.RETURN_URL)

    if validate_email(email):
        persist_email_to_session(email)
        access_ok, access_meta = has_active_access(email)
        if access_ok and (access_meta.get("type") or "").strip().lower() == "subscription":
            set_premium_session(email)

    resp = redirect(requested_return, code=302)
    return _clear_free_cookie(resp)


@billing_bp.route("/dream-pack-success", methods=["GET"])
def dream_pack_success():
    email = normalize_email(request.args.get("email") or "")
    requested_return = _safe_return_url(request.args.get("return") or Config.RETURN_URL)

    if validate_email(email):
        persist_email_to_session(email)
        set_buyer_session(email)
        mark_dream_pack_purchase(
            email,
            uses=Config.DREAM_PACK_USES,
            hours=Config.DREAM_PACK_HOURS,
        )

    resp = redirect(requested_return, code=302)
    return _clear_free_cookie(resp)


@billing_bp.route("/webhook", methods=["POST"])
def stripe_webhook():
    if not stripe or not Config.STRIPE_SECRET_KEY or not Config.STRIPE_WEBHOOK_SECRET:
        return ("not configured", 400)

    stripe.api_key = Config.STRIPE_SECRET_KEY
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            Config.STRIPE_WEBHOOK_SECRET,
        )
    except Exception:
        return ("bad signature", 400)

    event_type = event.get("type", "")
    data_obj = (event.get("data") or {}).get("object") or {}

    if event_type == "checkout.session.completed":
        metadata = data_obj.get("metadata") or {}
        email = normalize_email(metadata.get("email") or data_obj.get("customer_email") or "")
        mode = (data_obj.get("mode") or "").strip().lower()
        purchase_type = (metadata.get("purchase_type") or "").strip().lower()

        if validate_email(email) and mode == "payment" and purchase_type == "dream_pack":
            mark_dream_pack_purchase(
                email,
                uses=Config.DREAM_PACK_USES,
                hours=Config.DREAM_PACK_HOURS,
            )

    return ("ok", 200)
