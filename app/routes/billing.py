from __future__ import annotations

import time
from typing import Any, Dict
from urllib.parse import urlencode

from flask import (
    Blueprint,
    jsonify,
    make_response,
    redirect,
    request,
    url_for,
)

from app.access import (
    get_dream_pack_status,
    mark_dream_pack_purchase,
    mark_subscriber,
    persist_email_to_session,
    set_buyer_session,
    set_premium_session,
)

from app.billing import (
    has_active_access,
    stripe,
    stripe_config_ok,
    stripe_dream_pack_ok,
)

from app.config import Config
from app.utils import normalize_email, validate_email


billing_bp = Blueprint("billing", __name__)

STRIPE_API_VERSION = "2025-03-31.basil"


# ============================================================
# Helpers
# ============================================================

def _json_response(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _empty_options_response():
    return make_response("", 204)


def _empty_dream_pack() -> Dict[str, Any]:
    return {
        "active": False,
        "uses_remaining": 0,
        "expires_at": "",
    }


def _extract_email_from_request() -> str:
    data = request.get_json(silent=True, force=False) or {}

    return normalize_email(
        data.get("email")
        or request.args.get("email")
        or request.form.get("email")
        or ""
    )


def _safe_return_url(value: str | None) -> str:
    value = (value or "").strip()

    if not value:
        return Config.RETURN_URL

    allowed_prefixes = (
        "https://jamaicantruestories.com",
        "https://www.jamaicantruestories.com",
        "https://interpreter.jamaicantruestories.com",
        "/",
    )

    if value.startswith(allowed_prefixes):
        return value

    return Config.RETURN_URL


def _requested_return_url() -> str:
    return _safe_return_url(
        request.form.get("return")
        or request.args.get("return")
        or request.headers.get("X-Return-Url")
        or Config.RETURN_URL
    )


def _stripe_ready() -> bool:
    return bool(stripe and Config.STRIPE_SECRET_KEY)


def _configure_stripe() -> None:
    if not stripe:
        return

    stripe.api_key = Config.STRIPE_SECRET_KEY
    stripe.api_version = STRIPE_API_VERSION


def _clear_free_cookie(resp):
    resp.set_cookie(
        Config.COOKIE_NAME,
        "0",
        expires=0,
        samesite=Config.SESSION_COOKIE_SAMESITE,
        secure=Config.SESSION_COOKIE_SECURE,
        httponly=True,
    )

    return resp


def _build_success_url(endpoint: str, requested_return: str) -> str:
    """
    Stripe replaces {CHECKOUT_SESSION_ID} after successful checkout.

    We verify that session ID on the success route before granting access.
    This prevents someone from manually visiting a success URL with an email.
    """
    base_url = url_for(endpoint, _external=True)

    query = urlencode(
        {
            "session_id": "{CHECKOUT_SESSION_ID}",
            "return": requested_return,
        }
    )

    # Stripe requires the braces to stay literal.
    query = query.replace("%7B", "{").replace("%7D", "}")

    return f"{base_url}?{query}"


def _build_cancel_url(email: str, requested_return: str) -> str:
    return url_for(
        "interpreter.upgrade",
        _external=True,
        email=email,
        **{"return": requested_return},
    )


def _checkout_error(message: str, status: int = 400):
    return make_response(message, status)


def _dream_pack_payload(pack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "active": bool(pack.get("active")),
        "uses_remaining": int(pack.get("uses_remaining", 0) or 0),
        "expires_at": pack.get("expires_at", "") or "",
    }


# ============================================================
# Access Check
# ============================================================

@billing_bp.route("/check-access", methods=["POST", "OPTIONS"])
def check_access():
    if request.method == "OPTIONS":
        return _empty_options_response()

    email = _extract_email_from_request()

    if not validate_email(email):
        return _json_response(
            {
                "ok": False,
                "error": "invalid_email",
                "message": "Please enter a valid email.",
                "access": "invalid",
                "is_paid": False,
                "email": "",
                "dream_pack": _empty_dream_pack(),
                "return_url": Config.RETURN_URL,
            },
            400,
        )

    persist_email_to_session(email)

    try:
        access_ok, access_meta = has_active_access(email)

    except Exception as exc:
        print(f"Access lookup failed for {email}: {exc}", flush=True)

        return _json_response(
            {
                "ok": False,
                "error": "access_lookup_failed",
                "message": "Access lookup failed. Please try again.",
                "access": "error",
                "is_paid": False,
                "email": email,
                "dream_pack": _empty_dream_pack(),
                "return_url": Config.RETURN_URL,
            },
            500,
        )

    access_type = (access_meta.get("type") or "").strip().lower()

    if access_ok and access_type == "subscription":
        set_premium_session(email)

        resp = _json_response(
            {
                "ok": True,
                "access": "active",
                "access_type": "subscription",
                "is_paid": True,
                "email": email,
                "dream_pack": _empty_dream_pack(),
                "return_url": Config.RETURN_URL,
            }
        )

        return _clear_free_cookie(resp)

    if access_ok and access_type == "dream_pack":
        set_buyer_session(email)

        pack = access_meta.get("details") or get_dream_pack_status(email)

        resp = _json_response(
            {
                "ok": True,
                "access": "dream_pack_active",
                "access_type": "dream_pack",
                "is_paid": False,
                "email": email,
                "dream_pack": _dream_pack_payload(pack),
                "return_url": Config.RETURN_URL,
            }
        )

        return _clear_free_cookie(resp)

    return _json_response(
        {
            "ok": False,
            "error": "payment_required",
            "message": "No active subscription or dream pack found.",
            "access": "inactive",
            "is_paid": False,
            "email": email,
            "dream_pack": _empty_dream_pack(),
            "return_url": Config.RETURN_URL,
        },
        402,
    )


# ============================================================
# Subscription Checkout
# ============================================================

@billing_bp.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not stripe_config_ok():
        return _checkout_error("Stripe subscription config missing.", 500)

    email = _extract_email_from_request()

    if not validate_email(email):
        return _checkout_error("Missing valid email.", 400)

    persist_email_to_session(email)
    _configure_stripe()

    requested_return = _requested_return_url()

    try:
        checkout = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=email,
            billing_address_collection="auto",
            allow_promotion_codes=True,
            expires_at=int(time.time()) + 1800,
            line_items=[
                {
                    "price": Config.DEFAULT_STRIPE_PRICE_ID,
                    "quantity": 1,
                }
            ],
            metadata={
                "purchase_type": "subscription",
                "email": email,
                "return_url": requested_return,
            },
            success_url=_build_success_url(
                "billing.payment_success",
                requested_return,
            ),
            cancel_url=_build_cancel_url(
                email,
                requested_return,
            ),
        )

        return redirect(checkout.url, code=303)

    except Exception as exc:
        print(f"Stripe subscription checkout error: {exc}", flush=True)
        return _checkout_error("Stripe checkout failed. Please try again.", 500)


# ============================================================
# Dream Pack Checkout
# ============================================================

@billing_bp.route("/create-dream-pack-checkout-session", methods=["POST", "GET"])
def create_dream_pack_checkout_session():
    if not stripe_dream_pack_ok():
        return _checkout_error("Dream pack Stripe config missing.", 500)

    email = _extract_email_from_request()

    if not validate_email(email):
        return _checkout_error("Missing valid email.", 400)

    persist_email_to_session(email)
    _configure_stripe()

    requested_return = _requested_return_url()

    try:
        checkout = stripe.checkout.Session.create(
            mode="payment",
            customer_email=email,
            allow_promotion_codes=True,
            expires_at=int(time.time()) + 1800,
            line_items=[
                {
                    "price": Config.PRICE_DREAM_PACK,
                    "quantity": 1,
                }
            ],
            metadata={
                "purchase_type": "dream_pack",
                "dream_pack_uses": str(Config.DREAM_PACK_USES),
                "dream_pack_hours": str(Config.DREAM_PACK_HOURS),
                "email": email,
                "return_url": requested_return,
            },
            success_url=_build_success_url(
                "billing.dream_pack_success",
                requested_return,
            ),
            cancel_url=_build_cancel_url(
                email,
                requested_return,
            ),
        )

        return redirect(checkout.url, code=303)

    except Exception as exc:
        print(f"Stripe dream pack checkout error: {exc}", flush=True)
        return _checkout_error("Dream pack checkout failed. Please try again.", 500)


# ============================================================
# Payment Success
# ============================================================

@billing_bp.route("/payment-success", methods=["GET"])
def payment_success():
    session_id = (request.args.get("session_id") or "").strip()

    requested_return = _safe_return_url(
        request.args.get("return") or Config.RETURN_URL
    )

    if not session_id:
        return redirect(requested_return, code=302)

    try:
        if not _stripe_ready():
            return redirect(requested_return, code=302)

        _configure_stripe()

        checkout = stripe.checkout.Session.retrieve(
            session_id,
            expand=["subscription"],
        )

        mode = (checkout.get("mode") or "").strip().lower()

        if mode != "subscription":
            return redirect(requested_return, code=302)

        email = normalize_email(
            checkout.get("customer_email")
            or (
                (checkout.get("customer_details") or {}).get("email")
            )
            or ""
        )

        if not validate_email(email):
            return redirect(requested_return, code=302)

        subscription = checkout.get("subscription") or {}
        status = ""
        customer_id = ""

        if isinstance(subscription, dict):
            status = (subscription.get("status") or "").strip().lower()
            customer_id = subscription.get("customer") or ""
        else:
            sub = stripe.Subscription.retrieve(subscription)
            status = (sub.get("status") or "").strip().lower()
            customer_id = sub.get("customer") or ""

        if status in {"active", "trialing"}:
            persist_email_to_session(email)
            set_premium_session(email)

            mark_subscriber(
                email=email,
                is_active=True,
                stripe_customer_id=customer_id,
            )

    except Exception as exc:
        print(f"Subscription success verification failed: {exc}", flush=True)

    resp = redirect(requested_return, code=302)

    return _clear_free_cookie(resp)


# ============================================================
# Dream Pack Success
# ============================================================

@billing_bp.route("/dream-pack-success", methods=["GET"])
def dream_pack_success():
    session_id = (request.args.get("session_id") or "").strip()

    requested_return = _safe_return_url(
        request.args.get("return") or Config.RETURN_URL
    )

    if not session_id:
        return redirect(requested_return, code=302)

    try:
        if not _stripe_ready():
            return redirect(requested_return, code=302)

        _configure_stripe()

        checkout = stripe.checkout.Session.retrieve(session_id)

        mode = (checkout.get("mode") or "").strip().lower()
        payment_status = (checkout.get("payment_status") or "").strip().lower()
        metadata = checkout.get("metadata") or {}
        purchase_type = (metadata.get("purchase_type") or "").strip().lower()

        if (
            mode != "payment"
            or purchase_type != "dream_pack"
            or payment_status not in {"paid", "no_payment_required"}
        ):
            return redirect(requested_return, code=302)

        email = normalize_email(
            checkout.get("customer_email")
            or (
                (checkout.get("customer_details") or {}).get("email")
            )
            or metadata.get("email")
            or ""
        )

        if not validate_email(email):
            return redirect(requested_return, code=302)

        persist_email_to_session(email)
        set_buyer_session(email)

        mark_dream_pack_purchase(
            email=email,
            uses=Config.DREAM_PACK_USES,
            hours=Config.DREAM_PACK_HOURS,
        )

    except Exception as exc:
        print(f"Dream pack success verification failed: {exc}", flush=True)

    resp = redirect(requested_return, code=302)

    return _clear_free_cookie(resp)


# ============================================================
# Stripe Webhook
# ============================================================

@billing_bp.route("/webhook", methods=["POST"])
def stripe_webhook():
    if not _stripe_ready() or not Config.STRIPE_WEBHOOK_SECRET:
        return ("not configured", 400)

    _configure_stripe()

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            Config.STRIPE_WEBHOOK_SECRET,
        )

    except stripe.error.SignatureVerificationError:
        return ("bad signature", 400)

    except ValueError:
        return ("invalid payload", 400)

    event_type = event.get("type", "")

    data_obj = (
        (event.get("data") or {}).get("object")
        or {}
    )

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data_obj)

    elif event_type in {
        "customer.subscription.deleted",
        "customer.subscription.paused",
    }:
        _handle_subscription_inactive(data_obj)

    elif event_type in {
        "customer.subscription.updated",
        "customer.subscription.resumed",
    }:
        _handle_subscription_updated(data_obj)

    return ("ok", 200)


def _handle_checkout_completed(data_obj: Dict[str, Any]) -> None:
    metadata = data_obj.get("metadata") or {}

    email = normalize_email(
        metadata.get("email")
        or data_obj.get("customer_email")
        or ""
    )

    mode = (data_obj.get("mode") or "").strip().lower()

    purchase_type = (
        metadata.get("purchase_type") or ""
    ).strip().lower()

    if not validate_email(email):
        return

    if mode == "payment" and purchase_type == "dream_pack":
        mark_dream_pack_purchase(
            email=email,
            uses=Config.DREAM_PACK_USES,
            hours=Config.DREAM_PACK_HOURS,
        )

        return

    if mode == "subscription":
        subscription_id = data_obj.get("subscription")

        if not subscription_id:
            return

        try:
            sub = stripe.Subscription.retrieve(subscription_id)

            status = (sub.get("status") or "").lower()
            customer_id = sub.get("customer") or ""

            mark_subscriber(
                email=email,
                is_active=status in {"active", "trialing"},
                stripe_customer_id=customer_id,
            )

        except Exception as exc:
            print(f"Subscription retrieve failed: {exc}", flush=True)


def _handle_subscription_updated(data_obj: Dict[str, Any]) -> None:
    status = (data_obj.get("status") or "").lower()
    customer_id = data_obj.get("customer") or ""

    if not customer_id:
        return

    try:
        customer = stripe.Customer.retrieve(customer_id)
        email = normalize_email(customer.get("email") or "")

        if not validate_email(email):
            return

        mark_subscriber(
            email=email,
            is_active=status in {"active", "trialing"},
            stripe_customer_id=customer_id,
        )

    except Exception as exc:
        print(f"Subscription update handling failed: {exc}", flush=True)


def _handle_subscription_inactive(data_obj: Dict[str, Any]) -> None:
    customer_id = data_obj.get("customer") or ""

    if not customer_id:
        return

    try:
        customer = stripe.Customer.retrieve(customer_id)
        email = normalize_email(customer.get("email") or "")

        if not validate_email(email):
            return

        mark_subscriber(
            email=email,
            is_active=False,
            stripe_customer_id=customer_id,
        )

    except Exception as exc:
        print(f"Subscription inactive handling failed: {exc}", flush=True)