from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    import stripe
except Exception:
    stripe = None

from app.access import (
    get_dream_pack_status,
    get_session_email,
    is_premium_session,
    load_subscribers,
    mark_subscriber,
)
from app.config import Config
from app.utils import (
    normalize_email,
    validate_email,
)


STRIPE_API_VERSION = "2025-03-31.basil"


def _stripe_available() -> bool:
    return bool(stripe and Config.STRIPE_SECRET_KEY)


def configure_stripe() -> bool:
    if not _stripe_available():
        return False

    stripe.api_key = Config.STRIPE_SECRET_KEY
    stripe.api_version = STRIPE_API_VERSION
    return True


def stripe_config_ok() -> bool:
    return bool(
        stripe
        and Config.STRIPE_SECRET_KEY
        and Config.DEFAULT_STRIPE_PRICE_ID
    )


def stripe_dream_pack_ok() -> bool:
    return bool(
        stripe
        and Config.STRIPE_SECRET_KEY
        and Config.PRICE_DREAM_PACK
    )


# ============================================================
# SESSION TRUST HELPERS
# ============================================================

def _session_email_matches(email: str) -> bool:
    email_n = normalize_email(email)
    session_email = normalize_email(get_session_email())

    return bool(
        validate_email(email_n)
        and session_email
        and session_email == email_n
    )


def _buyer_session_matches(email: str) -> bool:
    """
    Dream-pack buyers are marked in the session by set_buyer_session().
    That sets:
      session["subscriber_email"] = email
      session["premium"] = False
      session["buyer_set_at"] = iso_now()
    """
    try:
        from flask import session

        email_n = normalize_email(email)

        return bool(
            validate_email(email_n)
            and _session_email_matches(email_n)
            and session.get("buyer_set_at")
            and session.get("premium") is False
        )

    except Exception:
        return False


def _premium_session_matches(email: str) -> bool:
    email_n = normalize_email(email)

    return bool(
        validate_email(email_n)
        and _session_email_matches(email_n)
        and is_premium_session()
    )


# ============================================================
# LOCAL SUBSCRIBER HELPERS
# ============================================================

def local_active_subscription_for_email(email: str) -> Tuple[bool, str]:
    """
    Read the local subscriber record written by a Stripe webhook or a
    verified checkout success callback.
    """
    email_n = normalize_email(email)

    if not validate_email(email_n):
        return False, ""

    try:
        subscribers = load_subscribers()
        rec = subscribers.get(email_n)

        if not isinstance(rec, dict):
            return False, ""

        is_active = bool(rec.get("is_active"))
        customer_id = rec.get("stripe_customer_id") or ""

        return is_active, customer_id

    except Exception:
        return False, ""


# ============================================================
# STRIPE LIVE LOOKUP
# ============================================================

def stripe_active_subscription_for_email(email: str) -> Tuple[bool, str]:
    """Look up active Stripe subscriptions for an email address."""
    email = normalize_email(email)

    if not validate_email(email):
        return False, ""

    if not configure_stripe():
        return False, ""

    try:
        customers = stripe.Customer.list(
            email=email,
            limit=10,
        )

        for customer in customers.data or []:
            customer_id = customer.get("id") or ""

            if not customer_id:
                continue

            subscriptions = stripe.Subscription.list(
                customer=customer_id,
                status="all",
                limit=20,
                expand=["data.default_payment_method"],
            )

            for subscription in subscriptions.data or []:
                status = (
                    subscription.get("status")
                    or ""
                ).lower()

                cancel_at_period_end = bool(
                    subscription.get("cancel_at_period_end")
                )

                if status in {"active", "trialing"}:
                    return True, customer_id

                if status == "past_due" and not cancel_at_period_end:
                    return True, customer_id

        return False, ""

    except Exception as exc:
        print(
            f"Stripe subscription lookup failed for {email}: {exc}",
            flush=True,
        )
        return False, ""


# ============================================================
# ACCESS CHECK
# ============================================================

def has_active_access(email: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Restore paid access by the checkout email saved in the app.

    Subscription access is accepted from the local webhook-backed record.
    If that record is missing, Stripe is checked directly and the local
    record is repaired automatically.

    Dream Pack access is restored from its active email record.
    """
    email = normalize_email(email)

    if not validate_email(email):
        return False, {}

    # --------------------------------------------------------
    # Subscription access
    # --------------------------------------------------------

    local_active, customer_id = local_active_subscription_for_email(email)

    if local_active:
        return True, {
            "type": "subscription",
            "customer_id": customer_id,
        }

    stripe_active, stripe_customer_id = (
        stripe_active_subscription_for_email(email)
    )

    if stripe_active:
        mark_subscriber(
            email=email,
            is_active=True,
            stripe_customer_id=stripe_customer_id,
        )

        return True, {
            "type": "subscription",
            "customer_id": stripe_customer_id,
        }

    # --------------------------------------------------------
    # Dream Pack access
    # --------------------------------------------------------

    pack = get_dream_pack_status(email)

    if pack.get("active"):
        return True, {
            "type": "dream_pack",
            "details": pack,
        }

    mark_subscriber(
        email=email,
        is_active=False,
    )

    return False, {
        "type": "none",
    }
