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

    We do not treat a random typed email as proof of ownership.
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
    """
    Subscription users must already have a verified premium session.

    This prevents someone from typing another customer's email and
    receiving subscription access.
    """
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
    Reads the local subscriber record written by Stripe webhook or verified
    checkout success.

    This does NOT prove the current visitor owns the email.
    Ownership proof is handled by _premium_session_matches().
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
    """
    Live Stripe lookup by email.

    Kept for admin/backfill/debug use, but has_active_access() does not use
    this as proof that the current visitor owns the email.
    """
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

    except Exception:
        return False, ""


# ============================================================
# ACCESS CHECK
# ============================================================

def has_active_access(email: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Access rules:

    Subscription:
      - Must have a local active subscriber record
      - Must also have a verified premium session for the same email

    Dream pack:
      - Must have an active dream pack record
      - Must also have a buyer session for the same email

    This prevents access from being granted just because someone typed
    another paying customer's email.
    """
    email = normalize_email(email)

    if not validate_email(email):
        return False, {}

    # --------------------------------------------------------
    # Subscription access
    # --------------------------------------------------------

    local_active, customer_id = local_active_subscription_for_email(email)

    if local_active and _premium_session_matches(email):
        return True, {
            "type": "subscription",
            "customer_id": customer_id,
        }

    # --------------------------------------------------------
    # Dream pack access
    # --------------------------------------------------------

    pack = get_dream_pack_status(email)

    if pack.get("active") and _buyer_session_matches(email):
        return True, {
            "type": "dream_pack",
            "details": pack,
        }

    # --------------------------------------------------------
    # Do not mark active users inactive just because this visitor
    # failed session verification.
    # --------------------------------------------------------

    if not local_active:
        mark_subscriber(
            email=email,
            is_active=False,
        )

    return False, {
        "type": "none",
    }