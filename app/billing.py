from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    import stripe
except Exception:
    stripe = None

from app.access import (
    get_dream_pack_status,
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


def stripe_active_subscription_for_email(email: str) -> Tuple[bool, str]:
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


def has_active_access(email: str) -> Tuple[bool, Dict[str, Any]]:
    email = normalize_email(email)

    if not validate_email(email):
        return False, {}

    is_active, customer_id = stripe_active_subscription_for_email(email)

    if is_active:
        mark_subscriber(
            email=email,
            is_active=True,
            stripe_customer_id=customer_id,
        )

        return True, {
            "type": "subscription",
            "customer_id": customer_id,
        }

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
