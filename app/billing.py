from typing import Any, Dict, Tuple

try:
    import stripe
except Exception:
    stripe = None

from app.config import Config
from app.access import get_dream_pack_status, mark_subscriber
from app.utils import normalize_email, validate_email


def stripe_config_ok() -> bool:
    return bool(stripe and Config.STRIPE_SECRET_KEY and Config.DEFAULT_STRIPE_PRICE_ID)


def stripe_dream_pack_ok() -> bool:
    return bool(stripe and Config.STRIPE_SECRET_KEY and Config.PRICE_DREAM_PACK)


def stripe_active_subscription_for_email(email: str) -> Tuple[bool, str]:
    email = normalize_email(email)
    if not validate_email(email):
        return False, ""

    if not stripe or not Config.STRIPE_SECRET_KEY:
        return False, ""

    stripe.api_key = Config.STRIPE_SECRET_KEY

    try:
        customers = stripe.Customer.list(email=email, limit=10)
        for customer in (customers.data or []):
            customer_id = customer.get("id") or ""
            if not customer_id:
                continue

            subs = stripe.Subscription.list(customer=customer_id, status="all", limit=10)
            for sub in (subs.data or []):
                status = (sub.get("status") or "").lower()
                if status in {"active", "trialing"}:
                    return True, customer_id

        return False, ""
    except Exception:
        return False, ""


def has_active_access(email: str) -> Tuple[bool, Dict[str, Any]]:
    if not validate_email(email):
        return False, {}

    is_active, customer_id = stripe_active_subscription_for_email(email)
    if is_active:
        mark_subscriber(email, True, customer_id)
        return True, {"type": "subscription", "customer_id": customer_id}

    pack = get_dream_pack_status(email)
    if pack["active"]:
        return True, {"type": "dream_pack", "details": pack}

    return False, {}
