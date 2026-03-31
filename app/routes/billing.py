from flask import Blueprint, jsonify, make_response, redirect, request, url_for

from app.access import (
    get_dream_pack_status,
    normalize_email if False else None,
    persist_email_to_session,
    set_buyer_session,
    set_premium_session,
    mark_dream_pack_purchase,
)
