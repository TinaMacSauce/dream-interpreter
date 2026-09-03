from __future__ import annotations

from flask import Blueprint, jsonify

from app.qa_access import (
    get_qa_grant_status,
    public_qa_access_metadata,
    qa_token_from_request,
)
from app.release_info import release_metadata
from app.teeth_doctrine import (
    build_teeth_doctrine_context,
    build_teeth_narration_facts,
)
from app.teeth_registry import get_teeth_registry_snapshot, public_registry_metadata


qa_bp = Blueprint("qa", __name__)

TEETH_QA_CONTRACT_VERSION = "teeth-qa-contract-v1"


@qa_bp.post("/qa/interpret")
def qa_interpret():
    """Run the real interpreter only when an isolated QA token is active."""
    token = qa_token_from_request()
    grant = get_qa_grant_status(token)
    if grant.get("active") is not True:
        response = jsonify(
            {
                "blocked": True,
                "reason": grant.get("reason") or "missing_token",
                "message": "An active QA access token is required.",
                "access": "blocked",
            }
        )
        response.status_code = 403
        response.headers["Cache-Control"] = "no-store"
        return response
    from app.services.interpreter_service import run_interpretation

    return run_interpretation()


@qa_bp.get("/qa/status")
def qa_status():
    registry = get_teeth_registry_snapshot()
    access = public_qa_access_metadata()
    response = jsonify(
        {
            "service": "dream-interpreter",
            "release": release_metadata(),
            "qa_access": access,
            "doctrine_registry": public_registry_metadata(registry),
            "ready": bool(
                registry.get("verified") is True
                and access.get("configured") is True
                and access.get("storage_ready") is True
            ),
        }
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response

# This surface is intentionally bounded to synthetic, founder-approved
# regression inputs. It cannot interpret arbitrary dreams and never touches
# access counters, paid entitlements, AI narration, or customer data.
TEETH_QA_CASES = (
    ("quantity_one", "My tooth fell out."),
    ("quantity_multiple", "Three of my teeth fell out."),
    ("ownership_other", "My sister's tooth fell out."),
    ("ownership_external_actor", "My sister pulled my tooth out."),
    ("painful_loss", "One of my teeth fell out and it hurt badly."),
    ("painless_loss", "One of my teeth fell out without pain."),
    (
        "blood_after_loss",
        "My tooth fell out and there was blood on the fallen tooth.",
    ),
    (
        "bleeding_gums_with_negations",
        "My gums were bleeding, but no tooth was loose and none fell out.",
    ),
    ("loose_without_loss", "My tooth was loose but did not fall out."),
    ("negated_loss", "My teeth did not fall out."),
    ("hypothetical_loss", "I thought my teeth might fall out."),
    (
        "genuine_terminal_ending",
        "My tooth fell out, but in the end the same tooth fitted firmly "
        "back into the same socket.",
    ),
    ("attempted_ending", "My tooth fell out and I tried to put it back."),
)


@qa_bp.get("/qa/teeth-regression")
def teeth_regression_contract():
    registry = get_teeth_registry_snapshot()
    cases = []
    for case_id, dream in TEETH_QA_CASES:
        cases.append(
            {
                "case_id": case_id,
                "dream": dream,
                "doctrine": build_teeth_doctrine_context(dream),
                "narration": build_teeth_narration_facts(dream),
            }
        )

    response = jsonify(
        {
            "contract_version": TEETH_QA_CONTRACT_VERSION,
            "release": release_metadata(),
            "doctrine_registry": public_registry_metadata(
                registry,
                include_rule_ids=True,
            ),
            "case_count": len(cases),
            "cases": cases,
        }
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response
