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

TEETH_QA_CONTRACT_VERSION = "teeth-qa-contract-v2"


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
BASE_TEETH_QA_CASES = (
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

ATTEMPT_BINDING_QA_CASES = (
    ("CTX-001-ATTEMPT-BIND-DREAMER-001", "My tooth fell out and I tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-NEGATED-001", "My tooth fell out, but I never tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-HYPOTHETICAL-001", "If my tooth fell out, I would try to put it back."),
    ("CTX-001-ATTEMPT-BIND-QUOTED-001", 'My tooth fell out. My aunt said, "I tried to put it back."'),
    ("CTX-001-ATTEMPT-BIND-WAKING-001", "My tooth fell out. After I woke up, I tried to put it back in my imagination."),
    ("CTX-001-ATTEMPT-BIND-OTHER-OWNER-001", "My sister's tooth fell out and she tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-EXTERNAL-ACTOR-001", "My tooth fell out and my sister tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-MULTI-OWNER-001", "My tooth fell out and I tried to put it back. My sister's tooth fell out and she left it there."),
    ("CTX-001-ATTEMPT-BIND-AMBIGUOUS-TARGET-001", "My tooth and my sister's tooth fell out. I tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-THEN-FIRM-001", "My tooth fell out. I tried to put it back, and then the same tooth fitted firmly back into the same socket."),
    ("CTX-001-ATTEMPT-BIND-THEN-SECOND-LOSS-001", "My left tooth fell out and I tried to put it back. Then another tooth fell out."),
    ("CTX-001-ATTEMPT-BIND-REPORTED-001", "My tooth fell out. My sister told me that she tried to put her tooth back yesterday."),
)

CONDITION_PROVENANCE_QA_CASES = (
    ("CTX-003-COND-PROV-GUMS-NEGATED-001", "My gums were bleeding, but no tooth was loose and none fell out."),
    ("CTX-003-COND-PROV-GUMS-ONLY-001", "My gums were bleeding."),
    ("CTX-003-COND-PROV-GUMS-RETAINED-001", "My gums were bleeding and every tooth stayed firm."),
    ("CTX-003-COND-PROV-LOOSE-NEGATED-LOSS-001", "My tooth was loose but did not fall out."),
    ("CTX-003-COND-PROV-WOBBLY-RETAINED-001", "My tooth was wobbly and stayed in my mouth."),
    ("CTX-003-COND-PROV-TWO-LOOSE-001", "Two of my teeth were loose, but neither fell out."),
    ("CTX-003-COND-PROV-OTHER-OWNER-001", "My sister's tooth was loose but did not fall out."),
    ("CTX-003-COND-PROV-NEGATED-LOOSE-THEN-LOSS-001", "My tooth was not loose, but it fell out."),
    ("CTX-003-COND-PROV-LOOSE-THEN-LOSS-001", "My tooth became loose, then it fell out."),
    ("CTX-003-COND-PROV-QUOTED-001", 'My aunt said, "My tooth is loose." My own tooth stayed firm.'),
    ("CTX-003-COND-PROV-HYPOTHETICAL-001", "If my tooth were loose, I would visit a dentist."),
    ("CTX-003-COND-PROV-MULTI-OWNER-001", "My gums were bleeding. My sister's tooth was loose but did not fall out."),
)

WARNING_CLAIM_PARTITION_QA_CASES = (
    ("CTX-003-CLAIM-PART-MULTI-OWNER-CONDITIONS-001", "My gums were bleeding. My sister's tooth was loose but did not fall out."),
    ("CTX-003-CLAIM-PART-SAME-OWNER-DISTINCT-CONDITIONS-001", "My gums were bleeding and my tooth was loose but did not fall out."),
    ("CTX-003-CLAIM-PART-TWO-OWNER-LOOSE-001", "My tooth was loose. My sister's tooth was loose."),
    ("CTX-003-CLAIM-PART-TWO-OTHER-OWNERS-001", "My sister's tooth was loose. My brother's tooth was loose."),
    ("CTX-003-CLAIM-PART-TWO-LOOSE-ONE-EVENT-001", "Two of my teeth were loose, but neither fell out."),
    ("CTX-003-CLAIM-PART-ONE-LOSS-MULTI-RULE-001", "My tooth fell out."),
    ("CTX-003-CLAIM-PART-LOSS-PLUS-OTHER-LOOSE-001", "My tooth fell out. My sister's tooth was loose but did not fall out."),
    ("CTX-003-CLAIM-PART-GUMS-PLUS-OTHER-LOSS-001", "My gums were bleeding. My sister's tooth fell out."),
    ("CTX-003-CLAIM-PART-LOOSE-THEN-LOSS-001", "My tooth became loose, then it fell out."),
    ("CTX-003-CLAIM-PART-QUOTED-PLUS-GUMS-001", 'My aunt said, "My tooth is loose." My gums were bleeding.'),
    ("CTX-003-CLAIM-PART-HYPOTHETICAL-PLUS-OTHER-001", "If my tooth were loose, I would worry. My sister's tooth was loose."),
    ("CTX-003-CLAIM-PART-NEGATED-PLUS-OTHER-001", "My tooth was not loose. My sister's tooth was loose."),
)

NARRATION_CLAIM_CONSUMPTION_QA_CASES = (
    ("CTX-003-NARR-CONS-MULTI-OWNER-CONDITIONS-001", "My gums were bleeding. My sister's tooth was loose but did not fall out."),
    ("CTX-003-NARR-CONS-SAME-OWNER-DISTINCT-CONDITIONS-001", "My gums were bleeding and my tooth was loose but did not fall out."),
    ("CTX-003-NARR-CONS-TWO-OWNER-LOOSE-001", "My tooth was loose. My sister's tooth was loose."),
    ("CTX-003-NARR-CONS-LOSS-PLUS-OTHER-LOOSE-001", "My tooth fell out. My sister's tooth was loose but did not fall out."),
    ("CTX-003-NARR-CONS-GUMS-PLUS-OTHER-LOSS-001", "My gums were bleeding. My sister's tooth fell out."),
    ("CTX-003-NARR-CONS-QUOTED-PLUS-GUMS-001", 'My aunt said, "My tooth is loose." My gums were bleeding.'),
    ("CTX-003-NARR-CONS-HYPOTHETICAL-PLUS-OTHER-001", "If my tooth were loose, I would worry. My sister's tooth was loose."),
    ("CTX-003-NARR-CONS-NEGATED-PLUS-OTHER-001", "My tooth was not loose. My sister's tooth was loose."),
    ("CTX-003-NARR-CONS-LOOSE-THEN-LOSS-001", "My tooth became loose, then it fell out."),
    ("CTX-003-NARR-CONS-ATTEMPT-THEN-SECOND-LOSS-001", "My tooth fell out and I tried to put it back. Then another tooth fell out."),
    ("CTX-003-NARR-CONS-TERMINAL-RETURN-001", "My tooth fell out, then the same tooth fitted firmly back into the same socket."),
    ("CTX-003-NARR-CONS-ONE-LOSS-MULTI-RULE-001", "My tooth fell out."),
)

TEETH_QA_CASES = (
    BASE_TEETH_QA_CASES
    + ATTEMPT_BINDING_QA_CASES
    + CONDITION_PROVENANCE_QA_CASES
    + WARNING_CLAIM_PARTITION_QA_CASES
    + NARRATION_CLAIM_CONSUMPTION_QA_CASES
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
