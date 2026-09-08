"""Fail-closed claim-to-narration binding for approved Teeth warnings.

This module changes no doctrine.  It records exactly which already-released
claim produced each user-visible clause and rejects provenance drift before a
structured narration can be released.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


NARRATION_CLAIM_CONTRACT_VERSION = "narration-claim-consumption/1.0"


def _owner_label(owner_ids: Iterable[str]) -> str:
    owner = next(iter(owner_ids), "unknown")
    if owner == "dreamer":
        return "Your"
    return f"Your {owner}'s"


def _safety_ids(family: str) -> List[str]:
    values = ["tradition_attribution", "non_guarantee"]
    if family in {"loose_sickness", "broken_sickness", "rotten_sickness"}:
        values.append("non_diagnosis")
    if family == "tooth_loss":
        values.append("non_prediction")
    return values


def _atomic_text(claim: Dict[str, Any]) -> str:
    family = str(claim.get("warning_family") or "")
    owner = _owner_label(claim.get("owner_ids") or [])
    if family == "tooth_loss":
        return (
            f"{owner} fallen tooth is treated in Jamaican and wider Caribbean dream tradition "
            "as a death-associated omen warning, not as a prediction or guarantee."
        )
    if family == "bleeding_gums":
        return (
            f"{owner} standalone bleeding gums are treated in Jamaican True Stories doctrine "
            "as a pre-warning that a bad omen may be approaching, not as a guaranteed outcome."
        )
    labels = {
        "loose_sickness": "loose or wobbly tooth",
        "broken_sickness": "broken or cracked tooth",
        "rotten_sickness": "rotten or decayed tooth",
    }
    label = labels.get(family, "tooth condition")
    return (
        f"{owner} {label} is treated in Jamaican True Stories doctrine as a sickness warning, "
        "not a medical diagnosis or a guarantee of illness."
    )


def _certainty_profile(doctrine: Dict[str, Any]) -> Dict[str, str]:
    return {
        "warning_presence": "present",
        "warning_severity": (
            "heightened" if doctrine.get("severity_modifier") == "increased" else "not_scaled"
        ),
        "rule_match_confidence": "approved_rule_match",
        "binding_confidence": "exact_claim_binding",
        "predictive_certainty": "none",
    }


def build_narration_claim_consumption(
    doctrine: Dict[str, Any],
    *,
    legacy_lead: str = "",
    legacy_details: Iterable[str] = (),
) -> Dict[str, Any]:
    graph = doctrine.get("context_graph")
    graph = graph if isinstance(graph, dict) else {}
    manifest = list(graph.get("claim_manifest") or [])
    atomics = [
        claim for claim in manifest
        if claim.get("claim_scope") == "atomic_warning" and claim.get("released") is True
    ]
    structural = [
        claim for claim in manifest
        if claim.get("claim_scope") in {"structural_narration", "structural_terminal"}
        and claim.get("released") is True
    ]
    compound = next(
        (claim for claim in manifest if claim.get("claim_scope") == "compound_presentation"),
        None,
    )

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for claim in atomics:
        key = (tuple(claim.get("owner_ids") or []), claim.get("warning_family"))
        grouped.setdefault(key, []).append(claim)

    clauses: List[Dict[str, Any]] = []
    for index, claims_in_clause in enumerate(grouped.values(), start=1):
        claim = claims_in_clause[0]
        rendered_text = _atomic_text(claim)
        if len(grouped) == 1 and legacy_lead:
            legacy_parts = [str(legacy_lead).strip()]
            legacy_parts.extend(
                str(detail).strip() for detail in legacy_details
                if str(detail).strip() and "attempt to put the tooth back" not in str(detail).lower()
            )
            rendered_text = " ".join(part for part in legacy_parts if part)
        clauses.append({
            "narration_clause_id": f"narration-warning-{index}",
            "render_slot": f"warning_member_{index}",
            "clause_type": "warning_member",
            "rendered_text": rendered_text,
            "output_span": {},
            "member_claim_ids": [item["claim_id"] for item in claims_in_clause],
            "owner_ids": list(claim.get("owner_ids") or []),
            "event_chain_ids": list(dict.fromkeys(
                chain_id
                for item in claims_in_clause
                for chain_id in item.get("event_chain_ids") or []
            )),
            "warning_family_or_null": claim.get("warning_family"),
            "source_layer": "jamaican_caribbean_tradition",
            "certainty_profile": _certainty_profile(doctrine),
            "safety_qualifier_ids": _safety_ids(str(claim.get("warning_family") or "")),
            "release_status": "released",
        })

    if compound and len(atomics) > 1:
        clauses.append({
            "narration_clause_id": "narration-warning-compound-1",
            "render_slot": "warning_compound_summary",
            "clause_type": "compound_summary",
            "rendered_text": "These warnings remain separate by owner, event chain, and warning family.",
            "output_span": {},
            "member_claim_ids": list(compound.get("member_claim_ids") or []),
            "owner_ids": [],
            "event_chain_ids": [],
            "warning_family_or_null": None,
            "source_layer": "documented_fact",
            "certainty_profile": {
                **_certainty_profile(doctrine),
                "warning_presence": "summary_only",
            },
            "safety_qualifier_ids": sorted({
                qualifier
                for claim in atomics
                for qualifier in _safety_ids(str(claim.get("warning_family") or ""))
            }),
            "release_status": "released",
        })

    for index, claim in enumerate(structural, start=1):
        if claim.get("claim_scope") == "structural_terminal":
            rendered = (
                "The same tooth returning firmly to the same place is the dream's final state. "
                "Its exact cultural consequence is unresolved, so no outcome is asserted."
            )
            clause_type = "terminal_structure"
        else:
            rendered = (
                "The attempt to put the tooth back is preserved as dream structure, but it is not "
                "treated as a completed restoration or terminal ending."
            )
            clause_type = "attempt_structure"
        clauses.append({
            "narration_clause_id": f"narration-structural-{index}",
            "render_slot": f"structural_{index}",
            "clause_type": clause_type,
            "rendered_text": rendered,
            "output_span": {},
            "member_claim_ids": [claim["claim_id"]],
            "owner_ids": [],
            "event_chain_ids": [],
            "warning_family_or_null": None,
            "source_layer": "documented_fact",
            "certainty_profile": {
                **_certainty_profile(doctrine),
                "warning_presence": "not_reactivated",
                "rule_match_confidence": "structural_only",
            },
            "safety_qualifier_ids": ["non_guarantee", "unresolved_consequence"],
            "release_status": "released",
        })

    if not clauses and doctrine.get("active_doctrine"):
        rendered = " ".join(
            part.strip() for part in [legacy_lead, *legacy_details] if str(part).strip()
        )
        if rendered:
            clauses.append({
                "narration_clause_id": "narration-approved-modifier-1",
                "render_slot": "approved_modifier",
                "clause_type": "approved_modifier",
                "rendered_text": rendered,
                "output_span": {},
                "member_claim_ids": [],
                "owner_ids": [],
                "event_chain_ids": [],
                "warning_family_or_null": None,
                "source_layer": "jts_spiritual_or_editorial_doctrine",
                "certainty_profile": _certainty_profile(doctrine),
                "safety_qualifier_ids": ["tradition_attribution", "non_guarantee"],
                "release_status": "released",
            })

    text_parts: List[str] = []
    offset = 0
    for clause in clauses:
        rendered = clause["rendered_text"]
        if text_parts:
            offset += 1
        start = offset
        offset += len(rendered)
        clause["output_span"] = {"start": start, "end": offset, "text": rendered}
        text_parts.append(rendered)
    narration_text = "\n".join(text_parts)
    payload = {
        "contract_version": NARRATION_CLAIM_CONTRACT_VERSION,
        "narration_text": narration_text,
        "narration_clauses": clauses,
    }
    payload["narration_integrity"] = validate_narration_claim_consumption(payload, graph)
    return payload


def validate_narration_claim_consumption(
    payload: Dict[str, Any], graph: Dict[str, Any]
) -> Dict[str, Any]:
    clauses = list(payload.get("narration_clauses") or [])
    narration_text = str(payload.get("narration_text") or "")
    manifest = list(graph.get("claim_manifest") or [])
    claims = {claim.get("claim_id"): claim for claim in manifest}
    atomics = {
        claim_id: claim for claim_id, claim in claims.items()
        if claim.get("claim_scope") == "atomic_warning" and claim.get("released") is True
    }
    reasons: List[str] = []

    for clause in clauses:
        required_fields = {
            "narration_clause_id", "render_slot", "clause_type", "rendered_text",
            "output_span", "member_claim_ids", "owner_ids", "event_chain_ids",
            "warning_family_or_null", "source_layer", "certainty_profile",
            "safety_qualifier_ids", "release_status",
        }
        if not required_fields.issubset(clause) or clause.get("release_status") != "released":
            reasons.append("GATED_BRANCH_NARRATED")
        if any(clause.get(key) for key in (
            "consumed_event_ids", "consumed_rule_ids", "consumed_span_ids", "source_event_ids"
        )):
            reasons.append("NARRATION_RAW_SOURCE_CONSUMPTION")
        for claim_id in clause.get("member_claim_ids") or []:
            claim = claims.get(claim_id)
            if claim is None:
                reasons.append("NARRATION_CLAIM_ORPHAN")
            elif claim.get("released") is not True or claim.get("release_status") != "released":
                reasons.append("GATED_BRANCH_NARRATED")

    member_clauses = [c for c in clauses if c.get("clause_type") == "warning_member"]
    counts = Counter(
        claim_id for clause in member_clauses for claim_id in clause.get("member_claim_ids") or []
    )
    for claim_id in atomics:
        if counts[claim_id] == 0:
            reasons.append("RELEASED_CLAIM_NOT_NARRATED")
        elif counts[claim_id] > 1:
            reasons.append("CLAIM_NARRATED_MULTIPLE_TIMES")

    for clause in member_clauses:
        member_ids = list(clause.get("member_claim_ids") or [])
        if not member_ids or any(member_id not in atomics for member_id in member_ids):
            continue
        member_claims = [atomics[member_id] for member_id in member_ids]
        expected_owners = {tuple(claim.get("owner_ids") or []) for claim in member_claims}
        if len(expected_owners) != 1 or tuple(clause.get("owner_ids") or []) not in expected_owners:
            reasons.append("NARRATION_OWNER_DRIFT")
        expected_chains = {
            chain_id for claim in member_claims for chain_id in claim.get("event_chain_ids") or []
        }
        if set(clause.get("event_chain_ids") or []) != expected_chains:
            reasons.append("NARRATION_CHAIN_DRIFT")
        expected_families = {claim.get("warning_family") for claim in member_claims}
        if len(expected_families) != 1 or clause.get("warning_family_or_null") not in expected_families:
            reasons.append("NARRATION_WARNING_FAMILY_DRIFT")
        required = {
            qualifier for claim in member_claims
            for qualifier in _safety_ids(str(claim.get("warning_family") or ""))
        }
        if not required.issubset(set(clause.get("safety_qualifier_ids") or [])):
            reasons.append("NARRATION_SAFETY_QUALIFIER_DROPPED")

    if any(c.get("clause_type") == "compound_summary" for c in clauses):
        if any(counts[claim_id] == 0 for claim_id in atomics):
            reasons.append("COMPOUND_MEMBER_CLAUSE_HIDDEN")
        compound_clauses = [c for c in clauses if c.get("clause_type") == "compound_summary"]
        required_compound = {
            qualifier for claim in atomics.values()
            for qualifier in _safety_ids(str(claim.get("warning_family") or ""))
        }
        if any(
            not required_compound.issubset(set(clause.get("safety_qualifier_ids") or []))
            for clause in compound_clauses
        ):
            reasons.append("NARRATION_SAFETY_QUALIFIER_DROPPED")
        if any(
            set(clause.get("member_claim_ids") or []) != set(atomics)
            for clause in compound_clauses
        ):
            reasons.append("COMPOUND_MEMBER_CLAUSE_HIDDEN")

    previous_end = 0
    for index, clause in enumerate(clauses):
        span = clause.get("output_span") or {}
        start, end = span.get("start"), span.get("end")
        expected_start = previous_end + (1 if index else 0)
        if (
            not isinstance(start, int) or not isinstance(end, int)
            or start != expected_start or end < start
            or narration_text[start:end] != clause.get("rendered_text")
            or span.get("text") != clause.get("rendered_text")
        ):
            reasons.append("NARRATION_OUTPUT_SPAN_MISMATCH")
        if isinstance(end, int):
            previous_end = end

    structural_types = {"terminal_structure", "attempt_structure"}
    for clause in clauses:
        if clause.get("clause_type") in structural_types:
            if any(member in atomics for member in clause.get("member_claim_ids") or []):
                reasons.append("STRUCTURAL_HISTORY_REACTIVATED")

    ordered = list(dict.fromkeys(reasons))
    return {"verified": not ordered, "reason_codes": ordered}
