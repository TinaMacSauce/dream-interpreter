from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence


PROVENANCE_CONTRACT_VERSION = "claim-provenance-reachability/1.0"
PROVENANCE_DIGEST_ALGORITHM = "sha256"


def _inventory(values: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Mapping[str, Any]]:
    return {str(value[key]): value for value in values if value.get(key)}


def _registry_rules(registry: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(rule.get("rule_id")): rule
        for rule in registry.get("rules", {}).values()
        if rule.get("rule_id")
    }


def _cardinality(event: Mapping[str, Any]) -> int:
    return max(1, int(event.get("quantity_cardinality") or 1))


def _canonical_digest(graph: Mapping[str, Any]) -> str:
    payload = {
        "contract_version": graph.get("provenance_contract_version"),
        "events": graph.get("event_inventory", []),
        "rule_sets": graph.get("rule_sets", {}),
        "claims": graph.get("claim_manifest", []),
        "frontiers": graph.get("terminal_frontiers", []),
        "nodes": graph.get("provenance_nodes", []),
        "paths": graph.get("provenance_paths", []),
        "edges": graph.get("provenance_edges", []),
        "summary": graph.get("provenance_summary", {}),
        "rule_registry": graph.get("provenance_rule_registry", {}),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{PROVENANCE_DIGEST_ALGORITHM}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def finalize_claim_provenance(
    graph: MutableMapping[str, Any],
    result: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    events = _inventory(graph.get("event_inventory", []), "event_id")
    claims = _inventory(graph.get("claim_manifest", []), "claim_id")
    registry_rules = _registry_rules(registry)
    partitions: Dict[str, str] = {}
    rule_records: Dict[str, Mapping[str, Any]] = {}
    for partition, records in graph.get("rule_sets", {}).items():
        if partition == "contract_version":
            continue
        for record in records:
            rule_id = str(record.get("rule_id") or "")
            if rule_id:
                partitions[rule_id] = partition
                rule_records[rule_id] = record

    nodes: Dict[str, Dict[str, Any]] = {}
    paths: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def put_node(node_id: str, node_type: str, *, active: bool, owner: Any = None, chain: Any = None) -> None:
        nodes.setdefault(node_id, {
            "node_id": node_id,
            "node_type": node_type,
            "active": active,
            "owner_id_or_ambiguous": owner,
            "event_chain_id_or_null": chain,
        })

    def add_edge(path_id: str, index: int, source: str, target: str, owner: Any, chain: Any) -> None:
        edges.append({
            "edge_id": f"edge-{path_id}-{index}",
            "path_id": path_id,
            "from_node_id": source,
            "to_node_id": target,
            "owner_id_or_ambiguous": owner,
            "event_chain_id_or_null": chain,
        })

    released_claim_for_rule: Dict[str, str] = {}
    for claim in claims.values():
        if claim.get("released"):
            for rule_id in claim.get("consumed_rule_ids", []):
                released_claim_for_rule.setdefault(str(rule_id), str(claim["claim_id"]))

    path_index = 0
    for rule_id, record in sorted(rule_records.items()):
        registry_rule = registry_rules.get(rule_id, {})
        partition = partitions.get(rule_id, "")
        verified_active = bool(registry_rule.get("status") == "APPROVED" and registry_rule.get("active") is True)
        released = partition in {"public_applied", "warning_active", "structural"} and verified_active
        claim_id = released_claim_for_rule.get(rule_id)
        if released and not claim_id:
            released = False
        for event_id in record.get("source_event_ids", []):
            event = events.get(str(event_id))
            if not event:
                continue
            units = _cardinality(event) if event.get("event_type") == "tooth_loss" else 1
            for unit in range(1, units + 1):
                path_index += 1
                path_id = f"path-{path_index:03d}"
                owner = event.get("owner_id_or_ambiguous")
                chain = event.get("event_chain_id_or_null")
                candidate_id = f"candidate-{rule_id.lower()}-{event_id}-{unit}"
                span_id = str(event.get("source_span", {}).get("span_id") or "")
                disposition = "released" if released else (
                    "historical" if partition == "matched_historical" else "withheld"
                )
                gate = ""
                if not event.get("doctrine_eligible"):
                    disposition = "gated"
                    gate = "INELIGIBLE_EVENT_GATE"
                elif not verified_active:
                    disposition = "withheld"
                    gate = "UNVERIFIED_RULE_DEPENDENCY"
                path = {
                    "path_id": path_id,
                    "path_type": "governed_claim" if disposition == "released" else "withheld",
                    "source_span_id": span_id,
                    "event_id": event_id,
                    "candidate_id": candidate_id,
                    "rule_id": rule_id,
                    "rule_partition": partition,
                    "claim_id_or_null": claim_id if disposition == "released" else None,
                    "gate_or_dependency": gate,
                    "owner_id_or_ambiguous": owner,
                    "event_chain_id_or_null": chain,
                    "cardinality_unit": unit,
                    "disposition": disposition,
                    "complete": bool(span_id and event_id and rule_id and (claim_id or disposition != "released")),
                    "warning_path": bool(result.get("active_warning") and partition in {"public_applied", "warning_active"}),
                    "modifier": event.get("event_type") in {"pain_modifier", "tooth_blood_modifier"}
                        or rule_id.startswith("TEETH-MOD-") or rule_id.startswith("TEETH-PULL-"),
                    "terminal": rule_id == "TEETH-END-TERMINAL",
                }
                paths.append(path)
                span_node = f"span:{span_id}"
                event_node = f"event:{event_id}"
                candidate_node = f"candidate:{candidate_id}"
                rule_node = f"rule:{rule_id}:{event_id}:{unit}"
                put_node(span_node, "source_span", active=True, owner=owner, chain=chain)
                put_node(event_node, "event", active=True, owner=owner, chain=chain)
                put_node(candidate_node, "candidate", active=True, owner=owner, chain=chain)
                put_node(rule_node, "verified_active_rule", active=released, owner=owner, chain=chain)
                add_edge(path_id, 1, span_node, event_node, owner, chain)
                add_edge(path_id, 2, event_node, candidate_node, owner, chain)
                add_edge(path_id, 3, candidate_node, rule_node, owner, chain)
                if disposition == "released" and claim_id:
                    claim_node = f"claim:{claim_id}:{event_id}:{unit}"
                    put_node(claim_node, "claim", active=True, owner=owner, chain=chain)
                    add_edge(path_id, 4, rule_node, claim_node, owner, chain)
                else:
                    gate_node = f"gate:{path_id}"
                    put_node(gate_node, "gate_or_dependency", active=False, owner=owner, chain=chain)
                    add_edge(path_id, 4, rule_node, gate_node, owner, chain)

    # Every factual event receives a literal or gated path, even when no doctrine applies.
    covered_events = {str(path["event_id"]) for path in paths}
    for event_id, event in sorted(events.items()):
        if event_id in covered_events:
            continue
        path_index += 1
        path_id = f"path-{path_index:03d}"
        owner = event.get("owner_id_or_ambiguous")
        chain = event.get("event_chain_id_or_null")
        span_id = str(event.get("source_span", {}).get("span_id") or "")
        eligible = bool(event.get("doctrine_eligible"))
        structural_claim = next((
            str(claim["claim_id"]) for claim in claims.values()
            if event_id in claim.get("consumed_event_ids", []) and claim.get("released")
        ), None)
        disposition = "released" if eligible and structural_claim else ("literal" if eligible else "gated")
        path = {
            "path_id": path_id,
            "path_type": "literal_fact" if eligible else "withheld",
            "source_span_id": span_id,
            "event_id": event_id,
            "candidate_id": None,
            "rule_id": None,
            "rule_partition": "structural",
            "claim_id_or_null": structural_claim,
            "gate_or_dependency": "" if eligible else "INELIGIBLE_EVENT_GATE",
            "owner_id_or_ambiguous": owner,
            "event_chain_id_or_null": chain,
            "cardinality_unit": 1,
            "disposition": disposition,
            "complete": bool(span_id and event_id),
            "warning_path": False,
            "modifier": False,
            "terminal": False,
        }
        paths.append(path)
        span_node = f"span:{span_id}"
        event_node = f"event:{event_id}"
        sink_node = f"claim:{structural_claim}" if structural_claim else f"gate:{path_id}"
        put_node(span_node, "source_span", active=True, owner=owner, chain=chain)
        put_node(event_node, "event", active=True, owner=owner, chain=chain)
        put_node(sink_node, "literal_fact_claim" if structural_claim else "gate_or_dependency", active=bool(structural_claim), owner=owner, chain=chain)
        add_edge(path_id, 1, span_node, event_node, owner, chain)
        add_edge(path_id, 2, event_node, sink_node, owner, chain)

    eligible_losses = [
        event for event in events.values()
        if event.get("event_type") == "tooth_loss" and event.get("doctrine_eligible")
    ]
    graph["provenance_contract_version"] = PROVENANCE_CONTRACT_VERSION
    graph["provenance_nodes"] = sorted(nodes.values(), key=lambda item: item["node_id"])
    graph["provenance_paths"] = paths
    graph["provenance_edges"] = edges
    graph["provenance_summary"] = {
        "loss_path_count": sum(_cardinality(event) for event in eligible_losses),
        "owner_path_count": len({event.get("owner_id_or_ambiguous") for event in eligible_losses}),
        "governing_chain_count": len({event.get("event_chain_id_or_null") for event in eligible_losses}),
        "released_path_count": sum(path["disposition"] == "released" for path in paths),
        "withheld_path_count": sum(path["disposition"] in {"withheld", "gated", "historical"} for path in paths),
        "modifier_path_count": sum(bool(path["modifier"]) for path in paths),
        "cross_owner_path_count": 0,
        "aggregate_contributing_event_count": len(eligible_losses),
    }
    graph["provenance_digest"] = _canonical_digest(graph)
    graph["provenance_integrity"] = validate_claim_provenance(graph, registry)


def validate_claim_provenance(
    graph: Mapping[str, Any], registry: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    events = _inventory(graph.get("event_inventory", []), "event_id")
    claims = _inventory(graph.get("claim_manifest", []), "claim_id")
    nodes = _inventory(graph.get("provenance_nodes", []), "node_id")
    paths = _inventory(graph.get("provenance_paths", []), "path_id")
    edges = list(graph.get("provenance_edges", []))
    if registry:
        registry_rules = _registry_rules(registry)
    else:
        registry_rules = {
            str(rule_id): rule
            for rule_id, rule in (graph.get("provenance_rule_registry") or {}).items()
        }

    released_paths = [path for path in paths.values() if path.get("disposition") == "released"]
    for claim in claims.values():
        if claim.get("released") and claim.get("consumed_rule_ids") and not any(
            path.get("claim_id_or_null") == claim.get("claim_id") for path in released_paths
        ):
            reasons.append("CLAIM_PATH_MISSING")
    if any(path.get("warning_path") and not path.get("complete") for path in released_paths):
        reasons.append("WARNING_PATH_INCOMPLETE")
    for path in released_paths:
        rule_id = path.get("rule_id")
        if rule_id:
            rule = registry_rules.get(str(rule_id))
            if not rule or rule.get("status") != "APPROVED" or rule.get("active") is not True:
                reasons.append("UNVERIFIED_RULE_IN_PATH")
        event = events.get(str(path.get("event_id")))
        if not event:
            reasons.append("CLAIM_PATH_MISSING")
            continue
        if path.get("owner_id_or_ambiguous") != event.get("owner_id_or_ambiguous"):
            reasons.append("CROSS_OWNER_PATH")
        if path.get("event_chain_id_or_null") != event.get("event_chain_id_or_null"):
            reasons.append("CROSS_CHAIN_PATH")
        if not event.get("doctrine_eligible"):
            reasons.append("INELIGIBLE_EVENT_REACHES_CLAIM")

    warning_paths = [path for path in released_paths if path.get("warning_path")]
    if warning_paths and all(path.get("modifier") for path in warning_paths):
        reasons.append("MODIFIER_PATH_REPLACES_BASE")

    terminal_paths = [path for path in released_paths if path.get("terminal")]
    frontiers = {item.get("event_chain_id"): item for item in graph.get("terminal_frontiers", [])}
    if any(
        frontiers.get(path.get("event_chain_id_or_null"), {}).get("terminal_event_id") != path.get("event_id")
        for path in terminal_paths
    ):
        reasons.append("TERMINAL_PATH_STALE")

    referenced_nodes = {
        value for edge in edges for value in (edge.get("from_node_id"), edge.get("to_node_id")) if value
    }
    if any(node.get("active") and node_id not in referenced_nodes for node_id, node in nodes.items()):
        reasons.append("ORPHAN_ACTIVE_NODE")

    adjacency: Dict[str, List[str]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge.get("from_node_id")), []).append(str(edge.get("to_node_id")))
    visiting: set[str] = set()
    visited: set[str] = set()

    def cyclic(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(cyclic(target) for target in adjacency.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(cyclic(node) for node in list(adjacency)):
        reasons.append("PROVENANCE_CYCLE")

    expected_cardinality = sum(
        _cardinality(event) for event in events.values()
        if event.get("event_type") == "tooth_loss" and event.get("doctrine_eligible")
    )
    if graph.get("provenance_summary", {}).get("loss_path_count") != expected_cardinality:
        reasons.append("PATH_CARDINALITY_MISMATCH")
    if graph.get("provenance_digest") != _canonical_digest(graph):
        reasons.append("GRAPH_DIGEST_MISMATCH")

    unique = list(dict.fromkeys(reasons))
    return {
        "verified": not unique,
        "reason_codes": unique,
        "path_count": len(paths),
        "edge_count": len(edges),
        "digest_algorithm": PROVENANCE_DIGEST_ALGORITHM,
    }
