"""
Dream Decoder AI
Ontology Service v1.1

Purpose:
Load and normalize ontology structures from Google Sheets.

This file does NOT perform dream interpretation and does NOT change
production interpretation output. It only prepares ontology data for the
future graph/shadow-mode layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.sheets import get_or_create_worksheet, worksheet_to_rows


# Sheet names are kept here for now so this service can be introduced
# without changing Config or the production interpreter flow.
SHEET_GRAPH_CONFIG = "GraphConfig"
SHEET_ONTOLOGY_MAPPINGS = "OntologyMappings"
SHEET_NODE_REGISTRY = "NodeRegistry"
SHEET_BEHAVIOR_BINDINGS = "BehaviorBindingRules"
SHEET_RELATION_RULES = "RelationRules"
SHEET_STATE_MODIFIERS = "StateModifierRules"
SHEET_BASE_SYMBOLS = "BaseSymbols"


TRUE_VALUES = {"true", "yes", "1", "on", "active", "y"}
FALSE_VALUES = {"false", "no", "0", "off", "inactive", "n"}


class OntologyService:
    """
    Loads graph/ontology support tables from Google Sheets.

    This service is intentionally read-only. It should not write to Sheets,
    should not modify interpreter output, and should be safe while GraphConfig
    mode remains off.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.behaviors: Dict[str, Dict] = {}
        self.relations: Dict[str, Dict] = {}
        self.states: Dict[str, Dict] = {}
        self.mappings: Dict[str, Dict] = {}
        self.graph_config: Dict[str, str] = {}

    # ============================================================
    # Normalization helpers
    # ============================================================

    @staticmethod
    def clean(value) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @classmethod
    def key(cls, value) -> str:
        return cls.clean(value).lower().replace(" ", "_")

    @classmethod
    def as_bool(cls, value, default: bool = False) -> bool:
        cleaned = cls.clean(value).lower()
        if cleaned in TRUE_VALUES:
            return True
        if cleaned in FALSE_VALUES:
            return False
        return default

    @classmethod
    def as_int(cls, value, default: int = 0) -> int:
        try:
            return int(float(cls.clean(value)))
        except Exception:
            return default

    @classmethod
    def as_float(cls, value, default: float = 0.0) -> float:
        try:
            return float(cls.clean(value))
        except Exception:
            return default

    @staticmethod
    def _read_sheet(sheet_name: str) -> List[Dict[str, str]]:
        ws = get_or_create_worksheet(sheet_name)
        _, rows = worksheet_to_rows(ws)
        return rows

    # ============================================================
    # Graph config
    # ============================================================

    def load_graph_config(self, force: bool = False) -> Dict[str, str]:
        if self.graph_config and not force:
            return self.graph_config

        rows = self._read_sheet(SHEET_GRAPH_CONFIG)
        config: Dict[str, str] = {}

        for row in rows:
            config_key = self.key(row.get("key", ""))
            value = self.clean(row.get("value", ""))

            if not config_key:
                continue

            config[config_key] = value

        if "mode" not in config:
            config["mode"] = "off"

        self.graph_config = config
        return config

    def graph_mode(self) -> str:
        return self.clean(self.load_graph_config().get("mode", "off")).lower() or "off"

    def graph_enabled(self) -> bool:
        return self.graph_mode() in {"shadow", "active"}

    def relation_enabled(self, relation_type: str) -> bool:
        config = self.load_graph_config()
        flag_key = f"enable_{self.clean(relation_type).upper()}".lower()
        return self.as_bool(config.get(flag_key, "false"), default=False)

    # ============================================================
    # Ontology mappings
    # ============================================================

    def load_mappings(self, force: bool = False) -> Dict[str, Dict]:
        if self.mappings and not force:
            return self.mappings

        rows = self._read_sheet(SHEET_ONTOLOGY_MAPPINGS)
        mappings: Dict[str, Dict] = {}

        for row in rows:
            old_category = self.key(row.get("old_category", ""))
            new_category = self.key(row.get("new_category", ""))
            ontology_type = self.key(row.get("ontology_type", ""))

            if not old_category:
                continue

            mappings[old_category] = {
                "old_category": old_category,
                "new_category": new_category,
                "ontology_type": ontology_type,
                "notes": self.clean(row.get("notes", "")),
            }

        self.mappings = mappings
        return mappings

    def map_category(self, category: str) -> Dict[str, str]:
        cleaned = self.key(category)
        mappings = self.load_mappings()

        return mappings.get(
            cleaned,
            {
                "old_category": cleaned,
                "new_category": "unknown",
                "ontology_type": "unknown",
                "notes": "Unmapped category",
            },
        )

    # ============================================================
    # Graph registry loaders
    # ============================================================

    def load_nodes(self, force: bool = False) -> Dict[str, Dict]:
        if self.nodes and not force:
            return self.nodes

        rows = self._read_sheet(SHEET_NODE_REGISTRY)
        nodes: Dict[str, Dict] = {}

        for row in rows:
            symbol_key = self.key(row.get("symbol_key", ""))
            if not symbol_key:
                continue

            nodes[symbol_key] = {
                "symbol_key": symbol_key,
                "category": self.key(row.get("category", "")),
                "tier": self.key(row.get("tier", "node")) or "node",
                "base_weight": self.as_int(row.get("base_weight", 0), default=0),
                "default_valence": self.key(row.get("default_valence", "neutral")),
                "dual_role_flag": self.as_bool(row.get("dual_role_flag", False)),
                "graph_ready": self.as_bool(row.get("graph_ready", True), default=True),
                "notes": self.clean(row.get("notes", "")),
            }

        self.nodes = nodes
        return nodes

    def load_behaviors(self, force: bool = False) -> Dict[str, Dict]:
        if self.behaviors and not force:
            return self.behaviors

        rows = self._read_sheet(SHEET_BEHAVIOR_BINDINGS)
        behaviors: Dict[str, Dict] = {}

        for row in rows:
            behavior_key = self.key(row.get("behavior_key", ""))
            if not behavior_key:
                continue

            behaviors[behavior_key] = {
                "behavior_key": behavior_key,
                "as_modifier": self.key(row.get("as_modifier", "")),
                "as_relation_type": self.clean(row.get("as_relation_type", "")).upper(),
                "default_role": self.key(row.get("default_role", "modifier")) or "modifier",
                "notes": self.clean(row.get("notes", "")),
            }

        self.behaviors = behaviors
        return behaviors

    def load_relations(self, force: bool = False) -> Dict[str, Dict]:
        if self.relations and not force:
            return self.relations

        rows = self._read_sheet(SHEET_RELATION_RULES)
        relations: Dict[str, Dict] = {}

        for row in rows:
            rule_id = self.clean(row.get("rule_id", ""))
            if not rule_id:
                continue

            relations[rule_id] = {
                "rule_id": rule_id,
                "relation_type": self.clean(row.get("relation_type", "")).upper(),
                "source_category": self.key(row.get("source_category", "")),
                "target_category": self.key(row.get("target_category", "")),
                "trigger_behaviors": [
                    self.key(item)
                    for item in self.clean(row.get("trigger_behaviors", "")).split(",")
                    if self.key(item)
                ],
                "operator": self.clean(row.get("operator", "")).upper(),
                "magnitude": self.as_float(row.get("magnitude", 0.0), default=0.0),
                "gate_condition": self.key(row.get("gate_condition", "")),
                "valence_effect": self.key(row.get("valence_effect", "neutral")),
                "confidence_effect": self.as_float(row.get("confidence_effect", 0.0), default=0.0),
                "precedence": self.as_int(row.get("precedence", 100), default=100),
                "enabled": self.as_bool(row.get("enabled", False), default=False),
                "notes": self.clean(row.get("notes", "")),
            }

        self.relations = relations
        return relations

    def load_states(self, force: bool = False) -> Dict[str, Dict]:
        if self.states and not force:
            return self.states

        rows = self._read_sheet(SHEET_STATE_MODIFIERS)
        states: Dict[str, Dict] = {}

        for row in rows:
            state_key = self.key(row.get("state_key", ""))
            applies_to = self.key(row.get("applies_to_category", "*")) or "*"

            if not state_key:
                continue

            registry_key = f"{state_key}:{applies_to}"
            states[registry_key] = {
                "state_key": state_key,
                "applies_to_category": applies_to,
                "weight_delta": self.as_int(row.get("weight_delta", 0), default=0),
                "weight_multiplier": self.as_float(row.get("weight_multiplier", 1.0), default=1.0),
                "valence_effect": self.key(row.get("valence_effect", "neutral")),
                "notes": self.clean(row.get("notes", "")),
            }

        self.states = states
        return states

    # ============================================================
    # Diagnostics
    # ============================================================

    def load_all(self, force: bool = False) -> Dict[str, Dict]:
        return {
            "graph_config": self.load_graph_config(force=force),
            "mappings": self.load_mappings(force=force),
            "nodes": self.load_nodes(force=force),
            "behaviors": self.load_behaviors(force=force),
            "relations": self.load_relations(force=force),
            "states": self.load_states(force=force),
        }

    def summary(self) -> Dict[str, object]:
        data = self.load_all()
        return {
            "graph_mode": self.graph_mode(),
            "graph_enabled": self.graph_enabled(),
            "mapping_count": len(data["mappings"]),
            "node_count": len(data["nodes"]),
            "behavior_count": len(data["behaviors"]),
            "relation_count": len(data["relations"]),
            "state_count": len(data["states"]),
        }
