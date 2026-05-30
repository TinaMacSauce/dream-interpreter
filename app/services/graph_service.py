"""
Dream Decoder AI
Graph Service v1

Shadow Mode Only

This service evaluates symbolic relationships
without changing production interpretations.
"""

from app.services.ontology_service import OntologyService


class GraphService:

    def __init__(self):
        self.ontology = OntologyService()

    def evaluate(
        self,
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
    ):
        """
        Future graph engine entry point.
        """

        return {
            "graph_enabled": self.ontology.graph_enabled(),
            "graph_mode": self.ontology.graph_mode(),
            "relations_detected": [],
            "weight_adjustments": [],
            "confidence_adjustments": [],
        }
