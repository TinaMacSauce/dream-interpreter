"""
Dream Decoder AI
Ontology Service v1

Purpose:
Load ontology structures from Google Sheets.

This file does NOT perform interpretation.

It only loads and normalizes ontology data.
"""

from typing import Dict, List


class OntologyService:
    """
    Future ontology manager.
    """

    def __init__(self):
        self.nodes = {}
        self.behaviors = {}
        self.relations = {}
        self.states = {}

    def load_nodes(self):
        return self.nodes

    def load_behaviors(self):
        return self.behaviors

    def load_relations(self):
        return self.relations

    def load_states(self):
        return self.states
