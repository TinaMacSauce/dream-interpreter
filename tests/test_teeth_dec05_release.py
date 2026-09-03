import os
import unittest
from unittest.mock import patch

from app import create_app
from app.release_info import release_metadata
from app.teeth_context import extract_teeth_context
from app.teeth_doctrine import build_teeth_doctrine_context, build_teeth_narration_facts
from app.teeth_integration import attach_teeth_narration_facts


class TeethDecisionFiveReleaseTests(unittest.TestCase):
    def test_other_person_is_the_subject(self):
        facts = build_teeth_narration_facts("My sister's tooth fell out.")

        self.assertEqual("sister", facts["subject_scope"])
        self.assertIn("TEETH-FALLOUT-OTHER", facts["applied_rule_ids"])
        self.assertIn("warning concerns that person", " ".join(facts["details"]).lower())

    def test_self_pull_adds_participation_without_blame(self):
        facts = build_teeth_narration_facts("I pulled one of my teeth out with pain.")

        self.assertEqual("self", facts["removal_actor"])
        self.assertEqual("self_participation", facts["pull_modifier"])
        self.assertIn("TEETH-PULL-SELF", facts["applied_rule_ids"])
        self.assertIn("without assigning blame", " ".join(facts["details"]).lower())

    def test_external_pull_does_not_identify_culprit(self):
        facts = build_teeth_narration_facts("A stranger pulled my tooth out.")

        self.assertEqual("other", facts["removal_actor"])
        self.assertEqual("external_interference", facts["pull_modifier"])
        self.assertIn("TEETH-PULL-EXTERNAL", facts["applied_rule_ids"])
        self.assertIn("without identifying a real-world culprit", " ".join(facts["details"]).lower())

    def test_nearby_relationship_actor_does_not_steal_tooth_ownership(self):
        context = extract_teeth_context("My sister pulled my tooth out.")

        self.assertEqual("dreamer", context["owner"])
        self.assertEqual("other", context["removal_actor"])

    def test_rotten_tooth_is_sickness_warning(self):
        facts = build_teeth_narration_facts("My tooth was rotten but did not fall out.")

        self.assertEqual("rotten_sickness", facts["warning_kind"])
        self.assertIn("TEETH-STATE-ROTTEN", facts["applied_rule_ids"])
        self.assertIn("not a medical diagnosis", facts["lead"].lower())

    def test_gold_does_not_erase_broken_state(self):
        facts = build_teeth_narration_facts("My gold tooth was broken.")

        self.assertEqual("broken_sickness", facts["warning_kind"])
        self.assertIn("TEETH-STATE-BROKEN", facts["applied_rule_ids"])
        self.assertIn("TEETH-MOD-GOLD", facts["applied_rule_ids"])
        self.assertIn("does not erase", " ".join(facts["details"]).lower())

    def test_repetition_changes_salience_only(self):
        facts = build_teeth_narration_facts("Again and again, one of my teeth fell out.")

        self.assertEqual("increased", facts["salience_modifier"])
        self.assertIn("TEETH-MOD-REPETITION", facts["applied_rule_ids"])
        self.assertIn("salience only", " ".join(facts["details"]).lower())

    def test_same_tooth_returned_firm_seals_without_invented_outcome(self):
        dream = "My tooth fell out, but in the end the same tooth fitted firmly back into the same socket."
        facts = build_teeth_narration_facts(dream)

        self.assertTrue(facts["ending_precedence"])
        self.assertEqual("same_tooth_returned_firm", facts["terminal_ending"])
        self.assertEqual("unresolved", facts["outcome_resolution"])
        self.assertEqual(["TEETH-END-TERMINAL"], facts["applied_rule_ids"])
        self.assertIn("no outcome is asserted", facts["lead"].lower())
        self.assertIn("TEETH-END-RETURNED-SAME", facts["unresolved_rule_ids"])

    def test_non_human_teeth_fail_closed(self):
        facts = build_teeth_doctrine_context("The dog's teeth fell out.")

        self.assertEqual("non_human", facts["subject_class"])
        self.assertFalse(facts["active_doctrine"])
        self.assertIn("TEETH-NONHUMAN", facts["unresolved_rule_ids"])

    def test_live_bridge_carries_versions_and_rules(self):
        enriched = attach_teeth_narration_facts("My broken tooth did not fall out.", {})

        teeth = enriched["teeth_narration"]
        self.assertEqual("DEC-TEETH-2026-09-03-05", teeth["doctrine_version"])
        self.assertEqual("teeth-context-v2", teeth["context_version"])
        self.assertEqual("Dream Symbol Dictionary!DoctrineRegistry", teeth["doctrine_source"])
        self.assertIn("TEETH-STATE-BROKEN", teeth["applied_rule_ids"])

    def test_release_metadata_uses_render_commit(self):
        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "abc123"}, clear=False):
            metadata = release_metadata()

        self.assertEqual("abc123", metadata["build_commit"])
        self.assertEqual("DEC-TEETH-2026-09-03-05", metadata["teeth_doctrine_version"])
        self.assertEqual("teeth-context-v2", metadata["teeth_context_version"])

    def test_live_endpoint_exposes_release_contract(self):
        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "route-sha"}, clear=False):
            response = create_app().test_client().get("/live")

        self.assertEqual(200, response.status_code)
        release = response.get_json()["release"]
        self.assertEqual("route-sha", release["build_commit"])
        self.assertEqual("teeth-dec05-v1", release["release_id"])


if __name__ == "__main__":
    unittest.main()
