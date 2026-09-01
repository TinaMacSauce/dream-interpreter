import unittest

from app.rules import detect_rule_hits


BROKEN_ROW = {
    "state_name": "Broken",
    "keywords": "broken, cracked, chipped",
    "priority": "10",
    "active": "yes",
}

LOOSE_ROW = {
    "state_name": "Loose",
    "keywords": "loose, wobbly",
    "priority": "10",
    "active": "yes",
}

BLEEDING_ROW = {
    "state_name": "Bleeding",
    "keywords": "bleeding, blood",
    "priority": "10",
    "active": "yes",
}

TEETH_FALLING_ROW = {
    "behavior_name": "Teeth Falling Out",
    "keywords": "teeth falling out, teeth fell out, tooth fell out, fall out",
    "priority": "20",
    "active": "yes",
}


class TeethStateRegressionTests(unittest.TestCase):
    def _state_names(self, dream: str):
        hits = detect_rule_hits(
            dream=dream,
            rows=[BROKEN_ROW, LOOSE_ROW, BLEEDING_ROW],
            kind="state",
            max_hits=5,
        )
        return [hit["name"] for hit in hits]

    def _behavior_names(self, dream: str):
        hits = detect_rule_hits(
            dream=dream,
            rows=[TEETH_FALLING_ROW],
            kind="behavior",
            max_hits=5,
        )
        return [hit["name"] for hit in hits]

    def test_broken_tooth_state_is_detected(self):
        self.assertIn("Broken", self._state_names("One of my teeth was broken."))

    def test_cracked_tooth_maps_to_broken_state(self):
        self.assertIn("Broken", self._state_names("I saw a cracked tooth in my mouth."))

    def test_loose_teeth_state_is_detected(self):
        self.assertIn("Loose", self._state_names("Two of my teeth were loose."))

    def test_wobbly_tooth_maps_to_loose_state(self):
        self.assertIn("Loose", self._state_names("My front tooth felt wobbly."))

    def test_bleeding_gums_state_is_detected(self):
        self.assertIn("Bleeding", self._state_names("My gums were bleeding around my teeth."))

    def test_blood_near_teeth_maps_to_bleeding_state(self):
        self.assertIn("Bleeding", self._state_names("There was blood around my teeth and gums."))

    def test_multiple_teeth_states_can_coexist(self):
        names = self._state_names("My tooth was cracked, another tooth was loose, and my gums were bleeding.")
        self.assertIn("Broken", names)
        self.assertIn("Loose", names)
        self.assertIn("Bleeding", names)

    def test_state_does_not_replace_teeth_fallout_behavior(self):
        dream = "My loose tooth fell out and my gums were bleeding."
        self.assertIn("Loose", self._state_names(dream))
        self.assertIn("Bleeding", self._state_names(dream))
        self.assertIn("Teeth Falling Out", self._behavior_names(dream))


if __name__ == "__main__":
    unittest.main()
