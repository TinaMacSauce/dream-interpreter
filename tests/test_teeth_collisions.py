import unittest

from app.rules import detect_rule_hits


TEETH_FALLING_ROW = {
    "behavior_name": "Teeth Falling Out",
    "keywords": "teeth fell out, teeth falling out, fall out, falling out, fell out, came out, coming out",
    "active": "yes",
    "priority": "10",
}

GENERIC_FALLING_ROW = {
    "behavior_name": "Falling",
    "keywords": "falling, fell, fall",
    "active": "yes",
    "priority": "10",
}


class TeethCollisionRegressionTests(unittest.TestCase):
    def _names(self, dream: str):
        hits = detect_rule_hits(
            dream=dream,
            rows=[TEETH_FALLING_ROW, GENERIC_FALLING_ROW],
            kind="behavior",
            max_hits=5,
        )
        return {str(hit.get("name", "")).lower() for hit in hits}

    def test_teeth_fallout_suppresses_generic_falling(self):
        names = self._names("My teeth fell out.")
        self.assertIn("teeth falling out", names)
        self.assertNotIn("falling", names)

    def test_separate_body_fall_keeps_generic_falling(self):
        names = self._names("My teeth fell out and then I fell down the stairs.")
        self.assertIn("teeth falling out", names)
        self.assertIn("falling", names)

    def test_falling_without_teeth_stays_generic(self):
        names = self._names("I was falling down a hill.")
        self.assertNotIn("teeth falling out", names)
        self.assertIn("falling", names)

    def test_teeth_falling_phrase_does_not_create_duplicate_behavior(self):
        names = self._names("My teeth were falling out one by one.")
        self.assertEqual({"teeth falling out"}, names)


if __name__ == "__main__":
    unittest.main()
