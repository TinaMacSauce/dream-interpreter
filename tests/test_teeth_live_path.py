from pathlib import Path
import unittest


SERVICE_PATH = Path(__file__).resolve().parents[1] / "app" / "services" / "interpreter_service.py"


class TeethLivePathTests(unittest.TestCase):
    def test_teeth_facts_are_attached_before_narration(self):
        source = SERVICE_PATH.read_text(encoding="utf-8")

        attach_call = "doctrine_facts = attach_teeth_narration_facts("
        narration_call = "narration = build_narration_result("
        payload_call = '"doctrine_facts": doctrine_facts'

        self.assertIn(attach_call, source)
        self.assertIn(narration_call, source)
        self.assertIn(payload_call, source)

        attach_index = source.index(attach_call)
        narration_index = source.index(narration_call, attach_index)
        payload_index = source.index(payload_call, narration_index)

        self.assertLess(attach_index, narration_index)
        self.assertLess(narration_index, payload_index)

        integration_slice = source[attach_index:narration_index]
        self.assertIn("dream,", integration_slice)
        self.assertIn("doctrine_facts,", integration_slice)

        narration_slice = source[narration_index:payload_index]
        self.assertIn("doctrine_facts=doctrine_facts", narration_slice)


if __name__ == "__main__":
    unittest.main()
