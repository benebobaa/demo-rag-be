import json
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.sse import format_sse_event


class TestSseFormatting(unittest.TestCase):
    def test_format_sse_event_emits_event_and_json_data(self):
        payload = format_sse_event(
            "status",
            {"seq": 1, "ts": "2026-01-28T00:00:00Z", "stage": "planning"},
        )

        self.assertTrue(payload.startswith("event: status\n"))
        lines = [line for line in payload.split("\n") if line.startswith("data: ")]
        self.assertEqual(len(lines), 1)

        data = json.loads(lines[0].replace("data: ", ""))
        self.assertEqual(data["seq"], 1)
        self.assertEqual(data["stage"], "planning")


if __name__ == "__main__":
    unittest.main()
