import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.stream_utils import build_stream_events, extract_final_answer, truncate_text


class FakeMessage:
    def __init__(self, content):
        self.content = content


class TestStreamUtils(unittest.TestCase):
    def test_truncate_text_leaves_short_strings(self):
        self.assertEqual(truncate_text("short", limit=10), "short")

    def test_truncate_text_trims_long_strings(self):
        result = truncate_text("abcdefghijk", limit=5)
        self.assertEqual(result, "abcde...")

    def test_extract_final_answer_from_message_objects(self):
        output = {"messages": [FakeMessage("first"), FakeMessage("final")]}
        self.assertEqual(extract_final_answer(output), "final")

    def test_extract_final_answer_from_message_dicts(self):
        output = {"messages": [{"content": "hello"}, {"content": "world"}]}
        self.assertEqual(extract_final_answer(output), "world")

    def test_extract_final_answer_missing_messages(self):
        self.assertIsNone(extract_final_answer({"value": "none"}))

    def test_build_stream_events_tool_start(self):
        events, answer = build_stream_events({
            "event": "on_tool_start",
            "name": "SearchDocuments",
            "data": {"input": "graph database"},
        })

        self.assertIsNone(answer)
        self.assertEqual(events[0]["type"], "trace")
        self.assertEqual(events[0]["tool"], "SearchDocuments")
        self.assertEqual(events[1]["type"], "status")

    def test_build_stream_events_tool_end(self):
        events, answer = build_stream_events({
            "event": "on_tool_end",
            "name": "SearchDocuments",
            "data": {"output": "result"},
        })

        self.assertIsNone(answer)
        self.assertEqual(events[0]["type"], "trace")
        self.assertEqual(events[0]["observation"], "result")

    def test_build_stream_events_chain_end_extracts_answer(self):
        output = {"messages": [FakeMessage("first"), FakeMessage("final")]} 
        events, answer = build_stream_events({
            "event": "on_chain_end",
            "data": {"output": output},
        })

        self.assertEqual(answer, "final")
        self.assertEqual(events[0]["type"], "status")

    def test_build_stream_events_unknown_event(self):
        events, answer = build_stream_events({"event": "other"})
        self.assertEqual(events, [])
        self.assertIsNone(answer)


if __name__ == "__main__":
    unittest.main()
