import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embedding_config import resolve_embedding_model, resolve_embedding_dimensions


class TestEmbeddingConfig(unittest.TestCase):
    def test_default_model_when_unset(self):
        self.assertEqual(resolve_embedding_model({}), "models/gemini-embedding-001")

    def test_model_prefix_added_when_missing(self):
        env = {"GOOGLE_EMBEDDING_MODEL": "text-embedding-004"}
        self.assertEqual(resolve_embedding_model(env), "models/text-embedding-004")

    def test_model_prefix_preserved(self):
        env = {"GOOGLE_EMBEDDING_MODEL": "models/text-embedding-004"}
        self.assertEqual(resolve_embedding_model(env), "models/text-embedding-004")

    def test_blank_model_falls_back(self):
        env = {"GOOGLE_EMBEDDING_MODEL": "   "}
        self.assertEqual(resolve_embedding_model(env), "models/gemini-embedding-001")

    def test_default_dimensions_when_unset(self):
        self.assertIsNone(resolve_embedding_dimensions({}))

    def test_dimensions_parse_int(self):
        env = {"GOOGLE_EMBEDDING_DIMENSIONS": "768"}
        self.assertEqual(resolve_embedding_dimensions(env), 768)

    def test_dimensions_invalid_returns_none(self):
        env = {"GOOGLE_EMBEDDING_DIMENSIONS": "abc"}
        self.assertIsNone(resolve_embedding_dimensions(env))

    def test_dimensions_negative_returns_none(self):
        env = {"GOOGLE_EMBEDDING_DIMENSIONS": "-1"}
        self.assertIsNone(resolve_embedding_dimensions(env))


if __name__ == "__main__":
    unittest.main()
