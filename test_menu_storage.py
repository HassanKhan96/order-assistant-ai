"""Offline tests for structured storage and vector-document construction."""

from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from menu_normalization import normalize_api_menu
from menu_storage import build_embedding_documents, index_menu, menu_from_snapshot


def _menu():
    payload = SimpleNamespace(
        menu=[
            {
                "id": "pizza",
                "name": "Pizza",
                "description": "Tomato and cheese pizza.",
                "category_id": "mains",
                "availability": "AVAILABLE",
                "sizes": [
                    {"id": "small", "name": "Small", "price": 0, "availability": "AVAILABLE"},
                    {"id": "large", "name": "Large", "price": 4, "availability": "AVAILABLE"},
                ],
                "menu_item_variations": [],
                "pricing_and_availabilities": [
                    {"channel": "ONLINE_DELIVERY", "price": 10, "availability": "AVAILABLE"}
                ],
            },
            {
                "id": "water",
                "name": "Water",
                "description": None,
                "category_id": "drinks",
                "availability": "UNAVAILABLE",
                "sizes": [],
                "menu_item_variations": [],
                "pricing_and_availabilities": [
                    {"channel": "ONLINE_DELIVERY", "price": 2, "availability": "AVAILABLE"}
                ],
            },
        ],
        categories=[
            {"id": "mains", "name": "Mains"},
            {"id": "drinks", "name": "Drinks"},
        ],
        menu_endpoint="items-url",
        categories_endpoint="categories-url",
        fetched_at=datetime(2026, 10, 9, 12, tzinfo=timezone.utc),
    )
    return normalize_api_menu(
        payload,
        menu_id="store",
        restaurant_name="Restaurant",
        currency="GBP",
        channel="ONLINE_DELIVERY",
    )


class FakeProvider:
    model_id = "fake-embedding-model"
    dimensions = 3

    def __init__(self):
        self.texts = ()

    def embed_documents(self, texts):
        self.texts = tuple(texts)
        return [(0.1, 0.2, 0.3) for _ in texts]


class FakeStore:
    def __init__(self):
        self.saved = None
        self.index_dimensions = None

    def replace_embeddings(self, menu_id, model_id, documents):
        self.saved = (menu_id, model_id, tuple(documents))

    def ensure_vector_index(self, dimensions):
        self.index_dimensions = dimensions


class MenuStorageTests(unittest.TestCase):
    def test_canonical_snapshot_round_trip(self):
        menu = _menu()
        self.assertEqual(menu_from_snapshot(asdict(menu)), menu)

    def test_builds_one_document_per_purchasable_item_or_variant(self):
        documents = build_embedding_documents(_menu())
        self.assertEqual(
            [(item_id, variant_id) for item_id, variant_id, _, _ in documents],
            [("pizza", "small"), ("pizza", "large"), ("water", None)],
        )
        self.assertIn("Variant: Large", documents[1][2])
        self.assertIn("Price: GBP 14", documents[1][2])
        self.assertEqual(documents[2][3]["availability"], "unavailable")

    def test_indexes_documents_and_creates_dimension_specific_index(self):
        provider = FakeProvider()
        store = FakeStore()
        count = index_menu(_menu(), provider, store)
        self.assertEqual(count, 3)
        self.assertEqual(len(provider.texts), 3)
        self.assertEqual(store.saved[:2], ("store", provider.model_id))
        self.assertEqual(store.index_dimensions, 3)
        self.assertTrue(all(len(row.embedding) == 3 for row in store.saved[2]))

    def test_rejects_wrong_embedding_dimensions_before_writing(self):
        provider = FakeProvider()
        provider.embed_documents = lambda texts: [(0.1, 0.2) for _ in texts]
        store = FakeStore()
        with self.assertRaisesRegex(ValueError, "expected 3"):
            index_menu(_menu(), provider, store)
        self.assertIsNone(store.saved)


if __name__ == "__main__":
    unittest.main()
