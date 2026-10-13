"""Opt-in pgvector tests; each test uses an isolated schema and rolls it back.

Run with TEST_DATABASE_URL set to a local PostgreSQL database with pgvector
available. No model service is called and no test records are committed.
"""

from dataclasses import replace
from decimal import Decimal
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from uuid import uuid4

from enrichment_cache import EnrichmentCache
from menu_normalization import Availability
from menu_retrieval import RetrievalFilters, retrieve_candidates
from menu_storage import EmbeddedDocument, PostgresMenuStore, index_menu
from test_menu_retrieval import QueryProvider, with_required_choices
from test_menu_storage import FakeProvider, _menu
from import_enrichment import import_saved_enrichment, prepare_import
from test_import_enrichment import saved_fixture


@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL'), 'TEST_DATABASE_URL is not set')
class PostgresIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = PostgresMenuStore.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.store.connection.close)
        self.enterContext(self.store.connection.transaction(force_rollback=True))
        # Only locally generated hexadecimal characters are interpolated.
        schema = 'retrieval_test_' + uuid4().hex
        self.store.connection.execute(f'CREATE SCHEMA {schema}')
        self.store.connection.execute(f'SET LOCAL search_path TO {schema}, public')
        self.store.initialize()

    def test_persistence_cache_index_and_grounded_retrieval(self):
        menu = with_required_choices(_menu())
        self.store.replace_menu(menu)
        self.assertEqual(self.store.load_menu(menu.id), menu)
        cache = EnrichmentCache(menu.id)
        cache.descriptions['pizza'] = {'text': 'test cached description'}
        self.store.save_enrichment_cache(cache)
        self.assertEqual(self.store.load_enrichment_cache(menu.id).descriptions, cache.descriptions)
        self.assertEqual(index_menu(menu, FakeProvider(), self.store), 3)
        hits = self.store.search(menu_id=menu.id, model_id=FakeProvider.model_id,
            query_embedding=(0.1, 0.2, 0.3), max_price=Decimal('10'), category_ids=('mains',))
        self.assertEqual([(hit.item_id, hit.variant_id) for hit in hits], [('pizza', 'small')])
        candidates = retrieve_candidates('pizza', menu_id=menu.id, store=self.store,
            provider=QueryProvider(), filters=RetrievalFilters(max_price=Decimal('12')))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].minimum_price, Decimal('12'))
        self.assertIsNotNone(candidates[0].semantic_score)
        self.assertEqual(self.store.search(menu_id='other-menu', model_id=FakeProvider.model_id,
            query_embedding=(0.1, 0.2, 0.3)), ())
        self.assertEqual(self.store.search(menu_id=menu.id, model_id='other-model',
            query_embedding=(0.1, 0.2, 0.3)), ())
        self.assertEqual(self.store.search(menu_id=menu.id, model_id=FakeProvider.model_id,
            query_embedding=(0.1, 0.2, 0.3), dietary_attributes=('vegan',)), ())

    def test_saved_import_is_repeatable_and_keeps_vectors_and_unrelated_cache(self):
        with TemporaryDirectory() as directory:
            prepared = prepare_import(*saved_fixture(directory))
        cache = EnrichmentCache(prepared.menu.id)
        cache.descriptions['unrelated'] = {'text': 'keep this historical cache row'}
        self.store.save_enrichment_cache(cache)
        report = import_saved_enrichment(prepared, self.store)
        self.assertEqual(report['applied_descriptions'], 1)
        self.assertEqual(self.store.load_menu(prepared.menu.id), prepared.menu)
        self.assertEqual(self.store.load_dietary_cache(prepared.menu.id), prepared.dietary_results)
        self.assertIn('unrelated', self.store.load_enrichment_cache(prepared.menu.id).descriptions)
        index_menu(prepared.menu, FakeProvider(), self.store)
        import_saved_enrichment(prepared, self.store)
        self.assertEqual(len(self.store.existing_document_hashes(prepared.menu.id, FakeProvider.model_id)), 1)
        self.assertEqual(self.store.load_menu(prepared.menu.id).items[0].dietary_attributes, ())

    def test_import_failure_rolls_back_menu_and_all_cache_writes(self):
        with TemporaryDirectory() as directory:
            prepared = prepare_import(*saved_fixture(directory))
        with patch.object(self.store, 'save_dietary_cache', side_effect=RuntimeError('write failure')):
            with self.assertRaisesRegex(RuntimeError, 'write failure'):
                import_saved_enrichment(prepared, self.store)
        self.assertIsNone(self.store.load_menu(prepared.menu.id))
        self.assertEqual(self.store.load_enrichment_cache(prepared.menu.id).descriptions, {})

    def test_unavailable_parent_cannot_be_returned_via_available_variant(self):
        menu = _menu()
        menu = replace(menu, items=(replace(menu.items[0], availability=Availability('unavailable')),))
        self.store.replace_menu(menu)
        index_menu(menu, FakeProvider(), self.store)
        self.assertEqual(self.store.search(menu_id=menu.id, model_id=FakeProvider.model_id,
            query_embedding=(0.1, 0.2, 0.3)), ())

    def test_failed_replacement_rolls_back_existing_vectors(self):
        menu = _menu()
        self.store.replace_menu(menu)
        index_menu(menu, FakeProvider(), self.store)
        invalid = EmbeddedDocument(item_id='missing', variant_id=None, text='invalid',
                                   metadata={}, embedding=(1, 0, 0))
        import psycopg
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.store.replace_embeddings(menu.id, FakeProvider.model_id, (invalid,))
        self.assertEqual(len(self.store.existing_document_hashes(menu.id, FakeProvider.model_id)), 3)

    def test_notebook_cells_run_against_persisted_menu_with_fake_embeddings(self):
        menu = _menu()
        item = menu.items[0]
        item = replace(item, variants=tuple(
            replace(variant, price=Decimal('2')) for variant in item.variants))
        menu = replace(menu, items=(item,))
        self.store.replace_menu(menu)
        provider = FakeProvider()
        provider.embed_query = lambda text: (0.1, 0.2, 0.3)
        provider.embedding_model_id = provider.model_id
        notebook = json.loads((Path(__file__).parent / '04_embeddings_retrieval.ipynb').read_text())
        namespace = {}
        with patch.dict(os.environ, {'MENU_ID': menu.id, 'DATABASE_URL': 'unused'}), \
             patch('menu_storage.PostgresMenuStore.connect', return_value=self.store), \
             patch('ai_adapters.create_ai_adapter', return_value=provider), \
             redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                source = ''.join(cell['source'])
                # The test owns the connection and closes it after rollback.
                if cell['cell_type'] == 'code' and 'store.connection.close()' not in source:
                    exec(compile(source, '04_embeddings_retrieval.ipynb', 'exec'), namespace)
        self.assertEqual(namespace['indexed_count'], 2)
        self.assertEqual(len(namespace['candidates']), 2)


if __name__ == '__main__':
    unittest.main()
