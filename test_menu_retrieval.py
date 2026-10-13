"""Offline checks for paired embeddings, ranking, and factual constraints."""

from dataclasses import replace
from decimal import Decimal
import math
from types import SimpleNamespace
import unittest

from menu_embeddings import GeminiEmbeddingProvider
from menu_normalization import AttributedValue, Availability, VariationChoice, VariationGroup
from menu_retrieval import RetrievalFilters, retrieve_candidates
from menu_storage import VectorSearchHit, index_menu
from test_menu_storage import FakeProvider, FakeStore, _menu


class EmbeddingTests(unittest.TestCase):
    def test_batches_documents_and_uses_distinct_query_task_with_normalization(self):
        calls = []

        def embed_content(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[2.0] * 128)
                                               for _ in kwargs['contents']])

        provider = GeminiEmbeddingProvider(dimensions=128, batch_size=2,
            client=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))
        vectors = provider.embed_documents(['one', 'two', 'three'])
        provider.embed_query('pizza')
        self.assertEqual([len(call['contents']) for call in calls], [2, 1, 1])
        self.assertEqual([call['config']['task_type'] for call in calls],
                         ['RETRIEVAL_DOCUMENT', 'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'])
        self.assertAlmostEqual(math.hypot(*vectors[0]), 1)
        self.assertEqual(provider.embed_documents([]), ())

    def test_invalid_provider_output_does_not_replace_index(self):
        for vector in [(0, 0, 0), (float('nan'), 1, 2), (float('inf'), 1, 2)]:
            with self.subTest(vector=vector):
                provider = FakeProvider()
                provider.embed_documents = lambda texts: [vector for _ in texts]
                store = FakeStore()
                with self.assertRaises(ValueError):
                    index_menu(_menu(), provider, store)
                self.assertIsNone(store.saved)

    def test_adapter_rejects_missing_wrong_sized_and_zero_vectors(self):
        for rows in [None, [], [SimpleNamespace(values=[1])],
                     [SimpleNamespace(values=[0] * 128)],
                     [SimpleNamespace(values=[float('nan')] * 128)]]:
            provider = GeminiEmbeddingProvider(dimensions=128,
                client=SimpleNamespace(models=SimpleNamespace(
                    embed_content=lambda **kwargs: SimpleNamespace(embeddings=rows))))
            with self.assertRaises(ValueError):
                provider.embed_query('pizza')


class SearchStore:
    def __init__(self, menu=None, hits=()):
        self.menu = menu or _menu()
        self.hits = hits
        self.search_args = None

    def load_menu(self, menu_id):
        return self.menu

    def search(self, **kwargs):
        self.search_args = kwargs
        return self.hits


class QueryProvider:
    model_id = 'fake-embedding-model'
    dimensions = 3

    def embed_query(self, text):
        return (1, 0, 0)


def hit(item_id, variant_id=None, score=0.9):
    # Intentionally wrong metadata: factual values must come from the menu.
    return VectorSearchHit(item_id, variant_id, 'irrelevant', {'price': '0'}, score)


def with_required_choices(menu, *, available=True, optional=False):
    item = menu.items[0]
    choices = tuple(VariationChoice(
        id=name, name=name, description=None, price_adjustment=Decimal(price),
        availability=Availability('available' if available else 'unavailable'),
        is_preselected=False, source_reference=replace(item.source_reference, external_record_id=name),
    ) for name, price in [('Cheese', '3'), ('Tomato', '2')])
    group = VariationGroup('topping', 'Topping', None, 'CUSTOM', optional, 1, 2,
                           Availability('available'), choices,
                           replace(item.source_reference, external_record_id='topping'))
    item = replace(item, variation_groups=(group,),
                   embedding_text=item.embedding_text + '\nTopping: Cheese Tomato')
    return replace(menu, items=(item, *menu.items[1:]))


class RetrievalTests(unittest.TestCase):
    def retrieve(self, store, query='pizza', **kwargs):
        return retrieve_candidates(query, menu_id='store', store=store,
                                   provider=QueryProvider(), **kwargs)

    def test_budget_variant_and_unavailable_or_invented_hits(self):
        store = SearchStore(hits=(hit('ghost'), hit('water'), hit('pizza', 'missing'),
                                  hit('pizza'), hit('pizza', 'large'), hit('pizza', 'small')))
        result = self.retrieve(store, filters=RetrievalFilters(max_price=Decimal('10')))
        self.assertEqual([(row.item_id, row.variant_id) for row in result], [('pizza', 'small')])
        self.assertEqual(result[0].minimum_price, Decimal('10'))
        self.assertIsNotNone(result[0].semantic_score)
        self.assertIsNotNone(result[0].lexical_score)

    def test_lexical_branch_finds_candidates_missing_from_vector_results(self):
        result = self.retrieve(SearchStore())
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0].semantic_score)
        self.assertGreater(result[0].lexical_score, 0)

    def test_duplicate_semantic_hits_do_not_boost_rank(self):
        store = SearchStore(hits=(hit('pizza', 'large'),) * 3)
        result = self.retrieve(store)
        self.assertEqual(result[0].variant_id, 'large')
        self.assertAlmostEqual(result[0].score, 2 / 61)

    def test_unknown_requires_opt_in_and_remains_unknown(self):
        menu = _menu()
        item = replace(menu.items[0], availability=Availability('unknown'))
        store = SearchStore(replace(menu, items=(item,)))
        self.assertEqual(self.retrieve(store), ())
        result = self.retrieve(store, filters=RetrievalFilters(allow_unknown_availability=True))
        self.assertTrue(all(row.availability == 'unknown' for row in result))

    def test_unavailable_parent_blocks_available_variant(self):
        menu = _menu()
        store = SearchStore(replace(menu, items=(replace(menu.items[0],
            availability=Availability('unavailable')),)))
        self.assertEqual(self.retrieve(store, filters=RetrievalFilters(allow_unknown_availability=True)), ())

    def test_required_choices_are_included_in_budget_and_configuration(self):
        store = SearchStore(with_required_choices(_menu()))
        self.assertEqual(self.retrieve(store, filters=RetrievalFilters(max_price=Decimal('11'))), ())
        result = self.retrieve(store, filters=RetrievalFilters(max_price=Decimal('12')))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].minimum_price, Decimal('12'))
        self.assertEqual(result[0].required_choices, (('topping', ('Tomato',)),))

    def test_unavailable_required_choices_block_but_optional_can_be_omitted(self):
        self.assertEqual(self.retrieve(SearchStore(with_required_choices(_menu(), available=False))), ())
        result = self.retrieve(SearchStore(with_required_choices(_menu(), available=False, optional=True)))
        self.assertEqual(result[0].required_choices, ())

    def test_verified_dietary_currency_and_category_filters(self):
        menu = _menu()
        item = replace(menu.items[0], semantic_attributes=(AttributedValue('vegetarian', 'inferred'),))
        store = SearchStore(replace(menu, items=(item,)))
        filters = RetrievalFilters(dietary_attributes=('vegetarian',))
        self.assertEqual(self.retrieve(store, filters=filters), ())
        store.menu = replace(menu, items=(replace(item, dietary_attributes=('vegetarian',)),))
        self.assertEqual(len(self.retrieve(store, filters=filters)), 2)
        self.assertEqual(self.retrieve(store, filters=RetrievalFilters(currency='INR')), ())
        self.assertEqual(self.retrieve(store, filters=RetrievalFilters(category_ids=('drinks',))), ())
        self.assertEqual(len(self.retrieve(store, filters=RetrievalFilters(currency=' gbp '))), 2)

    def test_invalid_price_and_empty_query_are_rejected(self):
        for price in [10.0, Decimal('-1'), Decimal('NaN')]:
            with self.assertRaises(ValueError):
                RetrievalFilters(max_price=price)
        with self.assertRaises(ValueError):
            self.retrieve(SearchStore(), query=' ')


if __name__ == '__main__':
    unittest.main()
