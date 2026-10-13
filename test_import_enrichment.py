"""Saved enrichment imports must reuse current work without model calls."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from enrichment_cache import EnrichmentCache
from import_enrichment import prepare_import
from menu_descriptions import apply_description_overrides, load_cached_descriptions
from menu_dietary import dietary_input_hash
from menu_enrichment import apply_enrichment, score_menu
from test_semantic_enrichment import _menu, _scores


class FixtureScorer:
    model_id = 'fixture-model'

    def score(self, text):
        return _scores(savory=0.9)


def saved_fixture(directory):
    menu = _menu()
    cache = EnrichmentCache(menu.id)
    apply_description_overrides(menu, cache, {'naan': 'A soft baked flatbread.'})
    descriptions = load_cached_descriptions(menu, cache, require_all=True)
    tags = score_menu(menu, descriptions, FixtureScorer(), cache)
    enriched = apply_enrichment(menu, descriptions, tags)
    dietary = {'naan': {
        'origin': 'inferred', 'vegetarian': 'likely', 'vegan': 'unknown',
        'reason': 'Preparation is not fully specified.', 'model_id': 'fixture-dietary',
        'prompt_version': 'dietary-v1',
        'input_hash': dietary_input_hash(enriched.items[0], enriched),
    }}
    enrichment_path = Path(directory) / 'saved.json'
    dietary_path = Path(directory) / 'saved.dietary.json'
    cache.save(enrichment_path)
    dietary_path.write_text(json.dumps({'menu_id': menu.id, 'results': dietary}))
    return menu, enrichment_path, dietary_path


class ImportPreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.menu, self.enrichment_path, self.dietary_path = saved_fixture(self.directory)

    def test_current_cache_is_applied_and_dietary_stays_inferred(self):
        prepared = prepare_import(self.menu, self.enrichment_path, self.dietary_path)
        item = prepared.menu.items[0]
        self.assertEqual(item.inferred_description.value, 'A soft baked flatbread.')
        self.assertIn('savory', [row.value for row in item.semantic_attributes])
        self.assertEqual(item.dietary_attributes, ())
        self.assertEqual(prepared.report['current_dietary_hints'], 1)

    def test_stale_entries_are_preserved_but_not_applied(self):
        changed = replace(self.menu, items=(replace(self.menu.items[0], name='Changed name',
                          embedding_text='Changed name\nMains'),))
        prepared = prepare_import(changed, self.enrichment_path, self.dietary_path)
        self.assertEqual(prepared.report['cached_descriptions'], 1)
        self.assertEqual(prepared.report['applied_descriptions'], 0)
        self.assertEqual(prepared.report['applied_semantic_scores'], 0)
        self.assertEqual(prepared.report['current_dietary_hints'], 0)
        self.assertIsNone(prepared.menu.items[0].inferred_description)
        self.assertEqual(prepared.menu.items[0].semantic_attributes, ())

    def test_removed_items_are_kept_only_in_cache(self):
        payload = json.loads(self.enrichment_path.read_text())
        payload['tags']['removed'] = payload['tags']['naan']
        self.enrichment_path.write_text(json.dumps(payload))
        prepared = prepare_import(self.menu, self.enrichment_path, self.dietary_path)
        self.assertEqual(prepared.report['cached_semantic_scores'], 2)
        self.assertEqual(prepared.report['applied_semantic_scores'], 1)
        self.assertEqual(len(prepared.menu.items), 1)

    def test_foreign_menu_and_invalid_dietary_labels_are_rejected(self):
        payload = json.loads(self.dietary_path.read_text())
        payload['menu_id'] = 'foreign'
        self.dietary_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'different menu'):
            prepare_import(self.menu, self.enrichment_path, self.dietary_path)
        payload['menu_id'] = self.menu.id
        payload['results']['naan']['vegetarian'] = 'verified'
        self.dietary_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'invalid vegetarian'):
            prepare_import(self.menu, self.enrichment_path, self.dietary_path)

    def test_current_invalid_scores_are_rejected(self):
        payload = json.loads(self.enrichment_path.read_text())
        payload['tags']['naan']['scores']['tags']['savory'] = 4
        self.enrichment_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'out-of-range'):
            prepare_import(self.menu, self.enrichment_path, self.dietary_path)


if __name__ == '__main__':
    unittest.main()
