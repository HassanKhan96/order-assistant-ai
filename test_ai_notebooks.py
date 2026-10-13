"""Execute enrichment notebook code in order using local fixtures and fake SDKs."""

from contextlib import chdir, redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from enrichment_cache import EnrichmentCache
from test_ai_adapters import adapter_for, client_for
from test_import_enrichment import FixtureScorer, saved_fixture


def run_cells(notebook, namespace):
    with redirect_stdout(io.StringIO()):
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # Ingestion/pip magic is replaced by the test's supplied menu.
                source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                                   if not line.startswith('%'))
                exec(compile(source, '<notebook>', 'exec'), namespace)


class AINotebookTests(unittest.TestCase):
    def test_semantic_notebook_runs_in_order_with_provider_neutral_generator(self):
        notebook = json.loads((Path(__file__).parent / '02_semantic_enrichment.ipynb').read_text())
        with TemporaryDirectory() as directory:
            menu, saved, _ = saved_fixture(directory)
            cache = EnrichmentCache.load(saved, menu.id)
            store = SimpleNamespace(initialize=Mock(), load_enrichment_cache=Mock(return_value=cache),
                                    save_enrichment_cache=Mock(), replace_menu=Mock())
            adapter = adapter_for('openai', client_for('openai', {}))
            namespace = {'menu': menu}
            with chdir(directory), patch.dict(os.environ, {'DATABASE_URL': 'unused'}), \
                 patch('menu_storage.PostgresMenuStore.connect', return_value=store), \
                 patch('ai_adapters.create_ai_adapter', return_value=adapter), \
                 patch('menu_enrichment.StrandsDeciderScorer', return_value=FixtureScorer()):
                run_cells(notebook, namespace)
            store.replace_menu.assert_called_once()
            self.assertTrue(namespace['enriched_menu'].items[0].inferred_description)
            adapter.client.responses.create.assert_not_called()  # Curated description reused.

    def test_dietary_notebook_uses_shared_adapter_and_records_provider_provenance(self):
        notebook = json.loads((Path(__file__).parent / '03_dietary_inference.ipynb').read_text())
        with TemporaryDirectory() as directory:
            menu, saved, _ = saved_fixture(directory)
            target = Path(directory) / 'enrichment'
            target.mkdir()
            (target / f'{menu.id}.json').write_text(saved.read_text())
            rows = {'items': [{'item_id': 'naan', 'vegetarian': 'likely', 'vegan': 'unknown',
                              'reason': 'Preparation unspecified.'}]}
            adapter = adapter_for('openai', client_for('openai', rows))
            namespace = {'menu': menu}
            with chdir(directory), patch('ai_adapters.create_ai_adapter', return_value=adapter):
                run_cells(notebook, namespace)
            output = json.loads((target / f'{menu.id}.dietary.json').read_text())
            self.assertEqual(output['results']['naan']['model_id'], adapter.model_id)
            self.assertEqual(output['results']['naan']['origin'], 'inferred')
            adapter.client.responses.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()
