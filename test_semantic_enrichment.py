"""Offline tests for description and semantic enrichment behavior."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from enrichment_cache import EnrichmentCache, content_hash
from menu_descriptions import (
    DescriptionGenerationError,
    DescriptionRequest,
    GeminiDescriptionGenerator,
    PROMPT_VERSION,
    apply_description_overrides,
    check_description,
    clean_generated_description,
    generate_descriptions,
    load_cached_descriptions,
    needs_description,
)
from menu_enrichment import (
    SCALE_QUESTIONS,
    TAG_QUESTIONS,
    ItemScores,
    ScaleScore,
    StrandsDeciderScorer,
    apply_enrichment,
    describe_item,
    score_menu,
)
from menu_normalization import normalize_api_menu
from menu_validation import validate_menu


def _item(identifier, name, description=None, category="mains", groups=()):
    return {
        "id": identifier,
        "name": name,
        "description": description,
        "category_id": category,
        "availability": "AVAILABLE",
        "sizes": [],
        "menu_item_variations": list(groups),
        "pricing_and_availabilities": [
            {"channel": "ONLINE_DELIVERY", "price": 10, "availability": "AVAILABLE"}
        ],
    }


def _menu(records=None, categories=None):
    payload = SimpleNamespace(
        menu=records or [_item("naan", "Plain Naan")],
        categories=categories or [{"id": "mains", "name": "Mains"}],
        menu_endpoint="items-url",
        categories_endpoint="categories-url",
        fetched_at=datetime.now(timezone.utc),
    )
    return normalize_api_menu(
        payload,
        menu_id="store",
        restaurant_name="Restaurant",
        currency="GBP",
        channel="ONLINE_DELIVERY",
    )


def _scores(**tag_overrides):
    tags = {name: 0.1 for name in TAG_QUESTIONS}
    tags.update(tag_overrides)
    scales = {
        name: ScaleScore(
            value=0.5,
            probabilities={level: 1 / len(levels) for level in levels},
            confidence=0.4,
        )
        for name, (_, levels) in SCALE_QUESTIONS.items()
    }
    return ItemScores(tags, scales)


class FakeGenerator:
    model_id = "fake-description-model"

    def __init__(self, outputs=None, fail_on_call=None):
        self.outputs = outputs or {}
        self.fail_on_call = fail_on_call
        self.calls = []

    def generate(self, requests):
        self.calls.append(tuple(requests))
        if self.fail_on_call == len(self.calls):
            raise DescriptionGenerationError("quota")
        return {request.item_id: self.outputs.get(request.item_id) for request in requests}


class FakeScorer:
    model_id = "fake-decider"

    def __init__(self, scores=None, fail_on_call=None):
        self.scores = scores or _scores()
        self.fail_on_call = fail_on_call
        self.calls = []

    def score(self, text):
        self.calls.append(text)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("interrupted")
        return self.scores


class DescriptionTests(unittest.TestCase):
    def test_placeholder_detection_and_guard(self):
        menu = _menu([
            _item("missing", "Missing"),
            _item("test", "Test", "Test dummy description for testing"),
            _item("real", "Real", "Crisp pastry with a fragrant filling."),
        ])
        self.assertTrue(needs_description(menu.items[0]))
        self.assertTrue(needs_description(menu.items[1]))
        self.assertFalse(needs_description(menu.items[2]))
        for rejected in ("", "Vegetarian curry", "Contains NUTS", "Gluten-Free", "healthy"):
            self.assertIsNotNone(check_description(rejected))
        self.assertIsNone(check_description("A coconut curry with a doughnut-like shape."))
        self.assertIsNotNone(check_description("x" * 201))
        self.assertEqual(clean_generated_description("A vegan dish."), "A dish.")
        shortened = clean_generated_description("word " * 60)
        self.assertIsNotNone(shortened)
        self.assertLessEqual(len(shortened), 200)
        self.assertIsNone(check_description(shortened))

    def test_generation_batches_guards_and_uses_cache(self):
        menu = _menu([_item("one", "One"), _item("two", "Two"), _item("real", "Real", "Real text")])
        cache = EnrichmentCache(menu.id)
        generator = FakeGenerator({"one": "A crisp savory snack.", "two": "A vegan dish."})
        results = generate_descriptions(menu, generator, cache, batch_size=1)
        self.assertEqual([result.status for result in results], ["described", "described"])
        self.assertEqual(results[1].text, "A dish.")
        self.assertEqual(len(generator.calls), 4)
        cached_generator = FakeGenerator()
        cached = generate_descriptions(menu, cached_generator, cache)
        self.assertEqual(cached, results)
        self.assertEqual(cached_generator.calls, [])
        self.assertEqual(cache.descriptions["one"]["prompt_version"], PROMPT_VERSION)

    def test_missing_model_description_fails_instead_of_caching_unknown(self):
        menu = _menu([_item("one", "One")])
        cache = EnrichmentCache(menu.id)
        with self.assertRaisesRegex(DescriptionGenerationError, "item 'one'"):
            generate_descriptions(menu, FakeGenerator(), cache)
        self.assertNotIn("one", cache.descriptions)

    def test_non_described_cache_rows_are_regenerated(self):
        menu = _menu([_item("one", "One")])
        for old_status in ("unknown", "rejected"):
            with self.subTest(status=old_status):
                cache = EnrichmentCache(menu.id)
                first = generate_descriptions(
                    menu, FakeGenerator({"one": "First description."}), cache
                )
                cache.descriptions["one"].update(status=old_status, text=None)
                replacement = FakeGenerator({"one": "Replacement description."})
                regenerated = generate_descriptions(menu, replacement, cache)
                self.assertEqual(regenerated[0].status, "described")
                self.assertEqual(regenerated[0].text, "Replacement description.")
                self.assertEqual(len(replacement.calls), 1)
                self.assertNotEqual(regenerated, first)

    def test_curated_override_is_validated_and_reused_without_model_call(self):
        menu = _menu([_item("one", "One")])
        cache = EnrichmentCache(menu.id)
        applied = apply_description_overrides(
            menu, cache, {"one": "A concise curated menu description."}
        )
        generator = FakeGenerator()
        results = generate_descriptions(menu, generator, cache)
        self.assertEqual(applied, 1)
        self.assertEqual(results[0].status, "described")
        self.assertEqual(results[0].text, "A concise curated menu description.")
        self.assertEqual(generator.calls, [])

    def test_completed_description_batches_survive_failure(self):
        menu = _menu([_item("one", "One"), _item("two", "Two")])
        cache = EnrichmentCache(menu.id)
        generator = FakeGenerator({"one": "First description."}, fail_on_call=2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            with self.assertRaises(DescriptionGenerationError):
                generate_descriptions(menu, generator, cache, cache_path=path, batch_size=1)
            loaded = EnrichmentCache.load(path, menu.id)
            self.assertIn("one", loaded.descriptions)
            self.assertNotIn("two", loaded.descriptions)

    def test_loads_completed_descriptions_without_a_generator(self):
        menu = _menu()
        cache = EnrichmentCache(menu.id)
        generated = generate_descriptions(
            menu, FakeGenerator({"naan": "A soft flatbread served as a side."}), cache
        )
        self.assertEqual(load_cached_descriptions(menu, cache, require_all=True), generated)
        cache.descriptions.clear()
        with self.assertRaisesRegex(ValueError, "no current description"):
            load_cached_descriptions(menu, cache, require_all=True)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class HttpError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(str(status_code))


class GeminiAdapterTests(unittest.TestCase):
    def test_loads_dotenv_and_overrides_inherited_shell_key(self):
        def load_new_key(*, override):
            self.assertTrue(override)
            os.environ["GEMINI_API_KEY"] = "new-key"

        google = ModuleType("google")
        genai = ModuleType("google.genai")
        client = Mock()
        genai.Client = client
        google.genai = genai
        with patch.dict(sys.modules, {"google": google, "google.genai": genai}):
            with patch.dict("os.environ", {"GEMINI_API_KEY": "old-key"}, clear=True):
                with patch("menu_descriptions.load_dotenv", side_effect=load_new_key) as loader:
                    GeminiDescriptionGenerator()

        loader.assert_called_once_with(override=True)
        client.assert_called_once_with(api_key="new-key")

    def test_parses_rows_and_ignores_unrequested_ids(self):
        response = SimpleNamespace(text=json.dumps([
            {"item_id": "one", "status": "described", "description": " A dish. "},
            {"item_id": "other", "status": "described", "description": "Ignore"},
        ]), parsed=None)
        client = SimpleNamespace(models=FakeModels([response]))
        generator = GeminiDescriptionGenerator(client=client, sleep=lambda _: None)
        output = generator.generate([DescriptionRequest("one", "One", (), ())])
        self.assertEqual(output, {"one": "A dish."})

    def test_retries_server_errors_but_not_client_errors(self):
        response = SimpleNamespace(text="[]", parsed=None)
        models = FakeModels([HttpError(500), response])
        generator = GeminiDescriptionGenerator(
            client=SimpleNamespace(models=models), sleep=lambda _: None, clock=lambda: 0
        )
        generator.generate([DescriptionRequest("one", "One", (), ())])
        self.assertEqual(models.calls, 2)
        failing = GeminiDescriptionGenerator(
            client=SimpleNamespace(models=FakeModels([HttpError(400)])), sleep=lambda _: None
        )
        with self.assertRaises(DescriptionGenerationError):
            failing.generate([DescriptionRequest("one", "One", (), ())])

    def test_invalid_json_and_missing_key_are_clear_errors(self):
        invalid = GeminiDescriptionGenerator(
            client=SimpleNamespace(models=FakeModels([SimpleNamespace(text="not json", parsed=None)])),
            sleep=lambda _: None,
        )
        with self.assertRaises(DescriptionGenerationError):
            invalid.generate([DescriptionRequest("one", "One", (), ())])
        with patch.dict("os.environ", {}, clear=True):
            with patch("menu_descriptions.load_dotenv"):
                with self.assertRaisesRegex(DescriptionGenerationError, "GEMINI_API_KEY"):
                    GeminiDescriptionGenerator()

    def test_spaces_requests_at_configured_rate(self):
        now = [0.0]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        responses = [SimpleNamespace(text="[]", parsed=None)] * 2
        generator = GeminiDescriptionGenerator(
            client=SimpleNamespace(models=FakeModels(responses)),
            requests_per_minute=10,
            sleep=sleep,
            clock=lambda: now[0],
        )
        request = [DescriptionRequest("one", "One", (), ())]
        generator.generate(request)
        generator.generate(request)
        self.assertEqual(sleeps, [6.0])


class EnrichmentTests(unittest.TestCase):
    def test_describe_item_chooses_best_input_and_omits_test_category(self):
        menu = _menu(categories=[{"id": "mains", "name": "Test cat"}])
        item = menu.items[0]
        self.assertEqual(describe_item(item, menu).input_level, "name_only")
        self.assertEqual(
            describe_item(item, menu, "Usually served as a bread side.").input_level,
            "inferred_description",
        )
        source_item = replace(item, description="A source description")
        described = describe_item(source_item, replace(menu, items=(source_item,)), "Ignored")
        self.assertEqual(described.input_level, "source_description")
        self.assertNotIn("Ignored", described.text)

    def test_score_cache_and_apply_provenance(self):
        menu = _menu()
        cache = EnrichmentCache(menu.id)
        descriptions = generate_descriptions(
            menu, FakeGenerator({"naan": "A soft flatbread usually served as a side."}), cache
        )
        scorer = FakeScorer(_scores(savory=0.8, light=0.7, shareable=0.95))
        tags = score_menu(menu, descriptions, scorer, cache)
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(score_menu(menu, descriptions, FakeScorer(), cache), tags)
        enriched = apply_enrichment(menu, descriptions, tags)
        item = enriched.items[0]
        self.assertEqual(item.description, None)
        self.assertEqual(item.inferred_description.origin, "inferred")
        self.assertEqual([value.value for value in item.semantic_attributes], ["light", "savory"])
        self.assertIn("Inferred description:", item.embedding_text)
        self.assertIn("Semantic attributes: light, savory", item.embedding_text)
        self.assertTrue(validate_menu(enriched).is_valid)
        with self.assertRaises(ValueError):
            apply_enrichment(enriched, descriptions, tags)

    def test_score_rejects_incomplete_values_and_saves_on_interruption(self):
        menu = _menu([_item("one", "One"), _item("two", "Two")])
        bad = ItemScores(tags={"sweet": 0.5}, scales={})
        with self.assertRaises(ValueError):
            score_menu(menu, (), FakeScorer(bad), EnrichmentCache(menu.id))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            cache = EnrichmentCache(menu.id)
            with self.assertRaises(RuntimeError):
                score_menu(
                    menu, (), FakeScorer(fail_on_call=2), cache, cache_path=path, save_every=25
                )
            loaded = EnrichmentCache.load(path, menu.id)
            self.assertIn("one", loaded.tags)
            self.assertNotIn("two", loaded.tags)

    def test_score_rejects_out_of_range_values_and_reports_progress(self):
        menu = _menu()
        bad_scores = _scores()
        bad_scores = replace(bad_scores, tags={**bad_scores.tags, "sweet": 1.1})
        with self.assertRaises(ValueError):
            score_menu(menu, (), FakeScorer(bad_scores), EnrichmentCache(menu.id))
        progress = []
        score_menu(
            menu,
            (),
            FakeScorer(),
            EnrichmentCache(menu.id),
            progress=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(progress, [(1, 1)])

    def test_changed_description_invalidates_tag_cache(self):
        menu = _menu()
        cache = EnrichmentCache(menu.id)
        first = generate_descriptions(menu, FakeGenerator({"naan": "First description."}), cache)
        first_scorer = FakeScorer()
        score_menu(menu, first, first_scorer, cache)
        changed = (replace(first[0], text="Changed description", input_hash=content_hash("changed")),)
        second_scorer = FakeScorer()
        score_menu(menu, changed, second_scorer, cache)
        self.assertEqual(len(second_scorer.calls), 1)

    def test_fully_cached_strands_run_does_not_load_engine(self):
        menu = _menu()
        cache = EnrichmentCache(menu.id)
        model_id = "StrandsAgents/strands-decider-2B-hobson-v19"
        initial = FakeScorer()
        initial.model_id = model_id
        expected = score_menu(menu, (), initial, cache)

        class NeverLoadScorer(StrandsDeciderScorer):
            def _load_engine(self):
                raise AssertionError("engine should not load for cache hits")

        progress = []
        self.assertEqual(
            score_menu(
                menu,
                (),
                NeverLoadScorer(),
                cache,
                progress=lambda done, total: progress.append((done, total)),
            ),
            expected,
        )
        self.assertEqual(progress, [(0, 0)])

    def test_apply_rejects_bad_arguments_and_ignores_stale_description(self):
        menu = _menu([_item("naan", "Plain Naan", "A real source description")])
        stale_menu = _menu()
        stale = generate_descriptions(
            stale_menu, FakeGenerator({"naan": "An inferred description."}), EnrichmentCache(menu.id)
        )
        enriched = apply_enrichment(menu, stale, ())
        self.assertIsNone(enriched.items[0].inferred_description)
        with self.assertRaises(ValueError):
            apply_enrichment(menu, (), (), threshold=0)
        with self.assertRaises(ValueError):
            apply_enrichment(menu, (), (), excluded_tags=("not-a-tag",))


try:
    from strands_decider.schema import NoulAnswer, ScoreAnswer, SystemOneResponse, Usage
except ImportError:
    NoulAnswer = ScoreAnswer = SystemOneResponse = Usage = None


@unittest.skipIf(SystemOneResponse is None, "strands-decider is not installed")
class StrandsAdapterTests(unittest.TestCase):
    def test_maps_engine_answers_to_stable_vocabulary(self):
        class Engine:
            def ask(self, state, questions):
                answers = {name: NoulAnswer(noul=0.8) for name in TAG_QUESTIONS}
                for name, (_, levels) in SCALE_QUESTIONS.items():
                    probabilities = {str(index): 1 / len(levels) for index in range(len(levels))}
                    answers[name] = ScoreAnswer(
                        score=(len(levels) - 1) / 2,
                        legend={str(index): level for index, level in enumerate(levels)},
                        probabilities=probabilities,
                        confidence=0.4,
                    )
                return SystemOneResponse(
                    model="fake", answers=answers, usage=Usage(input_tokens=1, output_tokens=13)
                )

        scores = StrandsDeciderScorer(engine=Engine(), checkpoint="fake").score("A menu item")
        self.assertEqual(set(scores.tags), set(TAG_QUESTIONS))
        self.assertEqual(scores.scales["heaviness"].value, 0.5)
        self.assertEqual(
            set(scores.scales["spice_level"].probabilities),
            set(SCALE_QUESTIONS["spice_level"][1]),
        )


class CacheTests(unittest.TestCase):
    def test_round_trip_missing_and_foreign_menu(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "cache.json"
            empty = EnrichmentCache.load(path, "menu")
            empty.descriptions["x"] = {"status": "unknown"}
            empty.save(path)
            self.assertEqual(EnrichmentCache.load(path, "menu").descriptions, empty.descriptions)
            with self.assertRaises(ValueError):
                EnrichmentCache.load(path, "other")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
