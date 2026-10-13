"""Contract tests for interchangeable AI backends; no network or keys required."""

import json
import math
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import Mock

from ai_adapters import AIAdapter, AIError, AIQuotaError, AIResponseError, create_ai_adapter
from ai_adapters.gemini import GeminiAdapter
from ai_adapters.openai import OpenAIAdapter
from ai_adapters.qwen import QwenAdapter
from menu_descriptions import AIDescriptionGenerator, DescriptionRequest, generate_descriptions
from menu_dietary import infer_dietary_batch
from menu_embeddings import AIEmbeddingProvider
from enrichment_cache import EnrichmentCache
from test_semantic_enrichment import _menu, FakeGenerator


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}


def openai_response(value):
    return Obj(status='completed', output=[], output_text=json.dumps(value))


def client_for(provider, value):
    if provider == 'gemini':
        return Obj(models=Obj(generate_content=Mock(return_value=Obj(parsed=value))))
    return Obj(responses=Obj(create=Mock(return_value=openai_response(value))))


def adapter_for(provider, client):
    return create_ai_adapter(provider, env={}, client=client, dimensions=128, sleep=lambda _: None)


class AIContractTests(unittest.TestCase):
    def test_interface_is_abstract_and_factory_switches_provider(self):
        with self.assertRaises(TypeError):
            AIAdapter(text_model='text', embedding_model='embedding')
        for provider, cls in [('gemini', GeminiAdapter), ('openai', OpenAIAdapter)]:
            adapter = create_ai_adapter(env={'AI_PROVIDER': provider.upper()}, client=Obj())
            self.assertIsInstance(adapter, cls)
            self.assertTrue(adapter.model_id.startswith(provider + ':'))
            self.assertTrue(adapter.embedding_model_id.startswith(provider + ':'))
        with self.assertRaisesRegex(ValueError, 'AI_PROVIDER'):
            create_ai_adapter(env={'AI_PROVIDER': 'typo'}, client=Obj())

    def test_model_overrides_are_provider_specific_and_dimensions_are_validated(self):
        adapter = create_ai_adapter(env={'AI_PROVIDER': 'openai', 'OPENAI_TEXT_MODEL': 'custom-text',
            'GEMINI_TEXT_MODEL': 'ignored', 'AI_EMBEDDING_DIMENSIONS': '256'}, client=Obj())
        self.assertEqual(adapter.text_model, 'custom-text')
        self.assertEqual(adapter.dimensions, 256)
        with self.assertRaises(ValueError):
            create_ai_adapter('openai', env={}, client=Obj(), dimensions=2000)
        with self.assertRaises(ValueError):
            create_ai_adapter('gemini', env={}, client=Obj(), embedding_model='gemini-embedding-2')
        for provider in ['gemini', 'openai']:
            with self.assertRaisesRegex(AIError, 'API_KEY is required'):
                create_ai_adapter(provider, env={})

    def test_same_structured_output_contract_for_both_providers(self):
        for provider in ['gemini', 'openai']:
            with self.subTest(provider=provider):
                client = client_for(provider, {'answer': 'yes'})
                adapter = adapter_for(provider, client)
                self.assertEqual(adapter.generate_json('Answer yes.', SCHEMA), {'answer': 'yes'})
                if provider == 'openai':
                    args = client.responses.create.call_args.kwargs
                    self.assertTrue(args['text']['format']['strict'])
                    self.assertFalse(args['store'])
                    self.assertNotIn('temperature', args)
                else:
                    args = client.models.generate_content.call_args.kwargs
                    self.assertEqual(args['config']['response_json_schema'], SCHEMA)

    def test_invalid_json_schema_mismatch_and_non_object_root_fail_closed(self):
        for provider in ['gemini', 'openai']:
            for value in [{'answer': 3}, {}, {'answer': 'yes', 'extra': 2}]:
                with self.subTest(provider=provider, value=value):
                    adapter = adapter_for(provider, client_for(provider, value))
                    with self.assertRaises(AIResponseError):
                        adapter.generate_json('prompt', SCHEMA)
            client = client_for(provider, {})
            if provider == 'openai':
                client.responses.create.return_value.output_text = 'invalid json'
            else:
                client.models.generate_content.return_value = Obj(parsed=None, text='invalid json')
            with self.assertRaises(AIResponseError):
                adapter_for(provider, client).generate_json('prompt', SCHEMA)
        with self.assertRaises(ValueError):
            adapter_for('openai', Obj()).generate_json('prompt', {'type': 'array'})

    def test_incomplete_and_refused_output_is_rejected(self):
        client = client_for('openai', {'answer': 'yes'})
        client.responses.create.return_value.status = 'incomplete'
        with self.assertRaises(AIResponseError):
            adapter_for('openai', client).generate_json('prompt', SCHEMA)
        client.responses.create.return_value.status = 'completed'
        client.responses.create.return_value.output = [Obj(content=[Obj(type='refusal')])]
        with self.assertRaises(AIResponseError):
            adapter_for('openai', client).generate_json('prompt', SCHEMA)
        client = client_for('gemini', {'answer': 'yes'})
        client.models.generate_content.return_value.candidates = [Obj(finish_reason='MAX_TOKENS')]
        with self.assertRaises(AIResponseError):
            adapter_for('gemini', client).generate_json('prompt', SCHEMA)

    def test_retries_are_bounded_and_authentication_errors_are_not_retried(self):
        for provider in ['gemini', 'openai']:
            for status, expected_calls in [(429, 4), (503, 4), (401, 1)]:
                client = client_for(provider, {})
                method = client.responses.create if provider == 'openai' else client.models.generate_content
                error = RuntimeError('secret request details')
                error.status_code = status
                method.side_effect = error
                with self.assertRaises(AIError) as caught:
                    adapter_for(provider, client).generate_json('prompt', SCHEMA)
                self.assertNotIn('secret', str(caught.exception))
                self.assertEqual(method.call_count, expected_calls)

    def test_openai_embeddings_restore_input_order_batch_and_normalize(self):
        def embed(**kwargs):
            return Obj(data=[Obj(index=i, embedding=[float(i == 0), float(i == 1)] + [0.] * 126)
                             for i in reversed(range(len(kwargs['input'])))])
        client = Obj(embeddings=Obj(create=Mock(side_effect=embed)))
        adapter = OpenAIAdapter(client=client, dimensions=128, batch_size=2, sleep=lambda _: None)
        provider = AIEmbeddingProvider(adapter)
        vectors = provider.embed_documents(['one', 'two', 'three'])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(vectors[0][:2], (1., 0.))
        self.assertEqual(vectors[1][:2], (0., 1.))
        self.assertEqual([len(call.kwargs['input']) for call in client.embeddings.create.call_args_list], [2, 1])
        self.assertAlmostEqual(math.hypot(*provider.embed_query('one')), 1)
        self.assertEqual(provider.model_id, adapter.embedding_model_id)
        self.assertNotEqual(provider.model_id, adapter.model_id)

    def test_malformed_embedding_response_is_rejected(self):
        for rows in [[], [Obj(index=1, embedding=[1] * 128)],
                     [Obj(index=0, embedding=[0] * 128)],
                     [Obj(index=0, embedding=[float('nan')] * 128)],
                     [Obj(index=0, embedding=[1, 2])]]:
            client = Obj(embeddings=Obj(create=Mock(return_value=Obj(data=rows))))
            with self.assertRaises(AIResponseError):
                adapter_for('openai', client).embed_query('test')

    def test_exhausted_credits_are_reported_without_retry(self):
        client = client_for('openai', {})
        error = RuntimeError('provider details')
        error.status_code = 429
        error.body = {'type': 'insufficient_quota', 'code': 'credit_balance_exhausted'}
        client.responses.create.side_effect = error
        with self.assertRaisesRegex(AIQuotaError, 'billing'):
            adapter_for('openai', client).generate_json('prompt', SCHEMA)
        client.responses.create.assert_called_once()

    def test_description_and_dietary_domain_code_works_with_either_adapter(self):
        for provider in ['gemini', 'openai']:
            description = {'items': [{'item_id': 'naan', 'status': 'described',
                                      'description': 'A soft baked flatbread.'}]}
            adapter = adapter_for(provider, client_for(provider, description))
            output = AIDescriptionGenerator(adapter).generate([DescriptionRequest('naan', 'Naan', (), ())])
            self.assertEqual(output, {'naan': 'A soft baked flatbread.'})
            dietary = {'items': [{'item_id': 'naan', 'vegetarian': 'likely',
                                 'vegan': 'unknown', 'reason': 'Preparation is unknown.'}]}
            adapter = adapter_for(provider, client_for(provider, dietary))
            self.assertEqual(infer_dietary_batch(adapter, [{'item_id': 'naan'}]), dietary['items'])
            with self.assertRaises(AIResponseError):
                infer_dietary_batch(adapter, [{'item_id': 'different'}])

    def test_legacy_gemini_cache_is_reused_but_openai_switch_regenerates(self):
        menu = _menu()
        cache = EnrichmentCache(menu.id)
        generator = FakeGenerator({'naan': 'A soft baked flatbread.'})
        generator.model_id = 'gemini-2.5-flash'
        generate_descriptions(menu, generator, cache)
        client = client_for('gemini', {})
        generate_descriptions(menu, AIDescriptionGenerator(adapter_for('gemini', client)), cache)
        client.models.generate_content.assert_not_called()
        client = client_for('openai', {'items': [{'item_id': 'naan', 'status': 'described',
                                                'description': 'A warm, soft flatbread.'}]})
        generate_descriptions(menu, AIDescriptionGenerator(adapter_for('openai', client)), cache)
        self.assertEqual(cache.descriptions['naan']['model_id'], 'openai:gpt-4.1-mini')
        client.responses.create.assert_called_once()


class FakeSentenceModel:
    def __init__(self, rows=None):
        self.rows, self.calls = rows, []

    def get_sentence_embedding_dimension(self):
        return 6

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return self.rows if self.rows is not None else [[3.0, 4.0, 0, 0, 9.0, 9.0] for _ in texts]


class QwenAdapterTests(unittest.TestCase):
    def test_factory_needs_no_key_reads_model_override_and_does_not_pace(self):
        model = FakeSentenceModel()
        adapter = create_ai_adapter(env={'AI_PROVIDER': 'qwen', 'QWEN_EMBEDDING_MODEL': 'Qwen/custom',
                                         'AI_EMBEDDING_DIMENSIONS': '4'}, client=model,
                                    sleep=Mock(side_effect=AssertionError('paced')))
        self.assertIsInstance(adapter, QwenAdapter)
        self.assertEqual(adapter.embedding_model_id, 'qwen:Qwen/custom:4:retrieval-v1')
        adapter.embed_documents(['one'])
        adapter.embed_query('two')

    def test_queries_get_instruction_documents_do_not_and_vectors_are_truncated(self):
        model = FakeSentenceModel()
        provider = AIEmbeddingProvider(QwenAdapter(client=model, dimensions=4, batch_size=2))
        vectors = provider.embed_documents(['one', 'two', 'three'])
        query = provider.embed_query('pizza')
        self.assertEqual(provider.model_id, 'qwen:Qwen/Qwen3-Embedding-0.6B:4:retrieval-v1')
        self.assertEqual([call[0] for call in model.calls], [['one', 'two'], ['three'], ['pizza']])
        self.assertNotIn('prompt', model.calls[0][1])
        self.assertTrue(model.calls[2][1]['prompt'].startswith('Instruct: '))
        self.assertEqual(vectors[0], (0.6, 0.8, 0.0, 0.0))
        self.assertEqual(query, (0.6, 0.8, 0.0, 0.0))

    def test_rejects_unsupported_dimensions_bad_output_and_text_generation(self):
        with self.assertRaises(ValueError):
            QwenAdapter(client=FakeSentenceModel(), dimensions=7)
        for rows in [[], [[1.0, 2.0]], [[0.0] * 6], [[float('nan')] * 6]]:
            with self.subTest(rows=rows), self.assertRaises(AIResponseError):
                QwenAdapter(client=FakeSentenceModel(rows), dimensions=4).embed_query('pizza')
        with self.assertRaisesRegex(AIError, 'embedding-only'):
            QwenAdapter(client=FakeSentenceModel(), dimensions=4).generate_json('prompt', SCHEMA)


if __name__ == '__main__':
    unittest.main()
