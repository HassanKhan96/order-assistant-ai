# Switching AI providers

Set one flag in `.env`:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your-key
```

Or use `AI_PROVIDER=gemini` with `GEMINI_API_KEY`, or the keyless, embedding-only
`AI_PROVIDER=qwen` (see [Local Qwen embeddings](#local-qwen-embeddings)). OpenAI uses `OPENAI_API_KEY`.
The factory
reloads the project's `.env` each time it is called, overriding inherited shell
values. Re-run notebook setup/adapter creation cells after changing configuration.

Install dependencies in the notebook's Python environment:

```bash
.venv/bin/python -m pip install -r requirements-enrichment.txt
```

Application code uses the abstract `AIAdapter` contract:

```python
from ai_adapters import create_ai_adapter
from menu_descriptions import AIDescriptionGenerator
from menu_embeddings import AIEmbeddingProvider

ai = create_ai_adapter()
descriptions = AIDescriptionGenerator(ai)
embeddings = AIEmbeddingProvider(ai)

# Shared low-level methods for additional workflows:
# ai.generate_json(prompt, schema)  -> validated JSON object
# ai.embed_documents(texts)        -> ordered, normalized vectors
# ai.embed_query(text)             -> normalized query vector
```

`ai_adapters/base.py` defines the abstract interface and shared validation,
pacing, and retries. `ai_adapters/gemini.py` and `ai_adapters/openai.py` implement
separate SDK adapters. Domain prompts stay in `menu_descriptions.py` and
`menu_dietary.py`; notebooks do not call either company's SDK directly.

The generation schema must have an object root. Use required fields and
`additionalProperties: false` for every object for compatibility with OpenAI's
strict structured output. Both implementations validate returned JSON locally.
Incomplete, refused, malformed, and schema-incompatible responses fail without
being cached. HTTP 429 and server errors receive at most three retries; other
errors are surfaced as `AIError`. Exhausted account credits produce `AIQuotaError`
without retries; adding an API key does not resolve an exhausted credit balance.
No automatic provider fallback occurs.

## Models and configuration

| Setting | Default |
| --- | --- |
| `AI_PROVIDER` | `gemini` when unset; this workspace is configured for `openai` |
| `OPENAI_TEXT_MODEL` | `gpt-4.1-mini` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `GEMINI_TEXT_MODEL` | `gemini-2.5-flash` |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` |
| `QWEN_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |
| `AI_EMBEDDING_DIMENSIONS` | `768` |
| `AI_EMBEDDING_BATCH_SIZE` | `32` |
| `AI_REQUESTS_PER_MINUTE` | `60` for OpenAI; `10` for Gemini |

Text model names are configurable without domain-code changes, provided the
model supports the adapter's structured-output API. OpenAI embeddings support
`text-embedding-3-small` and `text-embedding-3-large`. Gemini embeddings currently
support `gemini-embedding-001`; embedding-2 uses different task/batching semantics
and is rejected instead of silently producing incompatible results. Dimensions
are validated against the supported model limits. Unsupported models or account
access failures surface through the selected provider.

## Notebook workflow and existing data

- Notebook 02 uses the adapter for descriptions; Strands semantic scoring remains local.
- Notebook 03 uses the adapter for dietary hints and retains its JSON cache workflow.
- Notebook 04 uses the adapter for both indexing and query embeddings.

Changing the flag does not modify the stored menu or regenerate enrichment.
Notebook 04 can index the already enriched PostgreSQL menu directly. If you
choose to rerun description or dietary generation, current cache entries must
match the selected model; curated descriptions are reusable across providers.
Legacy Gemini text cache IDs remain recognized. New IDs record provider and
model, and vector IDs also include dimensions and the retrieval strategy version.

Re-run notebook 04's indexing cell when switching embedding provider/model or
dimensions. Old vectors are not compared with the new query vectors. Legacy
unprefixed Gemini vector IDs also require re-indexing once. Do not overwrite a
provider's model identity in application code to reuse another provider's vectors.

## Local Qwen embeddings

`AI_PROVIDER=qwen` selects `ai_adapters/qwen.py`, which runs
`Qwen/Qwen3-Embedding-0.6B` in-process through `sentence-transformers` (install it
in the notebook with `%pip install sentence-transformers`). It needs no API key and
skips request pacing and retries. Vectors are truncated to `AI_EMBEDDING_DIMENSIONS`
(the model supports up to 1024) and stored under `qwen:` vector IDs. Only queries
receive the retrieval instruction prompt.

The adapter is embedding-only: `generate_json` raises `AIError`, so notebooks 02 and
03 still need `AI_PROVIDER=openai` or `gemini`. Notebook 04 can use Qwen for both
indexing and queries, or create `create_ai_adapter("qwen")` explicitly to compare it
with the configured API provider.
