# PostgreSQL and vector storage

The canonical menu, enrichment cache, and embedding documents are stored in
PostgreSQL. `pgvector` provides cosine search; a separate vector service is not
required at the current scale.

## Local setup

```bash
docker compose up -d postgres
python3 -m pip install -r requirements-storage.txt
```

Set `DATABASE_URL` as shown in `.env.example`, then initialize the schema once:

```python
import os
from menu_storage import PostgresMenuStore

store = PostgresMenuStore.connect(os.environ["DATABASE_URL"])
store.initialize()
```

## Ingestion and enrichment

Load the model-output cache from PostgreSQL and pass the database saver to the
existing enrichment functions. This preserves completed batches if a model run
is interrupted without writing JSON files:

```python
cache = store.load_enrichment_cache(menu.id)
descriptions = generate_descriptions(
    menu,
    description_generator,
    cache,
    cache_saver=store.save_enrichment_cache,
)
tags = score_menu(
    menu,
    descriptions,
    semantic_scorer,
    cache,
    cache_saver=store.save_enrichment_cache,
)
enriched_menu = apply_enrichment(menu, descriptions, tags)
store.replace_menu(enriched_menu)
```

`replace_menu` validates and atomically replaces the menu and all of its
categories, item facts, variants, variation groups, choices, and semantic
attributes.

## Import existing enrichment files without model calls

Existing `enrichment/<menu-id>.json` and `<menu-id>.dietary.json` files can be
imported directly. Start with the freshly normalized `menu` from notebook 01;
notebooks 02 and 03 do not need to regenerate their results:

```python
from pathlib import Path
from import_enrichment import prepare_import, import_saved_enrichment

prepared = prepare_import(
    menu,
    Path("enrichment") / f"{menu.id}.json",
    Path("enrichment") / f"{menu.id}.dietary.json",
)
print(prepared.report)  # Validation and counts; no database writes or model calls.
store.initialize()
report = import_saved_enrichment(prepared, store)
```

All saved cache entries are retained, including records for removed items.
Only descriptions and scores with matching input hashes are applied to the
canonical menu. Missing or changed inputs remain unenriched; the import never
generates replacement results. Matching local cache rows replace database rows,
while unrelated database cache entries are preserved. The menu and both caches
are imported in one transaction. An identical prepared import can be repeated
without deleting vectors; a different existing canonical snapshot is rejected
to prevent overwriting newer menu work.

Dietary hints live in `dietary_inference_cache`, separate from verified item
attributes. Read the raw cache with `store.load_dietary_cache(menu.id)`. Before
using an entry, check its model/prompt versions and compare its `input_hash`
with `menu_dietary.dietary_input_hash(item, menu)`. The cache intentionally retains
historical entries, so loading it alone does not prove that every row is current.
Notebook 04 still filters only verified dietary attributes; importing hints does
not make them verified or automatically use them for ranking.

After importing, run notebook 04 directly to generate embeddings and inspect
retrieval. The original JSON files remain unchanged.

## Vector indexing

Use the configured AI adapter to index the enriched menu (see `AI_PROVIDERS.md`):

```python
from ai_adapters import create_ai_adapter
from menu_embeddings import AIEmbeddingProvider
from menu_storage import index_menu

embedding_provider = AIEmbeddingProvider(create_ai_adapter())
count = index_menu(enriched_menu, embedding_provider, store)
```

The provider exposes `model_id`, `dimensions`, and `embed_documents(texts)`.
The model ID and dimension are stored with every vector. The first indexing run
creates an HNSW cosine index for that dimension, allowing models to change
without changing the base table schema. Full-precision HNSW indexes support up
to 2,000 dimensions; dimensions from 2,001 through 4,000 use pgvector's
half-precision index representation.

Use `store.search(...)` for vector retrieval with database-enforced menu,
availability, price, category, and verified dietary filters. Results must still
pass the application's final deterministic validation and ranking.

## Hybrid retrieval notebook

Open `04_embeddings_retrieval.ipynb` after persisting the enriched menu in notebook
02. It loads the menu from PostgreSQL, previews documents, indexes them, and
displays candidates with their required choices, minimum configured prices, and
separate vector/keyword scores. Install `requirements-enrichment.txt` as well as
`requirements-storage.txt` for the AI SDKs and notebook dependencies.

Set `AI_PROVIDER=openai` or `AI_PROVIDER=gemini` in `.env` with the corresponding
API key. The separate adapters share structured-generation and embedding methods.
See [AI_PROVIDERS.md](AI_PROVIDERS.md) for models, configuration, and cache behavior.
The default embedding dimension is 768 for both providers.
It batches documents in groups of 32. Each indexing run re-embeds the menu and
atomically replaces that model's documents; it does not yet reuse unchanged
vectors. Re-index after changing the canonical menu, model, or dimension.
Replacing a menu removes its old embeddings through foreign-key cascades.

```python
from decimal import Decimal
from menu_retrieval import RetrievalFilters, retrieve_candidates

candidates = retrieve_candidates(
    "something sweet", menu_id=enriched_menu.id, store=store,
    provider=embedding_provider,
    filters=RetrievalFilters(max_price=Decimal("5.00"), currency="GBP"),
    limit=5,
)
```

Vector search uses PostgreSQL filters. Keyword matching scans the canonical
snapshot loaded from PostgreSQL; both branches pass the same application checks
before reciprocal-rank fusion. Only source-declared dietary attributes can meet
dietary filters. Unknown availability is excluded unless
`allow_unknown_availability=True`, and remains labelled unknown when included.
Unavailable items, variants, and required choices are always excluded.

`minimum_price` includes the cheapest valid required custom choices; their IDs
are returned in `required_choices`. Optional groups are omitted. These are
candidate configurations, not customer-selected orders. Budget filters never
use inferred descriptions or vector metadata as facts.

Filters must currently be supplied explicitly. Request text alone does not
enforce budgets, dietary needs, or allergens. Allergen safety is not supported
by this retrieval interface. Intent extraction, controlled fallback, and final
response validation are the next stages. Scores are ranks, not confidence or
proof of relevance. Semantic recall is limited to at most 100 vector candidates
and may be reduced by approximate-index filtering.

## Verification

```bash
.venv/bin/python -m unittest discover -v
TEST_DATABASE_URL=postgresql:///postgres .venv/bin/python -m unittest test_menu_storage_integration -v
```

The opt-in integration suite uses a unique temporary schema inside a transaction
and rolls back every test, including schema creation. The database must have
pgvector available and permit schema creation. Tests use fake embeddings and do
not call model APIs.
