# Agent Context

## Project overview

This repository prototypes a menu recommendation agent. It ingests restaurant menu data, normalizes it into a canonical domain model, validates the result, and will ultimately recommend grounded menu items or meals from natural-language requests.

The project is currently notebook-first and early-stage. API ingestion,
normalization, validation, and semantic enrichment are implemented. Read
`DESIGN.md` before making architectural or domain-model changes; it is the
source of truth for intended behavior and scope.

## Current files

- `DESIGN.md`: architecture, domain model, safety rules, roadmap, and design decisions.
- `01_menu_api_ingestion.ipynb`: API-ingestion prototype using the Python standard library.
- `menu_normalization.py`: immutable dataclasses and API-to-canonical-menu normalization.
- `menu_validation.py`: deterministic validation and structured error/warning reports.
- `test_menu_variations.py`: unit tests for configurable items, variants, and variation groups.
- `menu_descriptions.py`: guarded Gemini description generation for sparse items.
- `menu_enrichment.py`: local semantic scoring and immutable enrichment application.
- `enrichment_cache.py`: atomic, resumable JSON cache for model outputs.
- `02_semantic_enrichment.ipynb`: enrichment workflow, inspection, and quality comparison.
- `requirements-enrichment.txt`: optional Gemini and Strands dependencies.
- `test_semantic_enrichment.py`: offline enrichment, cache, and adapter tests.

## Core invariants

- Treat the supplied menu as the only source of truth. Never invent items, prices, variants, ingredients, availability, dietary facts, or allergens.
- Keep factual constraints deterministic: pricing, totals, availability, category filtering, verified dietary properties, and allergen handling belong in application code.
- Use an LLM only for language interpretation, non-safety semantic enrichment, candidate comparison, and response composition.
- Preserve provenance and distinguish source facts from inferred values.
- Represent money with `Decimal` (or integer minor units in future code), never binary floating-point values.
- Keep `unknown` availability distinct from `available`.
- Treat item variants as first-class purchasable options. A size price is the channel base price plus its size adjustment.
- Include required custom-choice adjustments when calculating a fully configured price, and enforce variation-group selection limits.
- Never relax safety-related constraints. When relaxing an eligible preference, state what changed and by how much.
- Do not recommend a record unless it exists in the normalized menu and passes deterministic validation.

## Implementation conventions

- Target clear, typed Python with small, testable functions.
- Preserve the immutable dataclass-based domain model unless an intentional migration is documented.
- Raise `MenuNormalizationError` when source data cannot be mapped without guessing.
- Return all validation findings through `ValidationReport`; use errors for blocking problems and warnings for usable but uncertain data.
- Retain source IDs and endpoint references on normalized records.
- Normalize external strings at system boundaries (for example, currency and channel values).
- Avoid embedding credentials or live API secrets in code or notebooks. Read them from environment variables or a secret manager.
- Keep provider-specific extraction separate from canonical normalization and validation.
- Avoid adding production infrastructure, ordering/payment behavior, or a UI unless the project scope is explicitly expanded.

## Working workflow

1. Inspect `DESIGN.md` and the relevant implementation before changing behavior.
2. Add or update focused tests with every behavioral change.
3. Run the test suite from the repository root:

   ```bash
   python3 -m unittest discover -v
   ```

4. If notebook behavior changes, keep its imports and examples aligned with the Python modules and run the affected cells when practical.
5. Update `DESIGN.md` when a decision changes the architecture, domain model, constraints, or roadmap.

## Near-term direction

Follow the roadmap in `DESIGN.md`. The next work is embeddings and hybrid
retrieval, followed by structured intent extraction, deterministic candidate
filtering and ranking, grounded response generation, and repeatable
recommendation-quality evaluation.

## Scope guardrails

The first prototype does not place orders or take payments, guarantee medical or allergen safety from inferred data, support every PDF layout, train a custom foundation model, or provide a production API or multi-tenant platform. Prefer the smallest change that advances the documented prototype.
