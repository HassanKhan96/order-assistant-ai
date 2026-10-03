# Menu Recommendation Agent — Design

Status: Draft 0.1  
Last updated: 2026-09-27

## 1. Purpose

Build a conversational agent that understands a customer's request and recommends real items, or a coherent combination of items, from a supplied restaurant menu.

Example requests:

- "Find me an item under 500 rupees."
- "Find me something sweet."
- "I want something sweet under 500 rupees, but not too heavy."
- "Suggest a vegetarian dinner for two under 2,000 rupees."

The agent must use the provided menu as its source of truth. When nothing satisfies the request exactly, it should return the closest useful alternatives and clearly explain which requirement could not be met.

The first implementation will be a Python notebook prototype. The design should allow that prototype to evolve into a reusable Python application or service without changing the core domain model.

## 2. Goals

- Understand natural-language requests with an LLM.
- Ingest menus from API endpoints and PDF documents through a shared abstraction.
- Normalize all source data into one canonical menu representation.
- Support semantic requests such as "sweet," "light," "filling," or "refreshing."
- Enforce factual constraints such as price, availability, and explicitly known dietary properties.
- Recommend individual items and, later, multi-item meals.
- Produce useful nearest alternatives when no exact match exists.
- Explain why each recommendation matches and disclose any unmet constraints.
- Evaluate recommendation quality with repeatable test cases.

## 3. Non-goals for the first prototype

- Placing or paying for an order.
- Managing inventory in real time.
- Supporting every menu layout or low-quality scanned PDF.
- Giving medical guarantees based on inferred ingredients or allergens.
- Training or fine-tuning a custom foundation model.
- Building a production API, user interface, or persistent multi-tenant platform.

## 4. Design principles

### 4.1 The menu is the source of truth

The LLM must not invent items, variants, prices, ingredients, availability, or dietary facts. A recommendation must refer to a normalized menu record returned by the retrieval layer.

### 4.2 Use the LLM for language and judgment

The LLM should:

- Interpret ambiguous natural language.
- Convert a request into structured intent and constraints.
- Infer non-safety semantic attributes during menu enrichment.
- Plan retrieval.
- Compare valid candidates.
- Compose a coherent item or meal recommendation.
- Explain exact and approximate matches conversationally.

### 4.3 Use deterministic logic for facts and arithmetic

Application logic should:

- Compare prices and calculate totals.
- Apply availability and category filters.
- Enforce verified dietary and allergen constraints.
- Validate that recommended records exist in the menu.
- Calculate the difference between a budget and an alternative's price.
- Validate a multi-item meal against the requested total budget.

### 4.4 Separate facts from inferences

Information explicitly stated by the source menu must be distinguishable from information inferred by an LLM. Inferred attributes may improve discovery, but inferred allergens or dietary properties must never be presented as guaranteed facts.

### 4.5 Degrade honestly

When an exact match does not exist, the agent may relax eligible preferences in a controlled order. It must identify what was relaxed and by how much. Safety-related constraints are not relaxed.

## 5. High-level architecture

```text
Menu API -------------------+
                            |
Menu PDF -> extraction/OCR -+-> normalization -> validation -> enrichment
                                                          |
                                                          +-> structured store
                                                          +-> vector index

Customer request -> LLM intent extraction -> search plan
                                              |
                                              v
                            hybrid retrieval and filtering
                           /                               \
                  exact candidates                 fallback candidates
                           \                               /
                            deterministic validation/ranking
                                              |
                                              v
                              LLM recommendation and explanation
```

The architecture has two main flows:

1. **Offline or ingestion flow:** read, normalize, validate, enrich, and index a menu.
2. **Online or recommendation flow:** interpret a request, retrieve candidates, enforce constraints, rank results, and generate a grounded answer.

## 6. Canonical domain model

The exact Python representation will be selected during implementation, likely typed data classes or Pydantic models. The conceptual model is defined here first.

### 6.1 Menu

```text
Menu
- id
- restaurant_name
- currency
- categories[]
- items[]
- source_metadata
- created_at
- updated_at
```

One currency per menu is sufficient for the first prototype. Money should eventually be represented as an exact decimal value or integer minor units, never a binary floating-point number.

### 6.2 Category

```text
Category
- id
- name
- description: optional
- parent_category_id: optional
- aliases[]
- source_reference
```

Examples include Desserts, Drinks, Starters, Main Course, and nested categories such as Cold Drinks.

### 6.3 Menu item

```text
MenuItem
- id
- name
- description: optional
- category_ids[]
- base_price: optional
- currency
- variants[]
- variation_groups[]
- ingredients[]
- allergens[]
- dietary_attributes[]
- semantic_attributes[]
- availability
- source_reference
- embedding_text
```

An item may omit `base_price` when every purchasable option is represented by a variant.

### 6.4 Item variant

```text
ItemVariant
- id
- item_id
- name
- price
- availability
- source_reference
```

Variants are first-class purchasable choices. This prevents the agent from incorrectly rejecting an item when one size is within budget or quoting the price of the wrong size.

Items can also have multiple variation groups, such as Meat, Crust, and Sauce.
Each group retains its name, type, description, availability, optional flag,
minimum and maximum selection counts, and source reference. Custom choices
retain their IDs, names, descriptions, price adjustments, availability, and
preselection flags. Product-based groups and `sub_product_selection` fields are
ignored for the current prototype.

Size variants contain the channel base price plus the size adjustment. A fully
configured item's price additionally includes selected custom choice adjustments.
Group selection rules must be enforced when
quoting that total; a base or size price alone may exclude required selections.
Group and choice names are included in item embedding text without enumerating
every combination. Presentation positions, course routing, and subgroups are
ignored. Validation checks custom group limits, scoped identifiers, and option
prices, and rejects required custom groups with too few options.

### 6.5 Provenance-aware value

Important extracted or enriched fields should carry provenance where practical:

```text
AttributedValue
- value
- origin: source | inferred | manually_confirmed
- confidence: optional
- evidence: optional source text or explanation
```

The initial notebook may implement provenance only for uncertain PDF extraction and inferred semantic attributes rather than wrapping every field.

### 6.6 Source reference

```text
SourceReference
- source_type: api | pdf | manual
- source_id or URL
- external_record_id: optional
- page_number: optional
- original_text: optional
- extraction_confidence: optional
```

### 6.7 Availability

```text
Availability
- status: available | unavailable | unknown
- checked_at: optional
```

Unknown must remain distinct from available.

## 7. Menu source abstraction

Every source adapter produces the same canonical `Menu` model.

```text
MenuSource
- load() -> raw source data
- extract() -> source-specific records
- normalize() -> Menu
```

Initial adapters:

- `ApiMenuSource`: maps structured API responses into the canonical model.
- `PdfMenuSource`: extracts text or OCR output, identifies menu records, and converts them into the canonical model.
- `InMemoryMenuSource`: accepts a small hand-authored fixture for early notebook experiments and tests.

Source adapters own source-specific parsing. Retrieval and recommendation components should never need to know whether a menu originated from an API or PDF.

## 8. Ingestion and normalization

### 8.1 API ingestion

API ingestion should map identifiers, category relationships, items, variants, prices, and availability directly when supplied. It should retain external identifiers so records can be refreshed rather than duplicated.

### 8.2 PDF ingestion

PDF ingestion may require:

1. Text extraction for text-based PDFs.
2. OCR for scanned pages.
3. Layout interpretation to associate item names, descriptions, and prices.
4. LLM-assisted conversion of extracted text into structured records.
5. Schema and price validation.
6. Human review for low-confidence or ambiguous records.

Page number, original text, and extraction confidence should be retained. The first notebook will not attempt to solve every PDF layout; it should use one representative sample and expose uncertainties.

### 8.3 Normalization rules

- Convert prices to a consistent exact numeric representation.
- Preserve the original currency and reject ambiguous currency when necessary.
- Treat item variants as separate purchasable options during search.
- Normalize category names without losing their original labels.
- Deduplicate records using stable source identifiers when available.
- Do not infer availability from a missing availability field.
- Do not infer allergen safety from a missing allergen list.

## 9. LLM enrichment and embedding strategy

### 9.1 Enrichment

Sparse menu records can be enriched with non-safety semantic attributes such as:

- sweet
- spicy
- savory
- light
- filling
- refreshing
- rich
- suitable for sharing
- breakfast-like
- dessert-like
- comfort food

All inferred attributes must be marked as inferred. The LLM may propose likely ingredients for retrieval context, but unverified ingredients must not be used to guarantee allergy or dietary suitability.

### 9.2 Embedding unit

Create one retrievable document per purchasable item or variant. Avoid embedding an entire menu as a single document because it weakens retrieval precision and makes item-level provenance difficult.

An embedding document may contain:

```text
Name: Gulab Jamun
Category: Desserts
Description: Fried milk dumplings served in sugar syrup
Variant: Regular
Price: INR 350
Verified dietary attributes: Vegetarian
Semantic attributes: sweet, rich, warm dessert
```

Price is included as useful context, but numerical budget enforcement must use structured metadata rather than embedding similarity.

### 9.3 Indexes

The initial retrieval layer should support:

- A vector index for semantic similarity.
- Structured metadata for price, currency, category, availability, variants, and verified dietary properties.
- Optional lexical search for exact item names and category terms.

For a small notebook menu, an in-memory index is sufficient. The design must not depend on a particular vector database.

## 10. Customer request model

```text
CustomerRequest
- raw_text
- conversation_context: optional
- locale: optional
- result_limit: optional
```

The LLM converts it into a structured request interpretation:

```text
RequestInterpretation
- request_type: item_recommendation | meal_recommendation | menu_question
- desired_attributes[]
- desired_categories[]
- preferred_ingredients[]
- excluded_ingredients[]
- dietary_constraints[]
- allergen_constraints[]
- minimum_price: optional
- maximum_item_price: optional
- maximum_total_price: optional
- currency: optional
- servings: optional
- desired_meal_components[]
- soft_preferences[]
- ambiguities[]
```

Each interpreted condition should also be classified:

```text
Constraint
- name
- value
- kind: safety | hard | soft
- source_text
```

Examples:

- A declared peanut allergy is a safety constraint.
- "Under 500 rupees" is normally a hard constraint.
- "Not too heavy" is normally a soft preference.

The agent may ask a question when a material ambiguity prevents safe or useful retrieval. Otherwise it should make a reasonable interpretation and state it when helpful.

## 11. Recommendation flow

### 11.1 Intent extraction

The LLM transforms the request into `RequestInterpretation`. Its output is schema-validated before use.

Example:

```json
{
  "request_type": "item_recommendation",
  "desired_attributes": ["sweet"],
  "maximum_item_price": 500,
  "currency": "INR",
  "hard_constraints": ["maximum_item_price"],
  "soft_preferences": []
}
```

### 11.2 Search planning

The request is split into:

- Semantic query: qualities such as sweet, light, or refreshing.
- Structured filters: price, category, availability, and verified dietary fields.
- Combination requirements: number of people, meal components, and total budget.

### 11.3 Hybrid retrieval

Candidate retrieval combines:

1. Semantic vector similarity.
2. Lexical or exact-name matching where useful.
3. Structured metadata filtering.
4. Optional reranking against the original request.

Filtering may occur before or after vector retrieval depending on the selected index, but every final candidate must pass deterministic validation.

### 11.4 Ranking

A candidate score can combine:

```text
semantic relevance
+ category match
+ verified attribute match
+ availability
+ budget fit
- unmet soft preferences
- distance from relaxed constraints
```

The notebook should keep scoring understandable and observable rather than prematurely introducing a complex learned ranker.

### 11.5 Grounded response generation

The LLM receives:

- The original customer request.
- Its validated interpretation.
- A small set of normalized and validated candidates.
- Exact price calculations.
- Any unmet or relaxed constraints.

It returns a concise recommendation with reasons. A final validator must ensure that referenced item IDs, variant IDs, prices, and totals agree with the supplied candidates.

## 12. Exact matches and nearest alternatives

### 12.1 Exact match

An exact candidate satisfies all safety and hard constraints and is relevant to the requested semantic qualities.

For `"something sweet under INR 500"`:

```text
semantic match: sweet
price: <= 500
availability: available, or handled according to the product's unknown policy
```

### 12.2 Controlled fallback

If no exact match exists, create explicit alternative search plans. A starting relaxation order is:

1. Never relax allergen or other safety constraints.
2. Preserve the main requested quality, such as sweet.
3. Relax budget by the smallest absolute amount.
4. Relax adjacent soft preferences one at a time.
5. Search semantically adjacent categories if necessary.

Fallback policies should be configurable rather than hidden in the prompt.

### 12.3 Alternative explanation

Every alternative should record:

```text
AlternativeMatch
- item or variant
- matched_constraints[]
- unmet_constraints[]
- relaxed_constraints[]
- price_difference: optional
- relevance_score
- explanation_facts
```

Example response:

> I couldn't find a sweet item under INR 500. The closest options are Chocolate Cake at INR 650 and Cheesecake at INR 675, which are INR 150 and INR 175 over your budget.

The price differences are calculated by application logic, not the LLM.

## 13. Meal recommendation

An item recommendation returns one or more independent candidates. A meal recommendation composes several compatible items.

```text
MealRecommendation
- components[]
- total_price
- currency
- estimated_servings
- matched_constraints[]
- unmet_constraints[]
- explanation
```

```text
MealComponent
- role: starter | main | side | drink | dessert | other
- item_id
- variant_id: optional
- quantity
- subtotal
```

The LLM may propose coherent combinations from retrieved candidates. A deterministic combination and validation step must calculate quantities, subtotals, and the final price, then reject combinations that violate hard constraints.

Meal composition is planned for a later notebook iteration after single-item retrieval works reliably.

## 14. Search result model

```text
MenuSearchResult
- request
- interpretation
- exact_matches[]
- alternatives[]
- unmatched_constraints[]
- retrieval_metadata
- response_summary
```

```text
ItemMatch
- item_id
- variant_id: optional
- match_type: exact | alternative
- matched_constraints[]
- unmet_constraints[]
- relevance_score
- price_difference: optional
- explanation_facts[]
```

Returning structured results before generating prose makes the system testable and allows a future API or interface to render recommendations differently.

## 15. Safety and trust rules

- Never state that an item is safe for an allergy unless supported by verified menu data and the product's safety policy.
- Never relax allergy constraints to create an alternative.
- Clearly label unknown availability, ingredients, or dietary status.
- Never create an item, price, or variant that is not present in the normalized menu.
- Use deterministic arithmetic for all prices, quantities, differences, and totals.
- Preserve source references so a recommendation can be traced to its menu record.
- Treat LLM-generated semantic tags as suggestions, not source facts.

## 16. Notebook-first implementation roadmap

### Notebook 1: End-to-end item recommendation

Objective: prove the complete recommendation loop using a small in-memory menu.

Planned sections:

1. Define the canonical data models.
2. Create a representative sample menu with categories, items, and variants.
3. Normalize and validate the menu.
4. Generate semantic enrichment and embedding documents.
5. Build an in-memory vector index.
6. Use an LLM to extract structured intent from customer requests.
7. Run hybrid retrieval with metadata constraints.
8. Produce exact results and controlled fallback alternatives.
9. Ask the LLM to generate a grounded recommendation.
10. Validate the final response against retrieved records.
11. Run an evaluation set and inspect failure cases.

### Notebook 2: Source adapters

- Add one representative menu API adapter.
- Add one representative PDF extraction path.
- Normalize both into the same model.
- Compare extraction accuracy and surface uncertain records.

### Notebook 3: Meal composition

- Add serving counts and meal-component roles.
- Retrieve candidates by component.
- Generate candidate combinations.
- Validate totals and hard constraints.
- Use the LLM to rank and explain coherent meals.

### Application phase

When the notebook behavior is reliable:

- Move reusable models and services into a Python package.
- Keep notebooks as experiments and demonstrations.
- Add persistent structured and vector storage if scale requires it.
- Expose a service API.
- Add conversational session state, logging, monitoring, and feedback.

## 17. Evaluation plan

The prototype should include a fixed suite of requests covering:

- Exact price match.
- No item below a requested price.
- Semantic quality such as sweet or refreshing.
- Category request.
- Item variant where only one size meets the budget.
- Conflicting constraints.
- Unknown dietary or allergen information.
- Unavailable item.
- Ambiguous currency or serving count.
- Prompt asking for an item not present on the menu.
- Later: multi-item meal within a total budget.

Useful measurements:

- Constraint extraction accuracy.
- Exact-match precision and recall.
- Rate of invented items or incorrect prices; target is zero.
- Quality and distance of fallback alternatives.
- Explanation faithfulness.
- Retrieval latency and LLM cost.
- Human preference among valid recommendations.

Each evaluation case should specify the expected interpretation, allowed item IDs, forbidden item IDs, and expected fallback behavior.

## 18. Observability

During development, retain a trace containing:

- Original request.
- Parsed intent and constraints.
- Generated search plans.
- Retrieved candidate IDs and scores.
- Filters applied and rejection reasons.
- Constraint relaxation steps.
- Exact price calculations.
- Candidate context sent to the LLM.
- Final structured and natural-language response.

These traces are essential for distinguishing retrieval failures, extraction failures, ranking failures, and response-generation failures.

## 19. Confirmed decisions

- Python is the implementation language.
- The first implementation will be notebook-based.
- The system will use an LLM to understand requests and generate grounded recommendations.
- Menu data will be normalized before recommendation.
- API and PDF inputs will share an abstract source layer.
- Retrieval will be hybrid: embeddings plus structured filtering, with optional lexical matching.
- The vector retrieval unit will normally be one item or purchasable variant.
- The normalized menu, not the LLM, is the source of truth.
- Deterministic code will enforce factual constraints and calculate prices.
- Exact matches and nearest alternatives will be represented separately.
- Safety constraints will not be relaxed.

## 20. Open decisions

These choices should be made close to implementation, after a small experiment where appropriate:

- LLM provider and model.
- Embedding provider and model.
- Local in-memory vector library for the first notebook.
- Typed model library: Pydantic, data classes, or another option.
- Whether menu enrichment happens automatically or requires review.
- Policy for recommending an item whose availability is unknown.
- Exact boundary between a hard constraint and a soft preference.
- Reranking method: heuristics, embedding score, LLM reranker, or a combination.
- PDF extraction tools and the minimum supported PDF quality.
- Conversation memory and how follow-up requests modify earlier constraints.
- Initial language and currency coverage beyond English and INR.

## 21. Suggested first experiment

Use a hand-authored menu of roughly 30–50 varied items and a test set of 20–30 customer requests. This is large enough to expose retrieval and fallback issues while keeping every result easy to inspect manually.

The first success criterion is:

> Given a natural-language request, the notebook returns only real menu records, correctly applies price and other structured constraints, finds semantically relevant items, and clearly identifies the closest alternatives when no exact match exists.

## 22. Decision log

Record future architectural decisions here so changes remain understandable.

| Date | Decision | Reason | Status |
|---|---|---|---|
| 2026-09-27 | Start with a Python notebook prototype | Fast experimentation and visible end-to-end inspection | Accepted |
| 2026-09-27 | Use hybrid RAG rather than vector search alone | Semantic retrieval does not reliably enforce numeric or safety constraints | Accepted |
| 2026-09-27 | Normalize API and PDF inputs into one model | Keeps retrieval and recommendation independent of source format | Accepted |
| 2026-09-27 | Ground LLM output in validated menu candidates | Prevents invented items and incorrect factual claims | Accepted |
