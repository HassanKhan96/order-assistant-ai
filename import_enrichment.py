"""Import saved model results and materialize an enriched menu without inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from enrichment_cache import EnrichmentCache
from menu_descriptions import load_cached_descriptions
from menu_dietary import dietary_input_hash, validate_dietary_results
from menu_enrichment import apply_enrichment, load_cached_tags
from menu_normalization import Menu
from menu_storage import PostgresMenuStore


@dataclass(frozen=True)
class PreparedImport:
    menu: Menu
    cache: EnrichmentCache
    dietary_results: dict
    report: dict[str, int]


def prepare_import(
    menu: Menu, enrichment_path: str | Path, dietary_path: str | Path | None = None,
) -> PreparedImport:
    """Preserve saved caches, applying only current entries without regenerating."""
    path = Path(enrichment_path)
    if not path.is_file():
        raise ValueError(f"enrichment file does not exist: {path}")
    cache = EnrichmentCache.load(path, menu.id)
    # Removed/stale records remain in the cache as historical work, but are not
    # applied to the current canonical menu.
    descriptions = load_cached_descriptions(menu, cache)
    tags = load_cached_tags(menu, descriptions, cache)
    enriched = apply_enrichment(menu, descriptions, tags)
    results = {}
    if dietary_path is not None:
        payload = json.loads(Path(dietary_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("menu_id") != menu.id:
            raise ValueError("dietary cache belongs to a different menu")
        results = payload.get("results")
        validate_dietary_results(results)
    current_dietary = sum(results.get(item.id, {}).get("input_hash") == dietary_input_hash(item, enriched)
                          for item in enriched.items)
    report = {
        "items": len(menu.items),
        "cached_descriptions": len(cache.descriptions),
        "cached_semantic_scores": len(cache.tags),
        "cached_dietary_hints": len(results),
        "applied_descriptions": len(descriptions),
        "applied_semantic_scores": len(tags),
        "current_dietary_hints": current_dietary,
    }
    return PreparedImport(enriched, cache, results, report)


def import_saved_enrichment(prepared: PreparedImport, store: PostgresMenuStore) -> dict[str, int]:
    """Atomically import validated results; repeat imports preserve existing vectors.

    An existing different canonical menu is rejected instead of silently replacing
    a newer snapshot and its embeddings. Local rows replace matching cache rows;
    unrelated database cache entries are retained.
    """
    with store.connection.transaction():
        existing = store.load_menu(prepared.menu.id)
        if existing is not None and existing != prepared.menu:
            raise ValueError("a different canonical menu is already stored; import requires review")
        cache = store.load_enrichment_cache(prepared.menu.id)
        cache.descriptions.update(prepared.cache.descriptions)
        cache.tags.update(prepared.cache.tags)
        if existing is None:
            store.replace_menu(prepared.menu)
        store.save_enrichment_cache(cache)
        store.save_dietary_cache(prepared.menu.id, prepared.dietary_results)
    return dict(prepared.report)
