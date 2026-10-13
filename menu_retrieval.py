"""Hybrid candidate retrieval grounded in a persisted canonical menu.

This layer accepts explicit filters; natural-language intent extraction follows
in a later step. Scores express retrieval order, not match confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import math
import re
from typing import Protocol, Sequence

from menu_normalization import Menu, MenuItem
from menu_storage import VectorSearchHit
from menu_validation import validate_menu


class QueryEmbeddingProvider(Protocol):
    model_id: str
    dimensions: int

    def embed_query(self, text: str) -> Sequence[float]: ...


class RetrievalStore(Protocol):
    def load_menu(self, menu_id: str) -> Menu | None: ...
    def search(self, **kwargs) -> tuple[VectorSearchHit, ...]: ...


@dataclass(frozen=True)
class RetrievalFilters:
    max_price: Decimal | None = None
    currency: str | None = None
    category_ids: tuple[str, ...] = ()
    dietary_attributes: tuple[str, ...] = ()
    allow_unknown_availability: bool = False

    def __post_init__(self) -> None:
        if self.max_price is not None and (
            not isinstance(self.max_price, Decimal)
            or not self.max_price.is_finite() or self.max_price < 0
        ):
            raise ValueError("max_price must be a finite, non-negative Decimal")
        if self.currency is not None:
            if not isinstance(self.currency, str) or not self.currency.strip():
                raise ValueError("currency must be a non-empty string")
            object.__setattr__(self, "currency", self.currency.strip().upper())


@dataclass(frozen=True)
class RetrievalCandidate:
    item_id: str
    variant_id: str | None
    name: str
    variant_name: str | None
    currency: str
    minimum_price: Decimal
    # One cheapest valid configuration, not a claim about every configuration.
    required_choices: tuple[tuple[str, tuple[str, ...]], ...]
    availability: str
    score: float = 0.0
    semantic_score: float | None = None
    lexical_score: float | None = None


def _candidates(item: MenuItem, filters: RetrievalFilters) -> tuple[RetrievalCandidate, ...]:
    allowed = {"available", "unknown"} if filters.allow_unknown_availability else {"available"}
    if item.availability.status not in allowed:
        return ()
    if filters.currency and item.currency != filters.currency:
        return ()
    if filters.category_ids and not set(filters.category_ids).intersection(item.category_ids):
        return ()
    if not set(filters.dietary_attributes).issubset(item.dietary_attributes):
        return ()
    extra = Decimal(0)
    selected = []
    statuses = [item.availability.status]
    for group in item.variation_groups:
        if group.is_optional:
            continue
        count = max(1, group.min_selection)
        if group.availability.status not in allowed or count > group.max_selection:
            return ()
        choices = sorted(
            (choice for choice in group.choices if choice.availability.status in allowed),
            key=lambda choice: (choice.price_adjustment, choice.id),
        )[:count]
        if len(choices) != count:
            return ()
        extra += sum((choice.price_adjustment for choice in choices), Decimal(0))
        selected.append((group.id, tuple(choice.id for choice in choices)))
        statuses.extend([group.availability.status, *(choice.availability.status for choice in choices)])
    results = []
    for variant in item.variants or (None,):
        if variant and variant.availability.status not in allowed:
            continue
        price = variant.price if variant else item.base_price
        if price is None:
            continue
        price += extra
        if filters.max_price is not None and price > filters.max_price:
            continue
        unknown = "unknown" in statuses or (variant and variant.availability.status == "unknown")
        results.append(RetrievalCandidate(
            item_id=item.id, variant_id=variant.id if variant else None,
            name=item.name, variant_name=variant.name if variant else None,
            currency=item.currency, minimum_price=price, required_choices=tuple(selected),
            availability="unknown" if unknown else "available",
        ))
    return tuple(results)


_STOP_WORDS = frozenset("a an and at for i in is me of please something the to want with".split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold())) - _STOP_WORDS


def retrieve_candidates(
    query: str, *, menu_id: str, store: RetrievalStore,
    provider: QueryEmbeddingProvider, filters: RetrievalFilters | None = None,
    limit: int = 10, candidate_limit: int = 100,
) -> tuple[RetrievalCandidate, ...]:
    """Fuse vector and lexical ranks after enforcing canonical constraints.

    Lexical matching scans the persisted snapshot, suitable for prototype menus.
    Semantic recall is bounded by candidate_limit and the approximate index.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not 1 <= limit <= candidate_limit <= 100:
        raise ValueError("require 1 <= limit <= candidate_limit <= 100")
    filters = filters or RetrievalFilters()
    menu = store.load_menu(menu_id)
    if menu is None:
        raise ValueError(f"menu {menu_id!r} is not stored")
    validate_menu(menu).raise_for_errors()
    eligible = {}
    for item in menu.items:
        for candidate in _candidates(item, filters):
            eligible[(candidate.item_id, candidate.variant_id)] = candidate
    if not eligible:
        return ()
    vector = provider.embed_query(query.strip())
    if len(vector) != provider.dimensions:
        raise ValueError("query embedding has the wrong dimensions")
    if not vector or not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError("query embedding must be finite and non-zero")
    hits = store.search(
        menu_id=menu_id, model_id=provider.model_id, query_embedding=vector,
        limit=candidate_limit, max_price=filters.max_price,
        category_ids=filters.category_ids, dietary_attributes=filters.dietary_attributes,
        availability=("available", "unknown") if filters.allow_unknown_availability else ("available",),
    )
    semantic = {}
    for hit in hits:
        key = (hit.item_id, hit.variant_id)
        if key in eligible and key not in semantic and math.isfinite(hit.score):
            semantic[key] = hit.score
    terms = _tokens(query)
    items = {item.id: item for item in menu.items}
    categories = {category.id: category.name for category in menu.categories}
    lexical = {}
    for key, candidate in eligible.items():
        item = items[candidate.item_id]
        text = " ".join((item.name, item.embedding_text, candidate.variant_name or "",
                         *(categories[cid] for cid in item.category_ids)))
        overlap = terms.intersection(_tokens(text))
        if overlap:
            exact = query.strip().casefold() == item.name.casefold()
            lexical[key] = len(overlap) / len(terms) + (1.0 if exact else 0.0)
    lexical_order = sorted(lexical, key=lambda key: (-lexical[key], key[0], key[1] or ""))[:candidate_limit]
    scores = {}
    for ranking in (list(semantic), lexical_order):
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key[0], key[1] or ""))[:limit]
    return tuple(replace(eligible[key], score=scores[key],
                         semantic_score=semantic.get(key), lexical_score=lexical.get(key))
                 for key in ordered)
