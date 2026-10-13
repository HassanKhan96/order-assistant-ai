"""PostgreSQL persistence for canonical menus, enrichment cache, and vectors."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from enrichment_cache import EnrichmentCache
from menu_dietary import validate_dietary_results
from menu_normalization import (
    AttributedValue,
    Availability,
    Category,
    ItemVariant,
    Menu,
    MenuItem,
    SourceReference,
    VariationChoice,
    VariationGroup,
)
from menu_validation import validate_menu


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, Decimal)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_value(value: Any) -> Any:
    """Return plain JSON-compatible data without losing money precision."""
    return json.loads(json.dumps(value, default=_json_default))


def _source_values(source: Any) -> tuple[str, str, str | None]:
    return source.source_type, source.source_id, source.external_record_id


def _availability_values(availability: Any) -> tuple[str, datetime | None]:
    return availability.status, availability.checked_at


def _datetime(value: str | datetime | None) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _source(row: Mapping[str, Any]) -> SourceReference:
    return SourceReference(
        source_type=row["source_type"],
        source_id=row["source_id"],
        external_record_id=row.get("external_record_id"),
    )


def _availability(row: Mapping[str, Any]) -> Availability:
    return Availability(status=row["status"], checked_at=_datetime(row.get("checked_at")))


def menu_from_snapshot(snapshot: str | Mapping[str, Any]) -> Menu:
    """Reconstruct the immutable domain model from a stored canonical snapshot."""
    row = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    categories = tuple(
        Category(
            id=value["id"],
            name=value["name"],
            description=value.get("description"),
            aliases=tuple(value["aliases"]),
            source_reference=_source(value["source_reference"]),
        )
        for value in row["categories"]
    )
    items = []
    for value in row["items"]:
        variants = tuple(
            ItemVariant(
                id=variant["id"],
                item_id=variant["item_id"],
                name=variant["name"],
                price=Decimal(variant["price"]),
                availability=_availability(variant["availability"]),
                source_reference=_source(variant["source_reference"]),
            )
            for variant in value["variants"]
        )
        groups = []
        for group in value.get("variation_groups", ()):
            choices = tuple(
                VariationChoice(
                    id=choice["id"],
                    name=choice["name"],
                    description=choice.get("description"),
                    price_adjustment=Decimal(choice["price_adjustment"]),
                    availability=_availability(choice["availability"]),
                    is_preselected=choice["is_preselected"],
                    source_reference=_source(choice["source_reference"]),
                )
                for choice in group["choices"]
            )
            groups.append(
                VariationGroup(
                    id=group["id"],
                    name=group["name"],
                    description=group.get("description"),
                    type=group["type"],
                    is_optional=group["is_optional"],
                    min_selection=group["min_selection"],
                    max_selection=group["max_selection"],
                    availability=_availability(group["availability"]),
                    choices=choices,
                    source_reference=_source(group["source_reference"]),
                )
            )
        inferred = value.get("inferred_description")
        items.append(
            MenuItem(
                id=value["id"],
                name=value["name"],
                description=value.get("description"),
                category_ids=tuple(value["category_ids"]),
                base_price=None
                if value.get("base_price") is None
                else Decimal(value["base_price"]),
                currency=value["currency"],
                variants=variants,
                ingredients=None
                if value.get("ingredients") is None
                else tuple(value["ingredients"]),
                allergens=tuple(value["allergens"]),
                dietary_attributes=tuple(value["dietary_attributes"]),
                semantic_attributes=tuple(
                    AttributedValue(**attribute)
                    for attribute in value["semantic_attributes"]
                ),
                availability=_availability(value["availability"]),
                source_reference=_source(value["source_reference"]),
                embedding_text=value["embedding_text"],
                variation_groups=tuple(groups),
                inferred_description=None if inferred is None else AttributedValue(**inferred),
            )
        )
    menu = Menu(
        id=row["id"],
        restaurant_name=row["restaurant_name"],
        currency=row["currency"],
        categories=categories,
        items=tuple(items),
        source_metadata=row["source_metadata"],
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )
    validate_menu(menu).raise_for_errors()
    return menu


def _vector_literal(values: Sequence[float]) -> str:
    if not values:
        raise ValueError("an embedding must contain at least one value")
    numbers = []
    for value in values:
        number = float(value)
        if not (-float("inf") < number < float("inf")):
            raise ValueError("embedding values must be finite")
        numbers.append(repr(number))
    if not any(float(value) for value in numbers):
        raise ValueError("cosine embeddings must be non-zero")
    return "[" + ",".join(numbers) + "]"


def _indexed_vector_type(dimensions: int) -> tuple[str, str]:
    if not isinstance(dimensions, int) or not 1 <= dimensions <= 4000:
        raise ValueError("indexed embedding dimensions must be between 1 and 4000")
    if dimensions <= 2000:
        return f"vector({dimensions})", "vector_cosine_ops"
    return f"halfvec({dimensions})", "halfvec_cosine_ops"


class ConnectionLike(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any: ...
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


class PostgresMenuStore:
    """Store validated menus and their retrieval data in PostgreSQL/pgvector."""

    def __init__(self, connection: ConnectionLike) -> None:
        self.connection = connection

    @classmethod
    def connect(cls, database_url: str) -> "PostgresMenuStore":
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required; install requirements-storage.txt"
            ) from exc
        return cls(psycopg.connect(database_url, autocommit=True))

    def _executemany(self, query: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.connection.cursor() as cursor:
            cursor.executemany(query, rows)

    def initialize(self) -> None:
        schema = (Path(__file__).parent / "db" / "001_initial.sql").read_text(
            encoding="utf-8"
        )
        with self._transaction():
            self.connection.execute(schema)
        if not hasattr(self.connection, "transaction"):
            self.connection.commit()

    def _transaction(self) -> Any:
        transaction = getattr(self.connection, "transaction", None)
        return transaction() if transaction is not None else nullcontext()

    def replace_menu(self, menu: Menu) -> None:
        """Atomically replace one menu and all dependent structured records."""
        validate_menu(menu).raise_for_errors()
        snapshot = _json_value(asdict(menu))
        with self._transaction():
            self.connection.execute("DELETE FROM menus WHERE id = %s", (menu.id,))
            self.connection.execute(
                """INSERT INTO menus
                   (id, restaurant_name, currency, source_metadata, canonical_snapshot,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)""",
                (
                    menu.id,
                    menu.restaurant_name,
                    menu.currency,
                    json.dumps(_json_value(menu.source_metadata)),
                    json.dumps(snapshot),
                    menu.created_at,
                    menu.updated_at,
                ),
            )
            self._insert_categories(menu)
            self._insert_items(menu)
        if not hasattr(self.connection, "transaction"):
            self.connection.commit()

    def load_menu(self, menu_id: str) -> Menu | None:
        cursor = self.connection.execute(
            "SELECT canonical_snapshot FROM menus WHERE id = %s", (menu_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return menu_from_snapshot(row[0])

    def _insert_categories(self, menu: Menu) -> None:
        self._executemany(
            """INSERT INTO categories
               (menu_id, id, name, description, aliases, source_type, source_id,
                external_record_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (
                    menu.id,
                    category.id,
                    category.name,
                    category.description,
                    list(category.aliases),
                    *_source_values(category.source_reference),
                )
                for category in menu.categories
            ],
        )

    def _insert_items(self, menu: Menu) -> None:
        self._executemany(
            """INSERT INTO menu_items
               (menu_id, id, name, description, base_price, currency, ingredients,
                allergens, dietary_attributes, availability_status,
                availability_checked_at, source_type, source_id, external_record_id,
                embedding_text, inferred_description)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s::jsonb)""",
            [
                (
                    menu.id,
                    item.id,
                    item.name,
                    item.description,
                    item.base_price,
                    item.currency,
                    None if item.ingredients is None else list(item.ingredients),
                    list(item.allergens),
                    list(item.dietary_attributes),
                    *_availability_values(item.availability),
                    *_source_values(item.source_reference),
                    item.embedding_text,
                    None
                    if item.inferred_description is None
                    else json.dumps(_json_value(asdict(item.inferred_description))),
                )
                for item in menu.items
            ],
        )
        self._executemany(
            """INSERT INTO item_categories (menu_id, item_id, category_id)
               VALUES (%s, %s, %s)""",
            [
                (menu.id, item.id, category_id)
                for item in menu.items
                for category_id in item.category_ids
            ],
        )
        self._executemany(
            """INSERT INTO item_variants
               (menu_id, item_id, id, name, price, availability_status,
                availability_checked_at, source_type, source_id, external_record_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (
                    menu.id,
                    item.id,
                    variant.id,
                    variant.name,
                    variant.price,
                    *_availability_values(variant.availability),
                    *_source_values(variant.source_reference),
                )
                for item in menu.items
                for variant in item.variants
            ],
        )
        self._executemany(
            """INSERT INTO variation_groups
               (menu_id, item_id, id, name, description, type, is_optional,
                min_selection, max_selection, availability_status,
                availability_checked_at, source_type, source_id, external_record_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (
                    menu.id,
                    item.id,
                    group.id,
                    group.name,
                    group.description,
                    group.type,
                    group.is_optional,
                    group.min_selection,
                    group.max_selection,
                    *_availability_values(group.availability),
                    *_source_values(group.source_reference),
                )
                for item in menu.items
                for group in item.variation_groups
            ],
        )
        self._executemany(
            """INSERT INTO variation_choices
               (menu_id, item_id, group_id, id, name, description, price_adjustment,
                availability_status, availability_checked_at, is_preselected,
                source_type, source_id, external_record_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (
                    menu.id,
                    item.id,
                    group.id,
                    choice.id,
                    choice.name,
                    choice.description,
                    choice.price_adjustment,
                    *_availability_values(choice.availability),
                    choice.is_preselected,
                    *_source_values(choice.source_reference),
                )
                for item in menu.items
                for group in item.variation_groups
                for choice in group.choices
            ],
        )
        self._executemany(
            """INSERT INTO semantic_attributes
               (menu_id, item_id, position, value, origin, confidence, evidence)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [
                (
                    menu.id,
                    item.id,
                    position,
                    attribute.value,
                    attribute.origin,
                    attribute.confidence,
                    attribute.evidence,
                )
                for item in menu.items
                for position, attribute in enumerate(item.semantic_attributes)
            ],
        )

    def save_enrichment_cache(self, cache: EnrichmentCache) -> None:
        rows = [
            (cache.menu_id, item_id, kind, json.dumps(_json_value(payload)))
            for kind, entries in (
                ("description", cache.descriptions),
                ("tag", cache.tags),
            )
            for item_id, payload in entries.items()
        ]
        with self._transaction():
            self.connection.execute(
                "DELETE FROM enrichment_cache_entries WHERE menu_id = %s",
                (cache.menu_id,),
            )
            self._executemany(
                """INSERT INTO enrichment_cache_entries
                   (menu_id, item_id, kind, payload) VALUES (%s, %s, %s, %s::jsonb)""",
                rows,
            )
        if not hasattr(self.connection, "transaction"):
            self.connection.commit()

    def load_enrichment_cache(self, menu_id: str) -> EnrichmentCache:
        cursor = self.connection.execute(
            """SELECT item_id, kind, payload
               FROM enrichment_cache_entries WHERE menu_id = %s""",
            (menu_id,),
        )
        cache = EnrichmentCache(menu_id)
        for item_id, kind, payload in cursor.fetchall():
            value = json.loads(payload) if isinstance(payload, str) else payload
            target = cache.descriptions if kind == "description" else cache.tags
            target[item_id] = value
        return cache

    def save_dietary_cache(self, menu_id: str, results: Mapping[str, Any]) -> None:
        """Upsert inferred hints, retaining unrelated cached records."""
        validate_dietary_results(results)
        with self._transaction():
            self._executemany(
                """INSERT INTO dietary_inference_cache (menu_id, item_id, payload)
                   VALUES (%s, %s, %s::jsonb)
                   ON CONFLICT (menu_id, item_id) DO UPDATE
                   SET payload = EXCLUDED.payload, updated_at = now()""",
                [(menu_id, item_id, json.dumps(row)) for item_id, row in results.items()],
            )
        if not hasattr(self.connection, "transaction"):
            self.connection.commit()

    def load_dietary_cache(self, menu_id: str) -> dict[str, dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT item_id, payload FROM dietary_inference_cache WHERE menu_id = %s",
            (menu_id,),
        )
        return {item_id: json.loads(row) if isinstance(row, str) else row
                for item_id, row in cursor.fetchall()}

    def existing_document_hashes(self, menu_id: str, model_id: str) -> dict[tuple[str, str], str]:
        cursor = self.connection.execute(
            """SELECT item_id, variant_id, document_hash FROM embedding_documents
               WHERE menu_id = %s AND embedding_model = %s""",
            (menu_id, model_id),
        )
        return {(item_id, variant_id): digest for item_id, variant_id, digest in cursor.fetchall()}

    def replace_embeddings(
        self,
        menu_id: str,
        model_id: str,
        documents: Sequence["EmbeddedDocument"],
    ) -> None:
        for document in documents:
            _vector_literal(document.embedding)
        with self._transaction():
            self.connection.execute(
                """DELETE FROM embedding_documents
                   WHERE menu_id = %s AND embedding_model = %s""",
                (menu_id, model_id),
            )
            self._executemany(
                """INSERT INTO embedding_documents
                   (menu_id, item_id, variant_id, embedding_model,
                    embedding_dimensions, document_text, document_hash, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)""",
                [
                    (
                        menu_id,
                        document.item_id,
                        document.variant_id or "",
                        model_id,
                        len(document.embedding),
                        document.text,
                        document.document_hash,
                        json.dumps(_json_value(document.metadata)),
                        _vector_literal(document.embedding),
                    )
                    for document in documents
                ],
            )
        if not hasattr(self.connection, "transaction"):
            self.connection.commit()

    def ensure_vector_index(self, dimensions: int) -> None:
        """Create an HNSW cosine index for one embedding dimension."""
        vector_type, operator_class = _indexed_vector_type(dimensions)
        with self._transaction():
            self.connection.execute(
                f"""CREATE INDEX IF NOT EXISTS embedding_documents_hnsw_{dimensions}
                    ON embedding_documents USING hnsw
                    ((embedding::{vector_type}) {operator_class})
                    WHERE embedding_dimensions = {dimensions}"""
            )
        if not hasattr(self.connection, "transaction"):
            self.connection.commit()

    def search(
        self,
        *,
        menu_id: str,
        model_id: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        max_price: Decimal | None = None,
        category_ids: Sequence[str] = (),
        dietary_attributes: Sequence[str] = (),
        availability: Sequence[str] = ("available",),
    ) -> tuple["VectorSearchHit", ...]:
        """Run cosine retrieval with deterministic structured constraints."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        dimensions = len(query_embedding)
        vector_type, _ = _indexed_vector_type(dimensions)
        vector = _vector_literal(query_embedding)
        clauses = [
            "e.menu_id = %s",
            "e.embedding_model = %s",
            f"e.embedding_dimensions = {dimensions}",
        ]
        params: list[Any] = [vector, menu_id, model_id]
        if availability:
            clauses.append(
                "i.availability_status = ANY(%s) AND "
                "CASE WHEN e.variant_id = '' THEN i.availability_status "
                "ELSE v.availability_status END = ANY(%s)"
            )
            params.extend([list(availability), list(availability)])
        if max_price is not None:
            clauses.append("COALESCE(v.price, i.base_price) <= %s")
            params.append(max_price)
        if category_ids:
            clauses.append(
                "EXISTS (SELECT 1 FROM item_categories ic "
                "WHERE ic.menu_id = e.menu_id AND ic.item_id = e.item_id "
                "AND ic.category_id = ANY(%s))"
            )
            params.append(list(category_ids))
        if dietary_attributes:
            clauses.append("i.dietary_attributes @> %s")
            params.append(list(dietary_attributes))
        params.append(limit)
        cursor = self.connection.execute(
            f"""SELECT e.item_id, NULLIF(e.variant_id, ''), e.document_text,
                       e.metadata, 1 - (e.embedding::{vector_type} <=>
                       %s::{vector_type}) AS score
                FROM embedding_documents e
                JOIN menu_items i ON i.menu_id = e.menu_id AND i.id = e.item_id
                LEFT JOIN item_variants v ON v.menu_id = e.menu_id
                    AND v.item_id = e.item_id AND v.id = NULLIF(e.variant_id, '')
                WHERE {' AND '.join(clauses)}
                ORDER BY e.embedding::{vector_type} <=> %s::{vector_type}
                LIMIT %s""",
            (*params[:-1], vector, params[-1]),
        )
        hits = []
        for item_id, variant_id, text, metadata, score in cursor.fetchall():
            hits.append(
                VectorSearchHit(
                    item_id=item_id,
                    variant_id=variant_id,
                    text=text,
                    metadata=json.loads(metadata) if isinstance(metadata, str) else metadata,
                    score=float(score),
                )
            )
        return tuple(hits)


class EmbeddingProvider(Protocol):
    model_id: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class VectorSearchHit:
    item_id: str
    variant_id: str | None
    text: str
    metadata: Mapping[str, Any]
    score: float


class EmbeddedDocument:
    def __init__(
        self,
        *,
        item_id: str,
        variant_id: str | None,
        text: str,
        metadata: Mapping[str, Any],
        embedding: Sequence[float],
    ) -> None:
        self.item_id = item_id
        self.variant_id = variant_id
        self.text = text
        self.metadata = dict(metadata)
        self.embedding = tuple(float(value) for value in embedding)
        _vector_literal(self.embedding)
        self.document_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_embedding_documents(
    menu: Menu,
) -> tuple[tuple[str, str | None, str, dict[str, Any]], ...]:
    """Build one retrieval document for each purchasable item or variant."""
    category_names = {category.id: category.name for category in menu.categories}
    documents = []
    for item in menu.items:
        categories = [
            category_names[value]
            for value in item.category_ids
            if value in category_names
        ]
        variants = item.variants or (None,)
        for variant in variants:
            price = variant.price if variant is not None else item.base_price
            availability = (
                variant.availability.status
                if variant is not None
                else item.availability.status
            )
            extra = []
            if variant is not None:
                extra.append(f"Variant: {variant.name}")
            if price is not None:
                extra.append(f"Price: {item.currency} {price}")
            text = "\n".join((item.embedding_text, *extra))
            metadata = {
                "menu_id": menu.id,
                "item_id": item.id,
                "variant_id": None if variant is None else variant.id,
                "currency": item.currency,
                "price": None if price is None else str(price),
                "availability": availability,
                "category_ids": list(item.category_ids),
                "categories": categories,
                "dietary_attributes": list(item.dietary_attributes),
            }
            documents.append(
                (item.id, None if variant is None else variant.id, text, metadata)
            )
    return tuple(documents)


def index_menu(
    menu: Menu,
    provider: EmbeddingProvider,
    store: PostgresMenuStore,
) -> int:
    """Embed the menu and atomically replace this model's vector documents."""
    validate_menu(menu).raise_for_errors()
    _indexed_vector_type(provider.dimensions)
    raw_documents = build_embedding_documents(menu)
    texts = [text for _, _, text, _ in raw_documents]
    vectors = provider.embed_documents(texts) if texts else ()
    if len(vectors) != len(raw_documents):
        raise ValueError("embedding provider returned the wrong number of vectors")
    embedded = []
    for (item_id, variant_id, text, metadata), vector in zip(raw_documents, vectors):
        if len(vector) != provider.dimensions:
            raise ValueError(
                f"embedding provider returned {len(vector)} dimensions; "
                f"expected {provider.dimensions}"
            )
        _vector_literal(vector)
        embedded.append(
            EmbeddedDocument(
                item_id=item_id,
                variant_id=variant_id,
                text=text,
                metadata=metadata,
                embedding=vector,
            )
        )
    store.replace_embeddings(menu.id, provider.model_id, embedded)
    store.ensure_vector_index(provider.dimensions)
    return len(embedded)
