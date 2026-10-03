"""Normalize the ordering API payload into the canonical menu domain model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


class MenuNormalizationError(ValueError):
    """Raised when an API payload cannot be mapped without guessing."""


@dataclass(frozen=True)
class SourceReference:
    source_type: str
    source_id: str
    external_record_id: str | None = None


@dataclass(frozen=True)
class Availability:
    status: str
    checked_at: datetime | None = None


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    description: str | None
    aliases: tuple[str, ...]
    source_reference: SourceReference


@dataclass(frozen=True)
class ItemVariant:
    id: str
    item_id: str
    name: str
    price: Decimal
    availability: Availability
    source_reference: SourceReference


@dataclass(frozen=True)
class VariationChoice:
    id: str
    name: str
    description: str | None
    price_adjustment: Decimal
    availability: Availability
    is_preselected: bool
    source_reference: SourceReference


@dataclass(frozen=True)
class VariationGroup:
    id: str
    name: str
    description: str | None
    type: str
    is_optional: bool
    min_selection: int
    max_selection: int
    availability: Availability
    choices: tuple[VariationChoice, ...]
    source_reference: SourceReference


@dataclass(frozen=True)
class MenuItem:
    id: str
    name: str
    description: str | None
    category_ids: tuple[str, ...]
    base_price: Decimal | None
    currency: str
    variants: tuple[ItemVariant, ...]
    ingredients: tuple[str, ...] | None
    allergens: tuple[str, ...]
    dietary_attributes: tuple[str, ...]
    semantic_attributes: tuple[str, ...]
    availability: Availability
    source_reference: SourceReference
    embedding_text: str
    variation_groups: tuple[VariationGroup, ...] = ()


@dataclass(frozen=True)
class Menu:
    id: str
    restaurant_name: str
    currency: str
    categories: tuple[Category, ...]
    items: tuple[MenuItem, ...]
    source_metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime


def _records(value: Any, field_name: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise MenuNormalizationError(f"{field_name} must be a list of objects")
    return value


def _required_text(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise MenuNormalizationError(f"record has no valid {field_name!r}: {record!r}")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _availability(
    *values: Any,
    checked_at: datetime,
) -> Availability:
    if "UNAVAILABLE" in values:
        return Availability(status="unavailable", checked_at=checked_at)
    if values and all(value == "AVAILABLE" for value in values):
        return Availability(status="available", checked_at=checked_at)
    return Availability(status="unknown", checked_at=checked_at)


def _money(value: Any, context: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MenuNormalizationError(f"invalid price for {context}: {value!r}") from exc
    if not amount.is_finite() or amount < 0:
        raise MenuNormalizationError(f"invalid price for {context}: {value!r}")
    return amount


def _source(endpoint: str, record_id: str) -> SourceReference:
    return SourceReference(
        source_type="api",
        source_id=endpoint,
        external_record_id=record_id,
    )


def _normalize_variation_groups(
    item_record: Mapping[str, Any],
    endpoint: str,
    fetched_at: datetime,
) -> tuple[VariationGroup, ...]:
    groups = []
    for group in _records(
        item_record.get("menu_item_variations") or [], "menu_item_variations"
    ):
        if group.get("type") == "PRODUCT_BASED":
            continue
        group_id = _required_text(group, "id")
        choices = []
        for choice in _records(group.get("choices") or [], "variation choices"):
            choice_id = _required_text(choice, "id")
            choices.append(VariationChoice(
                id=choice_id,
                name=_required_text(choice, "name"),
                description=_optional_text(choice.get("description")),
                price_adjustment=_money(choice.get("price"), f"choice {choice_id!r}"),
                availability=_availability(choice.get("availability"), checked_at=fetched_at),
                is_preselected=choice.get("is_preselected"),
                source_reference=_source(endpoint, choice_id),
            ))
        groups.append(VariationGroup(
            id=group_id,
            name=_required_text(group, "name"),
            description=_optional_text(group.get("description")),
            type=_required_text(group, "type"),
            is_optional=group.get("is_optional"),
            min_selection=group.get("min_selection"),
            max_selection=group.get("max_selection"),
            availability=_availability(group.get("availability"), checked_at=fetched_at),
            choices=tuple(choices),
            source_reference=_source(endpoint, group_id),
        ))
    return tuple(groups)


def normalize_api_menu(
    payload: Any,
    *,
    menu_id: str,
    restaurant_name: str,
    currency: str,
    channel: str,
) -> Menu:
    """Map an ``ApiMenuPayload`` into the canonical model.

    ``channel`` is mandatory because this API can return a different price and
    availability for each ordering channel. Size prices are treated as
    adjustments to the selected channel's base price.
    """
    for field_name, value in (
        ("menu_id", menu_id),
        ("restaurant_name", restaurant_name),
        ("currency", currency),
        ("channel", channel),
    ):
        if not isinstance(value, str) or not value.strip():
            raise MenuNormalizationError(f"{field_name} must not be empty")

    normalized_currency = currency.strip().upper()
    normalized_channel = channel.strip().upper()

    item_records = _records(payload.menu, "menu")
    category_records = _records(payload.categories, "categories")
    category_ids = {_required_text(record, "id") for record in category_records}

    categories: list[Category] = []
    for category_record in category_records:
        category_id = _required_text(category_record, "id")
        category_name = _required_text(category_record, "name")
        categories.append(
            Category(
                id=category_id,
                name=category_name,
                description=_optional_text(category_record.get("description")),
                aliases=(),
                source_reference=_source(payload.categories_endpoint, category_id),
            )
        )

    items: list[MenuItem] = []
    for item_record in item_records:
        item_id = _required_text(item_record, "id")
        item_name = _required_text(item_record, "name")
        item_source = _source(payload.menu_endpoint, item_id)
        variation_groups = _normalize_variation_groups(
            item_record, payload.menu_endpoint, payload.fetched_at
        )

        raw_prices = _records(
            item_record.get("pricing_and_availabilities") or [],
            f"pricing_and_availabilities for item {item_id!r}",
        )
        channel_prices = [
            price for price in raw_prices if price.get("channel") == normalized_channel
        ]
        if len(channel_prices) > 1:
            raise MenuNormalizationError(
                f"item {item_id!r} has duplicate prices for channel {normalized_channel!r}"
            )
        selected_price = channel_prices[0] if channel_prices else None
        base_amount = (
            _money(selected_price.get("price"), f"item {item_id!r}")
            if selected_price is not None
            else None
        )

        raw_sizes = _records(item_record.get("sizes") or [], f"sizes for item {item_id!r}")
        variants: list[ItemVariant] = []
        for size in raw_sizes:
            size_id = _required_text(size, "id")
            adjustment = _money(size.get("price"), f"size {size_id!r}")
            if base_amount is None:
                raise MenuNormalizationError(
                    f"item {item_id!r} has sizes but no price for channel {normalized_channel!r}"
                )
            variants.append(
                ItemVariant(
                    id=size_id,
                    item_id=item_id,
                    name=_required_text(size, "name"),
                    price=base_amount + adjustment,
                    availability=_availability(
                        item_record.get("availability"),
                        selected_price.get("availability"),
                        size.get("availability"),
                        checked_at=payload.fetched_at,
                    ),
                    source_reference=_source(payload.menu_endpoint, size_id),
                )
            )

        linked_categories = tuple(
            category_id
            for category_id in (item_record.get("category_id"),)
            if category_id in category_ids
        )
        raw_allergens = item_record.get("allergens")
        allergens = (
            tuple(str(value).strip().casefold() for value in raw_allergens if str(value).strip())
            if isinstance(raw_allergens, list)
            else ()
        )
        channel_availability = (
            selected_price.get("availability") if selected_price is not None else None
        )
        description = _optional_text(item_record.get("description"))
        category_names = [
            category.name for category in categories if category.id in linked_categories
        ]
        embedding_parts = [item_name, *category_names]
        if description:
            embedding_parts.append(description)
        for group in variation_groups:
            choice_names = [choice.name for choice in group.choices] if group.type == "CUSTOM" else []
            embedding_parts.append(f"{group.name}: {', '.join(dict.fromkeys(choice_names))}")
            if group.description:
                embedding_parts.append(group.description)

        items.append(
            MenuItem(
                id=item_id,
                name=item_name,
                description=description,
                category_ids=linked_categories,
                base_price=None if variants else base_amount,
                currency=normalized_currency,
                variants=tuple(variants),
                ingredients=None,
                allergens=allergens,
                dietary_attributes=(),
                semantic_attributes=(),
                availability=_availability(
                    item_record.get("availability"),
                    channel_availability,
                    checked_at=payload.fetched_at,
                ),
                source_reference=item_source,
                embedding_text="\n".join(embedding_parts),
                variation_groups=variation_groups,
            )
        )

    timestamp = payload.fetched_at
    return Menu(
        id=menu_id.strip(),
        restaurant_name=restaurant_name.strip(),
        currency=normalized_currency,
        categories=tuple(categories),
        items=tuple(items),
        source_metadata={
            "source_type": "api",
            "menu_endpoint": payload.menu_endpoint,
            "categories_endpoint": payload.categories_endpoint,
            "channel": normalized_channel,
        },
        created_at=timestamp,
        updated_at=timestamp,
    )
