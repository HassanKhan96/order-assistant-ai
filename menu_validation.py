"""Validation for normalized menu data before enrichment and embedding."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable

from menu_normalization import Availability, Menu, SourceReference


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    record_type: str
    record_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            summary = "; ".join(issue.message for issue in self.errors[:5])
            remaining = len(self.errors) - 5
            if remaining > 0:
                summary += f"; and {remaining} more error(s)"
            raise MenuValidationError(summary, report=self)


class MenuValidationError(ValueError):
    """Raised when a validation report contains blocking errors."""

    def __init__(self, message: str, *, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


class _IssueCollector:
    def __init__(self) -> None:
        self.errors: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []

    def error(
        self,
        code: str,
        message: str,
        record_type: str,
        record_id: str | None = None,
    ) -> None:
        self.errors.append(ValidationIssue(code, message, record_type, record_id))

    def warning(
        self,
        code: str,
        message: str,
        record_type: str,
        record_id: str | None = None,
    ) -> None:
        self.warnings.append(ValidationIssue(code, message, record_type, record_id))


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_unique_ids(
    records: Iterable[Any],
    record_type: str,
    issues: _IssueCollector,
) -> None:
    seen: set[str] = set()
    for record in records:
        record_id = getattr(record, "id", None)
        if not _is_non_empty_text(record_id):
            issues.error(
                "missing_id",
                f"{record_type} has no ID",
                record_type,
            )
        elif record_id in seen:
            issues.error(
                "duplicate_id",
                f"duplicate {record_type} ID {record_id!r}",
                record_type,
                record_id,
            )
        else:
            seen.add(record_id)


def _validate_availability(
    availability: Any,
    record_type: str,
    record_id: str,
    issues: _IssueCollector,
) -> None:
    if not isinstance(availability, Availability):
        issues.error(
            "invalid_availability",
            f"{record_type} {record_id!r} has an invalid availability value",
            record_type,
            record_id,
        )
        return
    if availability.status not in {"available", "unavailable", "unknown"}:
        issues.error(
            "invalid_availability_status",
            f"{record_type} {record_id!r} has invalid availability status "
            f"{availability.status!r}",
            record_type,
            record_id,
        )
    elif availability.status == "unknown":
        issues.warning(
            "unknown_availability",
            f"{record_type} {record_id!r} has unknown availability",
            record_type,
            record_id,
        )


def _validate_price(
    price: Any,
    record_type: str,
    record_id: str,
    issues: _IssueCollector,
) -> None:
    if not isinstance(price, Decimal) or not price.is_finite() or price < 0:
        issues.error(
            "invalid_price",
            f"{record_type} {record_id!r} has invalid price {price!r}",
            record_type,
            record_id,
        )


def _validate_source(
    source: Any,
    record_type: str,
    record_id: str,
    issues: _IssueCollector,
) -> None:
    if not isinstance(source, SourceReference):
        issues.error(
            "invalid_source_reference",
            f"{record_type} {record_id!r} has no valid source reference",
            record_type,
            record_id,
        )
        return
    if not _is_non_empty_text(source.source_type) or not _is_non_empty_text(source.source_id):
        issues.error(
            "incomplete_source_reference",
            f"{record_type} {record_id!r} has an incomplete source reference",
            record_type,
            record_id,
        )
    if source.external_record_id != record_id:
        issues.error(
            "source_id_mismatch",
            f"{record_type} {record_id!r} does not match its external record ID",
            record_type,
            record_id,
        )


def _validate_variation_groups(item: Any, issues: _IssueCollector) -> None:
    # Groups and choices can be shared across items, so IDs are unique within
    # their owning item/group rather than across the entire menu.
    _validate_unique_ids(item.variation_groups, "variation_group", issues)
    for group in item.variation_groups:
        def error(code: str, message: str) -> None:
            issues.error(code, f"item {item.id!r}, group {group.id!r}: {message}",
                         "variation_group", group.id)

        if not _is_non_empty_text(group.name):
            error("missing_name", "group has no name")
        if group.type != "CUSTOM":
            error("unsupported_variation_type", f"unsupported type {group.type!r}")
        if type(group.is_optional) is not bool:
            error("invalid_optional_flag", "is_optional must be a boolean")
        valid_limits = all(type(value) is int and value >= 0 for value in (
            group.min_selection, group.max_selection
        ))
        if not valid_limits or group.min_selection > group.max_selection:
            error("invalid_selection_limits", "selection limits must be non-negative integers with min <= max")
            valid_limits = False

        _validate_availability(group.availability, "variation_group", group.id, issues)
        _validate_source(group.source_reference, "variation_group", group.id, issues)
        _validate_unique_ids(group.choices, "variation_choice", issues)
        for choice in group.choices:
            if not _is_non_empty_text(choice.name):
                error("missing_choice_name", f"choice {choice.id!r} has no name")
            if type(choice.is_preselected) is not bool:
                error("invalid_preselection_flag", f"choice {choice.id!r} has invalid preselection flag")
            _validate_price(choice.price_adjustment, "variation_choice", choice.id, issues)
            _validate_availability(choice.availability, "variation_choice", choice.id, issues)
            _validate_source(choice.source_reference, "variation_choice", choice.id, issues)
            if (group.type == "CUSTOM" and _is_non_empty_text(choice.name) and isinstance(item.embedding_text, str)
                    and choice.name.casefold() not in item.embedding_text.casefold()):
                error("embedding_missing_choice", f"embedding text omits choice {choice.name!r}")

        option_count = len(group.choices)
        if valid_limits and not group.is_optional and group.min_selection > option_count:
            error("unsatisfiable_selection", f"requires {group.min_selection} selections but only {option_count} options exist")
        if valid_limits:
            preselected = sum(choice.is_preselected is True for choice in group.choices)
            if preselected > group.max_selection:
                error("too_many_preselected_choices", "preselected choices exceed max_selection")


def validate_menu(menu: Menu) -> ValidationReport:
    """Return all blocking errors and non-blocking warnings for ``menu``."""
    issues = _IssueCollector()

    for field_name, value in (
        ("id", menu.id),
        ("restaurant_name", menu.restaurant_name),
        ("currency", menu.currency),
    ):
        if not _is_non_empty_text(value):
            issues.error(
                f"missing_{field_name}",
                f"menu has no {field_name.replace('_', ' ')}",
                "menu",
                menu.id or None,
            )

    if _is_non_empty_text(menu.currency) and (
        len(menu.currency) != 3 or menu.currency != menu.currency.upper()
    ):
        issues.error(
            "invalid_currency",
            f"menu currency {menu.currency!r} must be a three-letter uppercase code",
            "menu",
            menu.id or None,
        )

    _validate_unique_ids(menu.categories, "category", issues)
    _validate_unique_ids(menu.items, "item", issues)
    all_variants = tuple(variant for item in menu.items for variant in item.variants)
    _validate_unique_ids(all_variants, "variant", issues)

    category_ids = {category.id for category in menu.categories}
    for category in menu.categories:
        if not _is_non_empty_text(category.name):
            issues.error(
                "missing_name",
                f"category {category.id!r} has no name",
                "category",
                category.id,
            )
        _validate_source(category.source_reference, "category", category.id, issues)

    for item in menu.items:
        if not _is_non_empty_text(item.name):
            issues.error("missing_name", f"item {item.id!r} has no name", "item", item.id)
        if item.currency != menu.currency:
            issues.error(
                "currency_mismatch",
                f"item {item.id!r} uses {item.currency!r}, not {menu.currency!r}",
                "item",
                item.id,
            )

        unknown_categories = set(item.category_ids) - category_ids
        for category_id in sorted(unknown_categories):
            issues.error(
                "unknown_category",
                f"item {item.id!r} references unknown category {category_id!r}",
                "item",
                item.id,
            )
        if not item.category_ids:
            issues.warning(
                "uncategorized_item",
                f"item {item.id!r} has no category",
                "item",
                item.id,
            )

        if item.base_price is None and not item.variants:
            issues.error(
                "missing_price",
                f"item {item.id!r} has neither a base price nor variants",
                "item",
                item.id,
            )
        elif item.base_price is not None:
            _validate_price(item.base_price, "item", item.id, issues)
        if item.base_price is not None and item.variants:
            issues.error(
                "ambiguous_item_price",
                f"item {item.id!r} has both a base price and priced variants",
                "item",
                item.id,
            )

        if not _is_non_empty_text(item.embedding_text):
            issues.error(
                "missing_embedding_text",
                f"item {item.id!r} has no embedding text",
                "item",
                item.id,
            )
        elif _is_non_empty_text(item.name) and item.name.casefold() not in item.embedding_text.casefold():
            issues.error(
                "embedding_missing_item_name",
                f"embedding text for item {item.id!r} does not contain its name",
                "item",
                item.id,
            )

        if not isinstance(item.allergens, tuple):
            issues.error(
                "invalid_allergens",
                f"item {item.id!r} allergens must be a tuple",
                "item",
                item.id,
            )
        elif any(not _is_non_empty_text(allergen) for allergen in item.allergens):
            issues.error(
                "invalid_allergen",
                f"item {item.id!r} contains an empty allergen",
                "item",
                item.id,
            )

        _validate_availability(item.availability, "item", item.id, issues)
        _validate_source(item.source_reference, "item", item.id, issues)
        _validate_variation_groups(item, issues)

        for variant in item.variants:
            if variant.item_id != item.id:
                issues.error(
                    "variant_item_mismatch",
                    f"variant {variant.id!r} references item {variant.item_id!r}, "
                    f"not {item.id!r}",
                    "variant",
                    variant.id,
                )
            if not _is_non_empty_text(variant.name):
                issues.error(
                    "missing_name",
                    f"variant {variant.id!r} has no name",
                    "variant",
                    variant.id,
                )
            _validate_price(variant.price, "variant", variant.id, issues)
            _validate_availability(variant.availability, "variant", variant.id, issues)
            _validate_source(variant.source_reference, "variant", variant.id, issues)

    return ValidationReport(tuple(issues.errors), tuple(issues.warnings))
