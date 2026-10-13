"""Validation and input fingerprints for unverified dietary discovery hints."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ai_adapters import AIAdapter, AIResponseError

from enrichment_cache import content_hash
from menu_descriptions import needs_description
from menu_normalization import Menu, MenuItem


def infer_dietary_batch(adapter: AIAdapter, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Request unverified hints through the same interface for either provider."""
    if not records:
        return []
    labels = {"type": "string", "enum": ["likely", "unlikely", "unknown"]}
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"item_id": {"type": "string"}, "vegetarian": labels,
                           "vegan": labels, "reason": {"type": "string"}},
            "required": ["item_id", "vegetarian", "vegan", "reason"],
        }}},
        "required": ["items"],
    }
    prompt = (
        "Classify every menu item as likely, unlikely, or unknown for vegetarian and "
        "vegan suitability. This is an unverified discovery hint, not a safety claim. "
        "Use unknown when text is insufficient or configurable choices materially change "
        "the answer. Do not assess allergens, gluten-free status, halal, kosher, health, "
        "or cross-contamination. Give one short reason based only on the supplied text. "
        "Return exactly one row per item. Items: " + json.dumps(list(records), ensure_ascii=False)
    )
    rows = adapter.generate_json(prompt, schema)["items"]
    requested = {record["item_id"] for record in records}
    returned = [row["item_id"] for row in rows]
    if len(returned) != len(records) or len(set(returned)) != len(returned) or set(returned) != requested:
        raise AIResponseError("dietary response must contain each requested item exactly once")
    if any(not row["reason"].strip() for row in rows):
        raise AIResponseError("dietary responses must include a reason")
    return rows


def dietary_input_hash(item: MenuItem, menu: Menu) -> str:
    categories = {category.id: category.name for category in menu.categories}
    description = item.description
    if needs_description(item):
        description = item.inferred_description.value if item.inferred_description else None
    record = {
        "item_id": item.id,
        "name": item.name,
        "categories": [categories[cid] for cid in item.category_ids],
        "description": description,
        "choices": [f"{group.name}: {', '.join(choice.name for choice in group.choices)}"
                    for group in item.variation_groups],
    }
    return content_hash(json.dumps(record, sort_keys=True, ensure_ascii=False))


def validate_dietary_results(results: Mapping[str, Any]) -> None:
    if not isinstance(results, Mapping):
        raise ValueError("dietary results must be an object")
    for item_id, row in results.items():
        if not isinstance(item_id, str) or not item_id or not isinstance(row, Mapping):
            raise ValueError("dietary results require item IDs and object values")
        if row.get("origin") != "inferred":
            raise ValueError(f"dietary hint for {item_id} must remain inferred")
        for field in ("vegetarian", "vegan"):
            if row.get(field) not in ("likely", "unlikely", "unknown"):
                raise ValueError(f"invalid {field} label for {item_id}")
        for field in ("reason", "model_id", "prompt_version"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"dietary hint for {item_id} needs {field}")
        if not isinstance(row.get("input_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["input_hash"]):
            raise ValueError(f"invalid dietary input hash for {item_id}")
