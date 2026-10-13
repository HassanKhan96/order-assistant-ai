"""Generate guarded, provenance-ready descriptions for sparse menu items."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from ai_adapters import AIAdapter, AIError
from enrichment_cache import EnrichmentCache, content_hash
from menu_normalization import Menu, MenuItem
from menu_validation import validate_menu

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Keep the core module importable when optional notebook deps are absent."""
        return False


PROMPT_VERSION = "v2"
MAX_DESCRIPTION_CHARS = 200
DESCRIPTION_GENERATION_ATTEMPTS = 3
CURATED_DESCRIPTION_MODEL = "codex-curated-v1"
COMPATIBLE_PROMPT_VERSIONS = frozenset({"v1", PROMPT_VERSION})
PLACEHOLDER_PATTERNS = (
    re.compile(r"\btest\b.*\b(?:description|dummy)\b", re.IGNORECASE),
    re.compile(r"\bdummy\b.*\bdescription\b", re.IGNORECASE),
    re.compile(r"\blorem ipsum\b", re.IGNORECASE),
    re.compile(r"^(?:n/?a|tbc|tbd|-|\.)$", re.IGNORECASE),
)
CATEGORY_PLACEHOLDER_PATTERN = re.compile(r"^test\b", re.IGNORECASE)
FORBIDDEN_TERMS = (
    "vegan",
    "vegetarian",
    "halal",
    "kosher",
    "gluten",
    "nut",
    "nuts",
    "peanut",
    "peanuts",
    "dairy-free",
    "lactose",
    "allergen",
    "allergens",
    "allergy",
    "allergies",
    "healthy",
    "low-calorie",
)
_FORBIDDEN_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(term) for term in FORBIDDEN_TERMS) + r")(?!\w)",
    re.IGNORECASE,
)


class DescriptionGenerationError(RuntimeError):
    """Raised when a description model request cannot be completed safely."""


@dataclass(frozen=True)
class DescriptionRequest:
    item_id: str
    name: str
    category_names: tuple[str, ...]
    option_names: tuple[str, ...]


class DescriptionGenerator(Protocol):
    model_id: str

    def generate(
        self, requests: Sequence[DescriptionRequest]
    ) -> Mapping[str, str | None]: ...


@dataclass(frozen=True)
class DescriptionResult:
    item_id: str
    status: str
    text: str | None
    reason: str | None
    model_id: str
    prompt_version: str
    input_hash: str


def needs_description(item: MenuItem) -> bool:
    if item.description is None or not item.description.strip():
        return True
    return any(pattern.search(item.description.strip()) for pattern in PLACEHOLDER_PATTERNS)


def check_description(text: str) -> str | None:
    """Return a rejection reason, or ``None`` when generated text is usable."""
    if not isinstance(text, str) or not text.strip():
        return "description is empty"
    if len(text.strip()) > MAX_DESCRIPTION_CHARS:
        return f"description is longer than {MAX_DESCRIPTION_CHARS} characters"
    match = _FORBIDDEN_PATTERN.search(text)
    if match:
        return f"description contains forbidden term {match.group(0)!r}"
    return None


def clean_generated_description(text: str) -> str | None:
    """Return safe, compact LLM text, or ``None`` when no usable text remains."""
    if not isinstance(text, str):
        return None
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return None

    # Dietary, allergen, and health claims are not reliable model inferences.
    clean = _FORBIDDEN_PATTERN.sub("", clean)
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"([(-])\s+", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip(" ,;:-")
    if not clean:
        return None

    if len(clean) > MAX_DESCRIPTION_CHARS:
        clipped = clean[: MAX_DESCRIPTION_CHARS + 1]
        if (
            len(clipped) > MAX_DESCRIPTION_CHARS
            and not clipped[MAX_DESCRIPTION_CHARS].isspace()
        ):
            clipped = clipped[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0]
        else:
            clipped = clipped[:MAX_DESCRIPTION_CHARS]
        clean = clipped.rstrip(" ,;:-")
        if clean and clean[-1] not in ".!?":
            clean = clean[: MAX_DESCRIPTION_CHARS - 1].rstrip(" ,;:-") + "."

    return clean if check_description(clean) is None else None


def _description_request(item: MenuItem, menu: Menu) -> DescriptionRequest:
    category_by_id = {category.id: category.name for category in menu.categories}
    categories = tuple(
        name
        for category_id in item.category_ids
        if (name := category_by_id.get(category_id))
        and not CATEGORY_PLACEHOLDER_PATTERN.search(name.strip())
    )
    options = tuple(
        f"{group.name}: {', '.join(dict.fromkeys(choice.name for choice in group.choices))}"
        for group in item.variation_groups
        if group.name.strip() and group.choices
    )
    return DescriptionRequest(item.id, item.name, categories, options)


def _request_hash(request: DescriptionRequest) -> str:
    return content_hash(json.dumps(asdict(request), sort_keys=True, separators=(",", ":")))


def _result_from_row(item_id: str, row: Mapping[str, Any]) -> DescriptionResult:
    return DescriptionResult(
        item_id=item_id,
        status=str(row["status"]),
        text=row.get("text"),
        reason=row.get("reason"),
        model_id=str(row["model_id"]),
        prompt_version=str(row["prompt_version"]),
        input_hash=str(row["input_hash"]),
    )


def _row_from_result(result: DescriptionResult) -> dict[str, Any]:
    row = asdict(result)
    row.pop("item_id")
    return row


def load_cached_descriptions(
    menu: Menu,
    cache: EnrichmentCache,
    *,
    require_all: bool = False,
) -> tuple[DescriptionResult, ...]:
    """Load current, valid descriptions from an enrichment cache without Gemini."""
    if cache.menu_id != menu.id:
        raise ValueError(f"cache belongs to menu {cache.menu_id!r}, not {menu.id!r}")
    validate_menu(menu).raise_for_errors()

    results: list[DescriptionResult] = []
    missing: list[str] = []
    for item in menu.items:
        if not needs_description(item):
            continue
        request = _description_request(item, menu)
        row = cache.descriptions.get(item.id)
        if (
            isinstance(row, Mapping)
            and row.get("input_hash") == _request_hash(request)
            and row.get("status") == "described"
            and check_description(row.get("text")) is None
        ):
            results.append(_result_from_row(item.id, row))
        else:
            missing.append(item.id)

    if require_all and missing:
        preview = ", ".join(repr(item_id) for item_id in missing[:5])
        remainder = len(missing) - 5
        suffix = f" and {remainder} more" if remainder > 0 else ""
        raise ValueError(
            f"enrichment cache has no current description for {preview}{suffix}"
        )
    return tuple(results)


def apply_description_overrides(
    menu: Menu,
    cache: EnrichmentCache,
    overrides: Mapping[str, str],
    *,
    cache_path: str | os.PathLike[str] | None = None,
    cache_saver: Callable[[EnrichmentCache], None] | None = None,
) -> int:
    """Validate and cache curated descriptions for matching menu items."""
    if cache.menu_id != menu.id:
        raise ValueError(f"cache belongs to menu {cache.menu_id!r}, not {menu.id!r}")
    requests = {
        item.id: _description_request(item, menu)
        for item in menu.items
        if needs_description(item)
    }
    applied = 0
    for item_id, text in overrides.items():
        request = requests.get(item_id)
        if request is None:
            continue
        clean_text = clean_generated_description(text)
        if clean_text is None:
            raise ValueError(f"invalid curated description for item {item_id!r}")
        result = DescriptionResult(
            item_id=item_id,
            status="described",
            text=clean_text,
            reason=None,
            model_id=CURATED_DESCRIPTION_MODEL,
            prompt_version="curated-v1",
            input_hash=_request_hash(request),
        )
        if cache.descriptions.get(item_id) != _row_from_result(result):
            cache.descriptions[item_id] = _row_from_result(result)
            applied += 1
    if cache_path is not None and applied:
        cache.save(cache_path)
    if cache_saver is not None and applied:
        cache_saver(cache)
    return applied


class AIDescriptionGenerator:
    """Menu-specific prompt and parsing over any AIAdapter implementation."""

    def __init__(self, adapter: "AIAdapter") -> None:
        self.adapter = adapter
        self.model_id = adapter.model_id
        self.cache_model_ids = adapter.cache_model_ids

    @staticmethod
    def _prompt(requests: Sequence[DescriptionRequest]) -> str:
        records = [asdict(request) for request in requests]
        return (
            "Write exactly one concise description for every menu item. Use one sentence and "
            "at most 160 characters, describing only its general style, taste, texture, or role. "
            "For vague names, write a cautious generic description based only on the supplied "
            "name and category; never return unknown and never omit an item. Do not mention "
            "allergens, dietary suitability, health claims, prices, portions, or availability. "
            "Phrase ingredients as typical, never as facts about this restaurant. Return one "
            "row per item with item_id, status 'described', and a non-empty description. "
            f"Items: {json.dumps(records, ensure_ascii=False)}"
        )

    def generate(self, requests: Sequence[DescriptionRequest]) -> Mapping[str, str | None]:
        if not requests:
            return {}
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {"items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "item_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["described"]},
                    "description": {"type": "string"},
                },
                "required": ["item_id", "status", "description"],
            }}},
            "required": ["items"],
        }
        try:
            value = self.adapter.generate_json(self._prompt(requests), schema)
        except AIError as exc:
            raise DescriptionGenerationError(str(exc)) from exc
        requested_ids = {request.item_id for request in requests}
        output = {}
        for row in value["items"]:
            item_id = row["item_id"]
            if item_id in requested_ids:
                if item_id in output:
                    raise DescriptionGenerationError("duplicate description item ID")
                output[item_id] = row["description"].strip()
        return output


class GeminiDescriptionGenerator(AIDescriptionGenerator):
    """Compatibility constructor; new code should inject create_ai_adapter()."""

    def __init__(self, *, model: str = "gemini-2.5-flash", requests_per_minute: float = 10,
                 client: Any = None, api_key: str | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        from ai_adapters.gemini import GeminiAdapter
        if client is None:
            load_dotenv(override=True)
        try:
            adapter = GeminiAdapter(text_model=model, client=client, api_key=api_key,
                                    requests_per_minute=requests_per_minute, sleep=sleep, clock=clock)
        except AIError as exc:
            raise DescriptionGenerationError(str(exc)) from exc
        super().__init__(adapter)


def generate_descriptions(
    menu: Menu,
    generator: DescriptionGenerator,
    cache: EnrichmentCache,
    *,
    cache_path: str | os.PathLike[str] | None = None,
    cache_saver: Callable[[EnrichmentCache], None] | None = None,
    batch_size: int = 25,
) -> tuple[DescriptionResult, ...]:
    """Generate and cache descriptions, returning results in menu order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cache.menu_id != menu.id:
        raise ValueError(f"cache belongs to menu {cache.menu_id!r}, not {menu.id!r}")
    validate_menu(menu).raise_for_errors()

    requests = [_description_request(item, menu) for item in menu.items if needs_description(item)]
    results: dict[str, DescriptionResult] = {}
    pending: list[DescriptionRequest] = []
    for request in requests:
        digest = _request_hash(request)
        row = cache.descriptions.get(request.item_id)
        if (
            isinstance(row, Mapping)
            and row.get("input_hash") == digest
            and row.get("status") == "described"
            and check_description(row.get("text")) is None
            and (
                row.get("model_id") == CURATED_DESCRIPTION_MODEL
                or (
                    row.get("model_id") in getattr(generator, "cache_model_ids", (generator.model_id,))
                    and row.get("prompt_version") in COMPATIBLE_PROMPT_VERSIONS
                )
            )
        ):
            results[request.item_id] = _result_from_row(request.item_id, row)
        else:
            pending.append(request)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        generated = generator.generate(batch)
        for request in batch:
            text = generated.get(request.item_id)
            for _ in range(DESCRIPTION_GENERATION_ATTEMPTS - 1):
                if isinstance(text, str) and check_description(text) is None:
                    break
                text = generator.generate((request,)).get(request.item_id)
            clean_text = clean_generated_description(text)
            if clean_text is None:
                raise DescriptionGenerationError(
                    "model did not return a usable description after "
                    f"{DESCRIPTION_GENERATION_ATTEMPTS} attempts for item {request.item_id!r}"
                )
            result = DescriptionResult(
                item_id=request.item_id,
                status="described",
                text=clean_text,
                reason=None,
                model_id=generator.model_id,
                prompt_version=PROMPT_VERSION,
                input_hash=_request_hash(request),
            )
            results[request.item_id] = result
            cache.descriptions[request.item_id] = _row_from_result(result)
        if cache_path is not None:
            cache.save(cache_path)
        if cache_saver is not None:
            cache_saver(cache)

    return tuple(results[request.item_id] for request in requests)
