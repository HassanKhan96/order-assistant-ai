"""Score and apply non-safety semantic attributes to a normalized menu."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from enrichment_cache import EnrichmentCache, content_hash
from menu_descriptions import (
    CATEGORY_PLACEHOLDER_PATTERN,
    FORBIDDEN_TERMS,
    DescriptionResult,
    needs_description,
)
from menu_normalization import AttributedValue, Menu, MenuItem
from menu_validation import validate_menu


VOCABULARY_VERSION = "v1"
TAG_QUESTIONS: Mapping[str, str] = {
    "sweet": "Is this dish or drink characteristically sweet?",
    "spicy": "Is this dish or drink characteristically spicy or hot with chilli?",
    "savory": "Is this dish characteristically savory?",
    "light": "Is this generally experienced as a light dish or drink?",
    "filling": "Is this generally filling or substantial?",
    "refreshing": "Is this generally refreshing?",
    "rich": "Is this generally rich in taste or texture?",
    "shareable": "Is this commonly suitable for sharing?",
    "breakfast": "Is this commonly eaten at breakfast?",
    "dessert": "Is this commonly served as a dessert?",
    "comfort_food": "Is this commonly regarded as comfort food?",
}
SCALE_QUESTIONS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "spice_level": (
        "How spicy is this dish?",
        ("not spicy", "mild", "medium", "hot"),
    ),
    "heaviness": (
        "How heavy or substantial is this dish or drink?",
        ("light", "moderate", "heavy"),
    ),
}


@dataclass(frozen=True)
class ScaleScore:
    value: float
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class ItemScores:
    tags: Mapping[str, float]
    scales: Mapping[str, ScaleScore]


class SemanticScorer(Protocol):
    model_id: str

    def score(self, text: str) -> ItemScores: ...


@dataclass(frozen=True)
class ItemText:
    text: str
    input_level: str


@dataclass(frozen=True)
class TagResult:
    item_id: str
    input_level: str
    scores: ItemScores
    model_id: str
    vocabulary_version: str
    input_hash: str


def _description_text(value: Any) -> str | None:
    if isinstance(value, DescriptionResult):
        return value.text if value.status == "described" else None
    if isinstance(value, AttributedValue):
        return value.value
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def describe_item(
    item: MenuItem,
    menu: Menu,
    inferred_description: DescriptionResult | AttributedValue | str | None = None,
) -> ItemText:
    """Build clean scorer input without consuming prior enrichment output."""
    category_by_id = {category.id: category.name for category in menu.categories}
    categories = [
        name
        for category_id in item.category_ids
        if (name := category_by_id.get(category_id))
        and not CATEGORY_PLACEHOLDER_PATTERN.search(name.strip())
    ]
    real_source_description = item.description if not needs_description(item) else None
    inferred = _description_text(inferred_description)

    lines = [f"Name: {item.name}"]
    if categories:
        lines.append(f"Categories: {', '.join(categories)}")
    if real_source_description:
        lines.append(f"Description: {real_source_description}")
        input_level = "source_description"
    elif inferred:
        lines.append(f"Inferred description: {inferred}")
        input_level = "inferred_description"
    elif categories:
        input_level = "name_category"
    else:
        input_level = "name_only"
    for group in item.variation_groups:
        choice_names = ", ".join(dict.fromkeys(choice.name for choice in group.choices))
        if choice_names:
            lines.append(f"{group.name}: {choice_names}")
    return ItemText("\n".join(lines), input_level)


def _validate_scores(scores: ItemScores) -> None:
    if set(scores.tags) != set(TAG_QUESTIONS):
        raise ValueError("tag scores must contain every vocabulary tag exactly once")
    if set(scores.scales) != set(SCALE_QUESTIONS):
        raise ValueError("scale scores must contain every vocabulary scale exactly once")
    for name, value in scores.tags.items():
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"tag {name!r} has an out-of-range score")
    for name, scale in scores.scales.items():
        expected_levels = SCALE_QUESTIONS[name][1]
        if set(scale.probabilities) != set(expected_levels):
            raise ValueError(f"scale {name!r} has incomplete probabilities")
        values = (scale.value, scale.confidence, *scale.probabilities.values())
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"scale {name!r} has an out-of-range score")


class StrandsDeciderScorer:
    """Local semantic scorer using Strands Decider's actual engine API."""

    def __init__(
        self,
        *,
        checkpoint: str = "StrandsAgents/strands-decider-2B-hobson-v19",
        engine: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = checkpoint
        self._engine = engine
        self._device = device

    def _load_engine(self) -> Any:
        if self._engine is None:
            try:
                import torch
                from strands_decider.infer import load_engine
            except ImportError as exc:
                raise RuntimeError("strands-decider and torch are required for semantic scoring") from exc
            selected_device = self._device or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
            self._engine = load_engine(self.model_id, device=selected_device)
        return self._engine

    @staticmethod
    def _questions() -> dict[str, Any]:
        from strands_decider.schema import NoulQuestion, ScoreQuestion

        questions: dict[str, Any] = {
            name: NoulQuestion(instructions=question)
            for name, question in TAG_QUESTIONS.items()
        }
        questions.update(
            {
                name: ScoreQuestion(instructions=question, criteria=list(levels))
                for name, (question, levels) in SCALE_QUESTIONS.items()
            }
        )
        return questions

    def score(self, text: str) -> ItemScores:
        response = self._load_engine().ask(text, self._questions())
        answers = response.answers if hasattr(response, "answers") else response
        tags = {name: float(answers[name].noul) for name in TAG_QUESTIONS}
        scales: dict[str, ScaleScore] = {}
        for name, (_, levels) in SCALE_QUESTIONS.items():
            answer = answers[name]
            raw_probabilities = dict(answer.probabilities)
            legend = dict(answer.legend)
            probabilities: dict[str, float] = {}
            for index, level in enumerate(levels):
                key = str(index)
                legend_label = legend.get(key, level)
                probability = raw_probabilities.get(key)
                if probability is None:
                    probability = raw_probabilities.get(index)
                probabilities[legend_label] = float(probability)
            # Normalize legend labels back to the stable vocabulary level names.
            if set(probabilities) != set(levels):
                probabilities = {
                    level: float(raw_probabilities.get(str(index), raw_probabilities.get(index)))
                    for index, level in enumerate(levels)
                }
            scales[name] = ScaleScore(
                value=float(answer.score) / (len(levels) - 1),
                probabilities=probabilities,
                confidence=float(answer.confidence),
            )
        result = ItemScores(tags=tags, scales=scales)
        _validate_scores(result)
        return result


def _scores_to_row(scores: ItemScores) -> dict[str, Any]:
    return {
        "tags": dict(scores.tags),
        "scales": {
            name: {
                "value": scale.value,
                "probabilities": dict(scale.probabilities),
                "confidence": scale.confidence,
            }
            for name, scale in scores.scales.items()
        },
    }


def _scores_from_row(row: Mapping[str, Any]) -> ItemScores:
    return ItemScores(
        tags={name: float(value) for name, value in row["tags"].items()},
        scales={
            name: ScaleScore(
                value=float(scale["value"]),
                probabilities={
                    level: float(probability)
                    for level, probability in scale["probabilities"].items()
                },
                confidence=float(scale["confidence"]),
            )
            for name, scale in row["scales"].items()
        },
    )


def _tag_from_row(item_id: str, row: Mapping[str, Any]) -> TagResult:
    return TagResult(
        item_id=item_id,
        input_level=str(row["input_level"]),
        scores=_scores_from_row(row["scores"]),
        model_id=str(row["model_id"]),
        vocabulary_version=str(row["vocabulary_version"]),
        input_hash=str(row["input_hash"]),
    )


def _tag_row(result: TagResult) -> dict[str, Any]:
    return {
        "input_level": result.input_level,
        "scores": _scores_to_row(result.scores),
        "model_id": result.model_id,
        "vocabulary_version": result.vocabulary_version,
        "input_hash": result.input_hash,
    }


def score_menu(
    menu: Menu,
    descriptions: Sequence[DescriptionResult],
    scorer: SemanticScorer,
    cache: EnrichmentCache,
    *,
    cache_path: str | os.PathLike[str] | None = None,
    save_every: int = 25,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[TagResult, ...]:
    """Return scores for every item, calling the scorer only for stale cache misses.

    The progress callback reports ``newly_scored, total_to_score``. Cached items
    are deliberately excluded from both values.
    """
    if save_every <= 0:
        raise ValueError("save_every must be positive")
    if cache.menu_id != menu.id:
        raise ValueError(f"cache belongs to menu {cache.menu_id!r}, not {menu.id!r}")
    validate_menu(menu).raise_for_errors()
    description_by_id = {result.item_id: result for result in descriptions}
    planned: list[tuple[MenuItem, ItemText, str, TagResult | None]] = []
    for item in menu.items:
        item_text = describe_item(item, menu, description_by_id.get(item.id))
        digest = content_hash(
            json.dumps(asdict(item_text), sort_keys=True, separators=(",", ":"))
        )
        row = cache.tags.get(item.id)
        cached_result: TagResult | None = None
        if (
            isinstance(row, Mapping)
            and row.get("input_hash") == digest
            and row.get("model_id") == scorer.model_id
            and row.get("vocabulary_version") == VOCABULARY_VERSION
        ):
            cached_result = _tag_from_row(item.id, row)
            _validate_scores(cached_result.scores)
        planned.append((item, item_text, digest, cached_result))

    total_to_score = sum(cached_result is None for *_, cached_result in planned)
    results: dict[str, TagResult] = {}
    newly_scored = 0
    last_reported = -1
    try:
        for item, item_text, digest, cached_result in planned:
            if cached_result is None:
                scores = scorer.score(item_text.text)
                _validate_scores(scores)
                result = TagResult(
                    item_id=item.id,
                    input_level=item_text.input_level,
                    scores=scores,
                    model_id=scorer.model_id,
                    vocabulary_version=VOCABULARY_VERSION,
                    input_hash=digest,
                )
                cache.tags[item.id] = _tag_row(result)
                newly_scored += 1
                if cache_path is not None and newly_scored % save_every == 0:
                    cache.save(cache_path)
                if progress is not None and (
                    newly_scored % 25 == 0 or newly_scored == total_to_score
                ):
                    progress(newly_scored, total_to_score)
                    last_reported = newly_scored
            else:
                result = cached_result
            results[item.id] = result
    finally:
        if cache_path is not None:
            cache.save(cache_path)
        if progress is not None and newly_scored != last_reported:
            progress(newly_scored, total_to_score)
    return tuple(results[item.id] for item in menu.items)


def apply_enrichment(
    menu: Menu,
    descriptions: Sequence[DescriptionResult],
    tags: Sequence[TagResult],
    *,
    threshold: float = 0.7,
    excluded_tags: Sequence[str] = ("shareable", "comfort_food"),
) -> Menu:
    """Return a newly enriched immutable menu."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than zero and at most one")
    unknown_exclusions = set(excluded_tags) - set(TAG_QUESTIONS)
    if unknown_exclusions:
        raise ValueError(f"unknown excluded tags: {sorted(unknown_exclusions)!r}")
    if any(item.semantic_attributes or item.inferred_description for item in menu.items):
        raise ValueError("apply_enrichment requires a freshly normalized menu")

    description_by_id = {result.item_id: result for result in descriptions}
    tag_by_id = {result.item_id: result for result in tags}
    enriched_items: list[MenuItem] = []
    for item in menu.items:
        description = description_by_id.get(item.id)
        inferred_value: AttributedValue | None = None
        if (
            needs_description(item)
            and description is not None
            and description.status == "described"
            and description.text
        ):
            inferred_value = AttributedValue(
                value=description.text,
                origin="inferred",
                evidence=f"{description.model_id} prompt {description.prompt_version}",
            )

        tag_result = tag_by_id.get(item.id)
        attributes: tuple[AttributedValue, ...] = ()
        if tag_result is not None:
            _validate_scores(tag_result.scores)
            attributes = tuple(
                AttributedValue(
                    value=name,
                    origin="inferred",
                    confidence=float(probability),
                    evidence=f"{tag_result.model_id} vocab {tag_result.vocabulary_version}",
                )
                for name, probability in sorted(tag_result.scores.tags.items())
                if probability >= threshold and name not in excluded_tags
            )

        embedding_lines = [item.embedding_text]
        if inferred_value is not None:
            embedding_lines.append(f"Inferred description: {inferred_value.value}")
        if attributes:
            embedding_lines.append(
                "Semantic attributes: " + ", ".join(value.value for value in attributes)
            )
        enriched_items.append(
            replace(
                item,
                inferred_description=inferred_value,
                semantic_attributes=attributes,
                embedding_text="\n".join(embedding_lines),
            )
        )
    enriched = replace(menu, items=tuple(enriched_items))
    validate_menu(enriched).raise_for_errors()
    return enriched


def _assert_safe_vocabulary() -> None:
    forbidden = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in FORBIDDEN_TERMS) + r")(?!\w)",
        re.IGNORECASE,
    )
    values = list(TAG_QUESTIONS.items()) + [
        (name, " ".join((question, *levels)))
        for name, (question, levels) in SCALE_QUESTIONS.items()
    ]
    for name, text in values:
        if forbidden.search(name) or forbidden.search(text):
            raise ValueError(f"semantic vocabulary {name!r} contains forbidden safety language")


_assert_safe_vocabulary()
