"""Persistent, resumable cache for semantic menu enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def content_hash(text: str) -> str:
    """Return a stable SHA-256 digest for enrichment input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EnrichmentCache:
    """JSON-backed model outputs for one normalized menu."""

    menu_id: str
    descriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike[str], menu_id: str) -> "EnrichmentCache":
        cache_path = Path(path)
        if not cache_path.exists():
            return cls(menu_id=menu_id)

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read enrichment cache {cache_path}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"enrichment cache {cache_path} must contain a JSON object")
        if data.get("menu_id") != menu_id:
            raise ValueError(
                f"enrichment cache belongs to menu {data.get('menu_id')!r}, not {menu_id!r}"
            )

        descriptions = data.get("descriptions", {})
        tags = data.get("tags", {})
        if not isinstance(descriptions, dict) or not isinstance(tags, dict):
            raise ValueError("enrichment cache descriptions and tags must be objects")
        return cls(menu_id=menu_id, descriptions=descriptions, tags=tags)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically write the cache, preserving the previous file on failure."""
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "menu_id": self.menu_id,
            "descriptions": self.descriptions,
            "tags": self.tags,
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(payload, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, cache_path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
