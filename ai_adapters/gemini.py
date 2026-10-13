"""Google Gemini implementation of the shared AI contract."""

import os
from typing import Any, Mapping, Sequence

from .base import AIAdapter, AIError, AIResponseError


class GeminiAdapter(AIAdapter):
    provider = "gemini"

    def __init__(self, *, text_model: str = "gemini-2.5-flash",
                 embedding_model: str = "gemini-embedding-001", api_key: str | None = None,
                 client: Any = None, **kwargs: Any) -> None:
        super().__init__(text_model=text_model, embedding_model=embedding_model, **kwargs)
        if embedding_model != "gemini-embedding-001":
            raise ValueError("Gemini adapter currently supports gemini-embedding-001 embeddings")
        if self.dimensions < 128:
            raise ValueError("Gemini embedding dimensions must be at least 128")
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise AIError("GEMINI_API_KEY is required")
            try:
                from google import genai
            except ImportError as exc:
                raise AIError("Install requirements-enrichment.txt for google-genai") from exc
            client = genai.Client(api_key=key)
        self.client = client

    @property
    def cache_model_ids(self) -> tuple[str, ...]:
        # Preserve the provenance of caches produced before provider namespacing.
        return (self.model_id, self.text_model)

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        self._check_schema(prompt, schema)
        response = self._call(lambda: self.client.models.generate_content(
            model=self.text_model, contents=prompt,
            config={"response_mime_type": "application/json", "response_json_schema": dict(schema)},
        ))
        candidates = getattr(response, "candidates", None) or ()
        for candidate in candidates:
            reason = getattr(candidate, "finish_reason", None)
            reason = getattr(reason, "value", reason)
            if reason not in (None, "STOP"):
                raise AIResponseError("Gemini returned an incomplete or blocked response")
        value = getattr(response, "parsed", None)
        if value is None:
            value = getattr(response, "text", None)
        return self._validate_json(value, schema)

    def _embed_batch(self, texts: Sequence[str], *, query: bool) -> Sequence[Sequence[float]]:
        response = self.client.models.embed_content(
            model=self.embedding_model, contents=list(texts),
            config={"task_type": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                    "output_dimensionality": self.dimensions},
        )
        return [row.values or () for row in (response.embeddings or ())]
