"""OpenAI Responses and embeddings implementation of the shared AI contract."""

import os
from typing import Any, Mapping, Sequence

from .base import AIAdapter, AIError, AIResponseError


class OpenAIAdapter(AIAdapter):
    provider = "openai"

    def __init__(self, *, text_model: str = "gpt-4.1-mini",
                 embedding_model: str = "text-embedding-3-small", api_key: str | None = None,
                 client: Any = None, **kwargs: Any) -> None:
        super().__init__(text_model=text_model, embedding_model=embedding_model, **kwargs)
        limits = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        if embedding_model not in limits or self.dimensions > limits[embedding_model]:
            raise ValueError("unsupported OpenAI embedding model or dimensions")
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise AIError("OPENAI_API_KEY is required")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise AIError("Install requirements-enrichment.txt for openai") from exc
            client = OpenAI(api_key=key, max_retries=0, timeout=60)
        self.client = client

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        self._check_schema(prompt, schema)
        response = self._call(lambda: self.client.responses.create(
            model=self.text_model, input=prompt, store=False,
            text={"format": {"type": "json_schema", "name": "result",
                             "schema": dict(schema), "strict": True}},
        ))
        if response.status != "completed":
            raise AIResponseError("OpenAI returned an incomplete response")
        for item in response.output:
            for content in getattr(item, "content", ()) or ():
                if getattr(content, "type", None) == "refusal":
                    raise AIResponseError("OpenAI declined the structured-output request")
        return self._validate_json(response.output_text, schema)

    def _embed_batch(self, texts: Sequence[str], *, query: bool) -> Sequence[Sequence[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model, input=list(texts),
            dimensions=self.dimensions, encoding_format="float",
        )
        rows = response.data
        if sorted(row.index for row in rows) != list(range(len(texts))):
            raise AIResponseError("OpenAI returned missing or duplicate embedding indices")
        return [row.embedding for row in sorted(rows, key=lambda row: row.index)]
