"""Expose any AI adapter through the menu indexer's embedding protocol."""

from typing import Any, Sequence

from ai_adapters import AIAdapter


class AIEmbeddingProvider:
    def __init__(self, adapter: AIAdapter) -> None:
        self.adapter = adapter
        self.model_id = adapter.embedding_model_id
        self.dimensions = adapter.dimensions

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self.adapter.embed_documents(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.adapter.embed_query(text)


class GeminiEmbeddingProvider(AIEmbeddingProvider):
    """Compatibility constructor; new code should inject create_ai_adapter()."""

    def __init__(self, *, dimensions: int = 768, batch_size: int = 32,
                 api_key: str | None = None, client: Any = None) -> None:
        from ai_adapters.gemini import GeminiAdapter
        super().__init__(GeminiAdapter(dimensions=dimensions, batch_size=batch_size,
                                      api_key=api_key, client=client))

