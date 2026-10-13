"""Local Qwen3 embeddings via sentence-transformers; embedding-only (no text generation)."""

from typing import Any, Callable, Mapping, Sequence, TypeVar

from .base import AIAdapter, AIError

T = TypeVar("T")


class QwenAdapter(AIAdapter):
    provider = "qwen"
    # Qwen3-Embedding expects a task instruction on queries only, never on documents.
    QUERY_PROMPT = ("Instruct: Given a restaurant customer's request, retrieve menu items "
                    "that satisfy it\nQuery: ")

    def __init__(self, *, embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 client: Any = None, **kwargs: Any) -> None:
        super().__init__(text_model="embedding-only", embedding_model=embedding_model, **kwargs)
        if client is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise AIError("Install sentence-transformers for the qwen provider") from exc
            client = SentenceTransformer(embedding_model)
        native = client.get_sentence_embedding_dimension()
        # Matryoshka training keeps truncated prefixes useful; longer is impossible.
        if self.dimensions > native:
            raise ValueError(f"{embedding_model} supports at most {native} dimensions")
        self.client = client

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        raise AIError("The qwen provider is embedding-only; use AI_PROVIDER=openai or "
                      "gemini for descriptions and dietary inference")

    def _embed_batch(self, texts: Sequence[str], *, query: bool) -> Sequence[Sequence[float]]:
        options = {"prompt": self.QUERY_PROMPT} if query else {}
        rows = self.client.encode(list(texts), batch_size=self.batch_size, **options)
        return [list(row)[:self.dimensions] for row in rows]

    def _call(self, operation: Callable[[], T]) -> T:
        # In-process inference has no rate limits or transient HTTP failures to retry.
        try:
            return operation()
        except AIError:
            raise
        except Exception as exc:
            raise AIError(f"qwen embedding failed ({type(exc).__name__})") from exc
