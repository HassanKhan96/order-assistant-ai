"""Provider-neutral AI contract, validation, and bounded request retries."""

from abc import ABC, abstractmethod
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar

T = TypeVar("T")


class AIError(RuntimeError):
    """An AI request failed; callers need not depend on an SDK exception."""


class AIResponseError(AIError, ValueError):
    """The provider returned unusable structured data or vectors."""


class AIQuotaError(AIError):
    """The provider reported exhausted account credits/quota, not transient pacing."""


class AIAdapter(ABC):
    provider: str

    def __init__(self, *, text_model: str, embedding_model: str, dimensions: int = 768,
                 batch_size: int = 32, requests_per_minute: float = 60,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not text_model.strip() or not embedding_model.strip():
            raise ValueError("model names must not be empty")
        if type(dimensions) is not int or not 1 <= dimensions <= 3072:
            raise ValueError("dimensions must be between 1 and 3072")
        if type(batch_size) is not int or not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100")
        if not math.isfinite(requests_per_minute) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive and finite")
        self.text_model = text_model
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self._sleep, self._clock = sleep, clock
        self._last_request_at: float | None = None

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.text_model}"

    @property
    def cache_model_ids(self) -> tuple[str, ...]:
        return (self.model_id,)

    @property
    def embedding_model_id(self) -> str:
        return f"{self.provider}:{self.embedding_model}:{self.dimensions}:retrieval-v1"

    @abstractmethod
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Return a validated JSON object matching the supplied JSON Schema."""

    @abstractmethod
    def _embed_batch(self, texts: Sequence[str], *, query: bool) -> Sequence[Sequence[float]]:
        """Translate one embedding request into the provider's SDK format."""

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._embed(texts, query=False)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed((text,), query=True)[0]

    def _embed(self, texts: Sequence[str], *, query: bool) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, str) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding inputs must be a sequence of non-empty strings")
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            rows = self._call(lambda: self._embed_batch(batch, query=query))
            if len(rows) != len(batch):
                raise AIResponseError("embedding provider returned the wrong number of vectors")
            for row in rows:
                try:
                    values = tuple(float(value) for value in row)
                except (TypeError, ValueError) as exc:
                    raise AIResponseError("embedding values must be numeric") from exc
                if len(values) != self.dimensions:
                    raise AIResponseError("embedding provider returned the wrong dimensions")
                norm = math.hypot(*values)
                if not math.isfinite(norm) or norm == 0:
                    raise AIResponseError("embedding must be finite and non-zero")
                result.append(tuple(value / norm for value in values))
        return tuple(result)

    def _call(self, operation: Callable[[], T]) -> T:
        for attempt in range(4):
            if self._last_request_at is not None:
                delay = 60 / self.requests_per_minute - (self._clock() - self._last_request_at)
                if delay > 0:
                    self._sleep(delay)
            try:
                return operation()
            except AIError:
                raise
            except Exception as exc:
                body = getattr(exc, "body", None)
                details = body.get("error", body) if isinstance(body, dict) else {}
                if not isinstance(details, dict):
                    details = {}
                quota_codes = {"insufficient_quota", "credit_balance_exhausted"}
                if any(value in quota_codes for value in (
                    str(getattr(exc, "code", "")), str(details.get("code")), str(details.get("type"))
                )):
                    raise AIQuotaError(f"{self.provider} API credits/quota exhausted; check API billing") from exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                try:
                    status = int(status)
                except (TypeError, ValueError):
                    status = None
                # Do not echo SDK messages, which may include request data.
                if attempt == 3 or (status != 429 and (status is None or status < 500)):
                    suffix = f" (HTTP {status})" if status else ""
                    raise AIError(f"{self.provider} request failed{suffix}") from exc
                self._sleep(2 ** attempt)
            finally:
                self._last_request_at = self._clock()
        raise AssertionError("unreachable")

    @staticmethod
    def _check_schema(prompt: str, schema: Mapping[str, Any]) -> None:
        from jsonschema import Draft202012Validator
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if schema.get("type") != "object":
            raise ValueError("structured output schema must have an object root")
        Draft202012Validator.check_schema(schema)

    @staticmethod
    def _validate_json(value: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
        from jsonschema import Draft202012Validator
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError) as exc:
                raise AIResponseError("provider returned invalid JSON") from exc
        if not isinstance(value, dict) or not Draft202012Validator(schema).is_valid(value):
            raise AIResponseError("provider response does not match the requested schema")
        return value
