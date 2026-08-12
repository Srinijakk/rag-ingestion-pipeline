"""
Embedding abstraction layer.

Why abstract the embedder?
  - Tests use MockEmbedder — no model download, no GPU, instant CI
  - Production uses SentenceTransformerEmbedder (local, free)
  - Can swap to OpenAIEmbedder in one line without touching pipeline code
  - The embedder is injected into ingest() — never hardcoded

Model choice: all-MiniLM-L6-v2
  - 384 dimensions — compact, fast
  - Strong performance on semantic similarity benchmarks
  - Runs on CPU — no GPU required for typical enterprise doc volumes
  - Entirely offline — no API key, no data leaves the instance
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Contract that all embedder implementations must satisfy.
    Input: list of text strings.
    Output: list of float vectors (one per input, same order).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector size — must match the Qdrant collection vector size."""
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Production embedder — runs sentence-transformers locally.

    Lazy-loads the model on first use so importing this module
    does not trigger a 400MB download in test environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None      # lazy-loaded
        self._dimensions = 384  # default for all-MiniLM-L6-v2

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class MockEmbedder(BaseEmbedder):
    """
    Test embedder — returns deterministic zero vectors instantly.

    Using zero vectors is fine for tests because we're asserting on
    chunk count, metadata, and pipeline flow — not retrieval quality.
    """

    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimensions for _ in texts]