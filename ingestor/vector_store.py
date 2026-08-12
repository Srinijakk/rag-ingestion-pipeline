"""
Vector store abstraction layer.

Tenant isolation strategy 

  Single Qdrant collection: "documents"
  Tenant isolation via payload filter on tenant_id + collection_id..

  Trade-off acknowledged:
    Separate collections give harder physical isolation. For tenants with
    strict regulatory requirements (e.g. healthcare, government), dedicated
    collections or Qdrant deployments may be needed. The BaseVectorStore
    abstraction here makes that a one-file change.

Idempotency:
  Point IDs include tenant_id + collection_id + document_id + chunk_index.
  Two tenants with the same document_id never collide.
  Re-ingesting the same document safely overwrites existing points.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorPoint:
    """A single vector + metadata payload to write to the store."""
    id: str
    vector: list[float]
    payload: dict


class BaseVectorStore(ABC):

    @abstractmethod
    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create the collection if it does not already exist."""
        ...

    @abstractmethod
    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Write a batch of vector points. Overwrites on ID collision."""
        ...


class QdrantVectorStore(BaseVectorStore):
    """
    Production vector store — Qdrant.

    Why Qdrant?
      - Open-source, runs via Docker (one line in docker-compose)
      - Payload filtering with HNSW index — fast tenant-scoped retrieval
      - Native support for sparse + dense hybrid search (future upgrade)
      - Strong Python SDK with upsert, scroll, and delete-by-filter
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(host=host, port=port)
        self._Distance = Distance
        self._VectorParams = VectorParams

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        from qdrant_client.models import PayloadSchemaType

        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=self._VectorParams(
                    size=vector_size,
                    distance=self._Distance.COSINE,
                ),
            )

            # Create payload indexes for fields used in every tenant query.
            # Without indexes, Qdrant scans all vectors for filtered queries.
            # With indexes, filtered retrieval is efficient at millions of docs.
            for field in ("tenant_id", "collection_id", "document_id"):
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )


class MockVectorStore(BaseVectorStore):
    """
    In-memory store for unit tests.
    Captures upserted points — tests assert directly on stored data.
    No Qdrant process required.
    """

    def __init__(self):
        self.collections: dict[str, list[VectorPoint]] = {}

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        self.collections.setdefault(collection_name, [])

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        self.collections.setdefault(collection_name, []).extend(points)
