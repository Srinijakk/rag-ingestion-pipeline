"""
Main ingestion pipeline.

Flow:
  file_bytes (parser JSON)
    → detect parser (ADI or Docling)
    → parse        → ParsedDocument
    → chunk        → list[Chunk]
    → embed        → list[vector]
    → upsert       → Qdrant (single collection, tenant isolated via payload filter)
    → IngestResult

"""

import json
import uuid
from dataclasses import dataclass

from .chunker import Chunk, chunk_document
from .embedder import BaseEmbedder, SentenceTransformerEmbedder
from .parsers.adi import ADIParser
from .parsers.docling import DoclingParser
from .parsers.base import BaseParser
from .vector_store import BaseVectorStore, QdrantVectorStore, VectorPoint

# Single shared Qdrant collection name.
# Tenant isolation is via tenant_id payload filter — not separate collections.
# Trade-off documented in DESIGN.md.
QDRANT_COLLECTION = "documents"


@dataclass
class IngestResult:
    document_id: str
    tenant_id: str
    collection_id: str
    document_type: str | None
    chunks_ingested: int
    status: str              # "success" | "partial" | "failed"
    error: str | None = None


async def ingest(
    file_bytes: bytes,
    filename: str,
    document_id: str,
    tenant_id: str,
    collection_id: str,
    document_type: str | None = None,  # business category
    tags: list[str] | None = None,
    metadata: dict | None = None,
    # Injectable dependencies — override in tests
    embedder: BaseEmbedder | None = None,
    vector_store: BaseVectorStore | None = None,
) -> IngestResult:
    """
    Ingest a parsed document JSON into the RAG vector store.

    Args:
        file_bytes:     Raw bytes of the parser output (ADI or Docling JSON).
        filename:       Original filename — used for logging/metadata.
        document_id:    Unique ID for this document (caller-assigned).
        tenant_id:      Tenant identifier — used for payload-level isolation.
        collection_id:  Logical collection within the tenant.
        document_type:  Business document category (e.g. "insurance",
                        "clinical_protocol", "regulatory_filing"). Stored in
                        chunk payload for downstream filtering. NOT used for
                        parser selection — parser is always auto-detected
                        from JSON structure.
        tags:           List of string tags attached to every chunk payload.
        metadata:       Arbitrary dict merged into every chunk payload.
        embedder:       Embedding model (injected — defaults to SentenceTransformer).
        vector_store:   Vector store (injected — defaults to Qdrant).
    """
    tags = tags or []
    metadata = metadata or {}

    # ------------------------------------------------------------------
    # Stage 1 — Parse
    # ------------------------------------------------------------------
    try:
        raw_json = json.loads(file_bytes.decode("utf-8"))
        parser = _detect_parser(raw_json)
        parsed = parser.parse(raw_json)
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError) as exc:
        # Operational errors: malformed JSON, encoding issues, missing fields
        return IngestResult(
            document_id=document_id, tenant_id=tenant_id,
            collection_id=collection_id, document_type=document_type,
            chunks_ingested=0,
            status="failed", error=f"Parse stage failed: {exc}",
        )

    # ------------------------------------------------------------------
    # Stage 2 — Chunk
    # ------------------------------------------------------------------
    try:
        chunks: list[Chunk] = chunk_document(
            parsed=parsed,
            document_id=document_id,
            tenant_id=tenant_id,
            collection_id=collection_id,
            tags=tags,
            metadata=metadata,
        )
    except (ValueError, TypeError) as exc:
        # Operational errors: unexpected block structure from parser
        return IngestResult(
            document_id=document_id, tenant_id=tenant_id,
            collection_id=collection_id, document_type=document_type,
            chunks_ingested=0,
            status="failed", error=f"Chunk stage failed: {exc}",
        )

    if not chunks:
        return IngestResult(
            document_id=document_id, tenant_id=tenant_id,
            collection_id=collection_id, document_type=document_type,
            chunks_ingested=0,
            status="partial", error="No chunks produced — document may be empty.",
        )

    # ------------------------------------------------------------------
    # Stage 3 — Embed
    # ------------------------------------------------------------------
    _embedder = embedder or SentenceTransformerEmbedder()
    try:
        vectors = _embedder.embed([c.text for c in chunks])
    except (RuntimeError, OSError) as exc:
        # Operational errors: model load failure, OOM, device errors
        return IngestResult(
            document_id=document_id, tenant_id=tenant_id,
            collection_id=collection_id, document_type=document_type,
            chunks_ingested=0,
            status="failed", error=f"Embed stage failed: {exc}",
        )

    # ------------------------------------------------------------------
    # Stage 4 — Upsert
    # Single collection, tenant isolated via payload filter.
    # Point ID includes tenant_id to prevent cross-tenant ID collisions.
    # ------------------------------------------------------------------
    _store = vector_store or QdrantVectorStore()
    vector_size = len(vectors[0]) if vectors else _embedder.dimensions

    try:
        _store.ensure_collection(QDRANT_COLLECTION, vector_size)

        points = [
            VectorPoint(
                # Stable, collision-free ID across all tenants
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{tenant_id}:{collection_id}:{document_id}:{chunk.chunk_index}"
                )),
                vector=vector,
                payload={
                    "tenant_id":       chunk.tenant_id,
                    "collection_id":   chunk.collection_id,
                    "document_id":     chunk.document_id,
                    "document_type":   document_type,
                    "filename":        filename,
                    "chunk_index":     chunk.chunk_index,
                    "page_number":     chunk.page_number,
                    "block_type":      chunk.block_type,
                    "section_heading": chunk.section_heading,
                    "table_id":        chunk.table_id,
                    "table_caption":   chunk.table_caption,
                    "text":            chunk.text,
                    "tags":            chunk.tags,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        _store.upsert(QDRANT_COLLECTION, points)

    except (ConnectionError, TimeoutError, OSError) as exc:
        # Operational errors: Qdrant unreachable, network timeout
        return IngestResult(
            document_id=document_id, tenant_id=tenant_id,
            collection_id=collection_id, document_type=document_type,
            chunks_ingested=0,
            status="failed", error=f"Upsert stage failed: {exc}",
        )

    return IngestResult(
        document_id=document_id,
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_type=document_type,
        chunks_ingested=len(chunks),
        status="success",
    )


def _detect_parser(data: dict) -> BaseParser:
    """
    Auto-detect which parser adapter to use from JSON structure.

    document_type is a business category (e.g. "insurance"), NOT a
    parser selector. The same insurance document could be parsed by
    ADI or Docling — they are orthogonal dimensions.

    Detection priority:
      1. schema_name == "DoclingDocument" → Docling
      2. paragraphs / content keys        → ADI
      3. ADI as safe default
    """
    if data.get("schema_name") == "DoclingDocument":
        return DoclingParser()
    if "paragraphs" in data or ("tables" in data and "content" in data):
        return ADIParser()
    return ADIParser()
