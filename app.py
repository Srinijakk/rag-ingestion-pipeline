"""
FastAPI  — exposes the ingest() pipeline as an HTTP endpoint.

POST /ingest
  - Accepts the parser JSON file as a multipart upload
  - Accepts document metadata as form fields
  - Returns IngestResult as JSON

"""

import json
import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from ingestor.pipeline import IngestResult, ingest
from ingestor.embedder import SentenceTransformerEmbedder
from ingestor.vector_store import QdrantVectorStore

app = FastAPI(
    title="RAG Ingestion Pipeline",
    description="Production-grade document ingestion pipeline for enterprise RAG systems.",
    version="1.0.0",
)

# Shared instances — created once at startup, reused across requests
# In production: use FastAPI lifespan events for proper startup/shutdown
_embedder = SentenceTransformerEmbedder()
_vector_store = QdrantVectorStore(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)


@app.get("/healthz")
async def health():
    """Health check — used by docker-compose and load balancers."""
    return {"status": "ok"}


@app.post("/ingest", response_model=None)
async def ingest_document(
    file: UploadFile = File(..., description="Parser output JSON (ADI or Docling format)"),
    document_id: str = Form(..., description="Unique document identifier"),
    tenant_id: str = Form(..., description="Tenant identifier for data isolation"),
    collection_id: str = Form(..., description="Logical collection within the tenant"),
    document_type: str = Form(None, description="Business document category e.g. 'insurance', 'clinical_protocol', 'regulatory_filing'"),
    tags: str = Form("", description="Comma-separated tags e.g. 'insurance,2024,benefits'"),
    metadata: str = Form("{}", description="JSON string of arbitrary metadata e.g. '{\"year\":\"2024\"}'"),
    filename: str = Form(None, description="Original filename (defaults to uploaded filename)"),
) -> JSONResponse:
    """
    Ingest a parsed document JSON into the RAG vector store.

    Accepts multipart/form-data with:
      - file:          parser output JSON (ADI or Docling)
      - document_id, tenant_id, collection_id: required identifiers
      - document_type: optional business category (e.g. 'insurance', 'clinical_protocol')
      - tags:          optional comma-separated tags
      - metadata:      optional JSON string of arbitrary key-value pairs
    """
    file_bytes = await file.read()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    original_filename = filename or file.filename or "unknown"

    # Parse metadata JSON string — default to empty dict on failure
    try:
        meta_dict = json.loads(metadata) if metadata else {}
        if not isinstance(meta_dict, dict):
            meta_dict = {}
    except (json.JSONDecodeError, TypeError):
        meta_dict = {}

    meta_dict["original_filename"] = original_filename

    result: IngestResult = await ingest(
        file_bytes=file_bytes,
        filename=original_filename,
        document_id=document_id,
        tenant_id=tenant_id,
        collection_id=collection_id,
        document_type=document_type or None,
        tags=tag_list,
        metadata=meta_dict,
        embedder=_embedder,
        vector_store=_vector_store,
    )

    status_code = 200 if result.status == "success" else 207 if result.status == "partial" else 422

    return JSONResponse(
        status_code=status_code,
        content={
            "document_id":     result.document_id,
            "tenant_id":       result.tenant_id,
            "collection_id":   result.collection_id,
            "document_type":   result.document_type,
            "chunks_ingested": result.chunks_ingested,
            "status":          result.status,
            "error":           result.error,
        },
    )