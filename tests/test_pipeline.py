"""
Integration tests for the ingest() pipeline.

Uses MockEmbedder + MockVectorStore — no Qdrant, no model download.

Key change from A1:
  All chunks now go into a single "documents" collection.
  Tenant isolation is via tenant_id payload filter, not separate collections.
  Tests updated to reflect this.
"""

import json
import asyncio

from ingestor.pipeline import ingest, IngestResult, QDRANT_COLLECTION
from ingestor.embedder import MockEmbedder
from ingestor.vector_store import MockVectorStore


# ======================================================================
# Fixtures
# ======================================================================

ADI_DOC = {
    "paragraphs": [
        {
            "content": "Annual Deductible",
            "role": "sectionHeading",
            "bounding_regions": [{"page_number": 1}],
        },
        {
            "content": "The deductible is $1,500 per individual.",
            "role": None,
            "bounding_regions": [{"page_number": 1}],
        },
        {
            "content": "Family coverage is capped at $3,000.",
            "role": None,
            "bounding_regions": [{"page_number": 1}],
        },
    ],
    "tables": [
        {
            "row_count": 2,
            "column_count": 2,
            "cells": [
                {"row_index": 0, "column_index": 0, "content": "Service",            "kind": "columnHeader"},
                {"row_index": 0, "column_index": 1, "content": "Your Cost",          "kind": "columnHeader"},
                {"row_index": 1, "column_index": 0, "content": "Primary Care Visit", "kind": "content"},
                {"row_index": 1, "column_index": 1, "content": "$20 copay",          "kind": "content"},
            ],
            "bounding_regions": [{"page_number": 1}],
            "caption": {"content": "Table 1: Cost Sharing"},
        }
    ],
    "figures": [
        {
            "bounding_regions": [{"page_number": 2}],
            "caption": {"content": "Figure 1: Network Map"},
        }
    ],
    "content": "Annual Deductible\nThe deductible is $1,500...",
}

DOCLING_DOC = {
    "schema_name": "DoclingDocument",
    "version": "1.3.0",
    "name": "benefit_summary",
    "origin": {"mimetype": "application/pdf", "filename": "benefit_summary.pdf"},
    "body": {"self_ref": "#/body", "children": [], "label": "unspecified"},
    "texts": [
        {
            "self_ref": "#/texts/0",
            "label": "section_header",
            "prov": [{"page_no": 1}],
            "text": "Annual Deductible",
        },
        {
            "self_ref": "#/texts/1",
            "label": "text",
            "prov": [{"page_no": 1}],
            "text": "The deductible is $1,500 per individual.",
        },
        {
            "self_ref": "#/texts/5",
            "label": "caption",
            "prov": [{"page_no": 1}],
            "text": "Table 1: Cost Sharing",
        },
    ],
    "tables": [
        {
            "self_ref": "#/tables/0",
            "prov": [{"page_no": 1}],
            "data": {
                "table_cells": [
                    {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Service",            "column_header": True},
                    {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "Your Cost",          "column_header": True},
                    {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Primary Care Visit", "column_header": False},
                    {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "$20 copay",          "column_header": False},
                ],
                "num_rows": 2,
                "num_cols": 2,
            },
            "captions": [{"$ref": "#/texts/5"}],
        }
    ],
    "pictures": [],
    "pages": {"1": {"size": {"width": 612, "height": 792}}},
}


def to_bytes(doc: dict) -> bytes:
    return json.dumps(doc).encode("utf-8")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ======================================================================
# IngestResult shape
# ======================================================================

class TestIngestResultShape:

    def test_returns_ingest_result(self):
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert isinstance(result, IngestResult)

    def test_success_status_on_valid_doc(self):
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "success"
        assert result.error is None

    def test_document_id_in_result(self):
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-42", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.document_id == "doc-42"

    def test_tenant_id_in_result(self):
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="acme-corp", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.tenant_id == "acme-corp"

    def test_chunks_ingested_greater_than_zero(self):
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        # heading excluded: 2 text + 1 table + 1 figure = 4 chunks
        assert result.chunks_ingested == 4


# ======================================================================
# Tenant isolation — single collection, payload filter
# ======================================================================

class TestTenantIsolation:

    def test_all_chunks_go_to_single_documents_collection(self):
        """
        Updated: all tenants share one 'documents' collection.
        Isolation is via tenant_id payload field, not collection name.
        """
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="acme", collection_id="benefits",
            embedder=MockEmbedder(), vector_store=store,
        ))
        assert QDRANT_COLLECTION in store.collections

    def test_tenant_id_in_every_payload(self):
        """Tenant isolation relies on tenant_id being in every point payload."""
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="acme-corp", collection_id="benefits",
            embedder=MockEmbedder(), vector_store=store,
        ))
        for point in store.collections[QDRANT_COLLECTION]:
            assert point.payload["tenant_id"] == "acme-corp"

    def test_collection_id_in_every_payload(self):
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="acme", collection_id="benefits-2024",
            embedder=MockEmbedder(), vector_store=store,
        ))
        for point in store.collections[QDRANT_COLLECTION]:
            assert point.payload["collection_id"] == "benefits-2024"

    def test_two_tenants_points_have_different_tenant_ids(self):
        """Both tenants write to same collection but have different tenant_id values."""
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="f.json",
            document_id="doc-1", tenant_id="tenant-a", collection_id="col",
            embedder=MockEmbedder(), vector_store=store,
        ))
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="f.json",
            document_id="doc-2", tenant_id="tenant-b", collection_id="col",
            embedder=MockEmbedder(), vector_store=store,
        ))
        all_points = store.collections[QDRANT_COLLECTION]
        tenant_a_points = [p for p in all_points if p.payload["tenant_id"] == "tenant-a"]
        tenant_b_points = [p for p in all_points if p.payload["tenant_id"] == "tenant-b"]
        assert len(tenant_a_points) > 0
        assert len(tenant_b_points) > 0
        # No cross-contamination
        assert all(p.payload["tenant_id"] != "tenant-b" for p in tenant_a_points)
        assert all(p.payload["tenant_id"] != "tenant-a" for p in tenant_b_points)


# ======================================================================
# Vector store upsert
# ======================================================================

class TestVectorStoreUpsert:

    def test_vectors_written_to_store(self):
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=store,
        ))
        assert len(store.collections[QDRANT_COLLECTION]) > 0

    def test_upserted_point_has_required_payload_fields(self):
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=store,
        ))
        required = {
            "document_id", "tenant_id", "collection_id",
            "page_number", "block_type", "section_heading",
            "chunk_index", "text", "tags",
        }
        for point in store.collections[QDRANT_COLLECTION]:
            for field in required:
                assert field in point.payload, f"Missing field: {field}"

    def test_upserted_point_has_vector(self):
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            embedder=MockEmbedder(), vector_store=store,
        ))
        for point in store.collections[QDRANT_COLLECTION]:
            assert isinstance(point.vector, list)
            assert len(point.vector) == 384

    def test_tags_flow_into_payload(self):
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            tags=["insurance", "2024"], embedder=MockEmbedder(), vector_store=store,
        ))
        for point in store.collections[QDRANT_COLLECTION]:
            assert point.payload["tags"] == ["insurance", "2024"]

    def test_metadata_flows_into_payload(self):
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="benefit.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col-1",
            metadata={"source": "upload", "year": "2024"},
            embedder=MockEmbedder(), vector_store=store,
        ))
        for point in store.collections[QDRANT_COLLECTION]:
            assert point.payload.get("source") == "upload"


# ======================================================================
# Parser auto-detection
# ======================================================================

class TestParserDetection:

    def test_adi_auto_detected(self):
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="doc.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "success"

    def test_docling_auto_detected(self):
        result = run(ingest(
            file_bytes=to_bytes(DOCLING_DOC), filename="doc.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "success"

    def test_adi_with_business_type(self):
        """document_type is a business category, not a parser selector.
        Parser is auto-detected from JSON shape regardless of document_type."""
        result = run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="doc.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            document_type="insurance", embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "success"

    def test_docling_with_business_type(self):
        """Same business type, different parser format — both auto-detected."""
        result = run(ingest(
            file_bytes=to_bytes(DOCLING_DOC), filename="doc.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            document_type="insurance", embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "success"

    def test_document_type_stored_in_payload(self):
        """document_type (business category) must appear in every point payload
        so the retrieval layer can filter by document category."""
        store = MockVectorStore()
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="doc.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            document_type="insurance", embedder=MockEmbedder(), vector_store=store,
        ))
        for point in store.collections[QDRANT_COLLECTION]:
            assert point.payload["document_type"] == "insurance"


# ======================================================================
# Error handling
# ======================================================================

class TestErrorHandling:

    def test_malformed_json_returns_failed(self):
        result = run(ingest(
            file_bytes=b"NOT JSON {{{{", filename="bad.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "failed"
        assert result.chunks_ingested == 0
        assert result.error is not None

    def test_empty_document_returns_partial(self):
        empty = {"paragraphs": [], "tables": [], "figures": [], "content": ""}
        result = run(ingest(
            file_bytes=to_bytes(empty), filename="empty.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            embedder=MockEmbedder(), vector_store=MockVectorStore(),
        ))
        assert result.status == "partial"
        assert result.chunks_ingested == 0


# ======================================================================
# Idempotency — collision-safe UUID5
# ======================================================================

class TestIdempotency:

    def test_same_document_produces_same_point_ids(self):
        """
        uuid5(tenant_id + collection_id + document_id + chunk_index)
        is deterministic — re-ingesting same doc produces same IDs.
        In Qdrant, upsert on same ID = overwrite, not duplicate.
        """
        store1 = MockVectorStore()
        store2 = MockVectorStore()

        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="f.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            embedder=MockEmbedder(), vector_store=store1,
        ))
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="f.json",
            document_id="doc-001", tenant_id="t", collection_id="c",
            embedder=MockEmbedder(), vector_store=store2,
        ))

        ids1 = {p.id for p in store1.collections[QDRANT_COLLECTION]}
        ids2 = {p.id for p in store2.collections[QDRANT_COLLECTION]}
        assert ids1 == ids2

    def test_different_tenants_same_document_id_produce_different_point_ids(self):
        """
        Critical: two tenants using the same document_id string
        must NOT produce the same point IDs — that would cause
        one tenant's data to silently overwrite another's.
        """
        store = MockVectorStore()

        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="f.json",
            document_id="doc-001", tenant_id="tenant-a", collection_id="col",
            embedder=MockEmbedder(), vector_store=store,
        ))
        run(ingest(
            file_bytes=to_bytes(ADI_DOC), filename="f.json",
            document_id="doc-001", tenant_id="tenant-b", collection_id="col",
            embedder=MockEmbedder(), vector_store=store,
        ))

        all_points = store.collections[QDRANT_COLLECTION]
        all_ids = [p.id for p in all_points]
        # All IDs must be unique — no collision between tenants
        assert len(all_ids) == len(set(all_ids))