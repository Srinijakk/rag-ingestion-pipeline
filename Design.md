# DESIGN.md — RAG Ingestion Pipeline

## 1. Architecture Overview

```
                    POST /ingest (FastAPI)
                    file_bytes, document_id, tenant_id,
                    collection_id, tags, metadata
                              │
                              ▼
                    ┌─────────────────────┐
                    │     ingest()        │
                    │    Orchestrator     │
                    │    (async)          │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │   Stage 1: Parse    │
                    │                     │
                    │  ADI Adapter   ─┐   │
                    │                 ▼   │
                    │  Docling Adapter─►  │
                    │           ParsedDoc │
                    │           RawBlock[]│
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │   Stage 2: Chunk    │
                    │                     │
                    │  Text → sentence    │
                    │  Table → row bound  │
                    │  Figure → atomic    │
                    │  Heading → metadata │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │   Stage 3: Embed    │
                    │                     │
                    │  all-MiniLM-L6-v2   │
                    │  384 dim, local     │
                    │                     │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │   Stage 4: Upsert   │
                    │                     │
                    │  Qdrant             │
                    │  Single collection  │
                    │  "documents"        │
                    │                     │
                    │  Payload filter:    │
                    │  tenant_id          │
                    │  collection_id      │
                    │  document_id        │
                    │  section_heading    │
                    │  block_type         │
                    │  table_id           │
                    └────────┬────────────┘
                             │
                             ▼
                        IngestResult
                   {status, chunks_ingested,
                    document_id, error}
```

*   **Error Handling:** Each stage catches specific operational exceptions only (e.g. JSONDecodeError, Timeout) and returns them in the `IngestResult`. Unexpected programming bugs intentionally crash to HTTP 500 so they aren't hidden in production.

---

## 2. Primary Design Driver

**"A malformed chunk or a missed table can produce a legally significant wrong answer."**
Every decision below prioritizes the correctness of chunk boundaries over simplicity to prevent dangerous hallucination in enterprise environments.

---

## 3. Key Architectural Decisions

*   **3.1 Parser Adapter Pattern:** 
    *   *Decision:* Parse outputs (ADI/Docling) are normalized into a generic `ParsedDocument`.
    *   *Why:* Isolates downstream code from upstream parser schema changes.
    *   *On parser selection:* The assignment requires support for at least one parser format and provides `document_type` as part of the ingestion interface, but does not prescribe parser-selection semantics. This implementation supports both ADI and Docling and uses schema-based detection when no parser hint is provided. ADI and Docling outputs are normalized through the `BaseParser` interface into the same `ParsedDocument` representation. This keeps parser selection decoupled from downstream chunking, embedding, and storage.

*   **3.2 Table Chunking:** 
    *   *Decision:* Tables split strictly at row boundaries, repeating the header in every chunk. 
    *   *Why:* Splitting mid-row separates a value from its label (e.g. separating "$20 copay" from "Primary Care Visit"), causing the LLM to generate incorrect answers.

*   **3.3 Section Heading Propagation:** 
    *   *Decision:* Headings are saved as metadata and attached to subsequent chunks, not embedded as standalone vectors.
    *   *Why:* Heading-only chunks pollute vector search. Propagating them as metadata allows accurate downstream filtering and source attribution.

*   **3.4 Dynamic, Structure-Aware Chunking:** 
    *   *Decision:* Chunks are variable size based on structure (sentences/rows), constrained by a hard ceiling (`MAX_CHARS = 800`, or ~200 tokens).
    *   *Why:* The limit prevents silent truncation by the chosen embedding model (`all-MiniLM-L6-v2` has a 256 word-piece limit).

*   **3.5 Tenant Isolation:** 
    *   *Decision:* One Qdrant collection (`"documents"`) with `tenant_id` payload filters.
    *   *Why:* Managing thousands of individual collections for hundreds of tenants creates massive overhead; payload filtering scales cleanly.

*   **3.6 Idempotent Ingestion:** 
    *   *Decision:* `Point ID = uuid5(tenant_id + collection_id + document_id + chunk_index)`
    *   *Why:* Ensures re-ingesting a document overwrites the old vectors exactly, preventing duplicate data and collision across tenants.

*   **3.7 Async Pipeline:** 
    *   *Decision:* The main `ingest()` function is `async`.
    *   *Why:* Allows FastAPI to handle concurrent HTTP requests and non-blocking Qdrant network I/O.

*   **3.8 Dependency Injection:** 
    *   *Decision:* Embedder and Vector Store are injected.
    *   *Why:* Enables instant, offline unit tests using `MockEmbedder` and `MockVectorStore` without real database instances or model downloads.

*   **3.9 Embedding Model:** 
    *   *Decision:* Local `sentence-transformers/all-MiniLM-L6-v2` (384 dim).
    *   *Why:* Fully offline (no data egress risk for enterprise docs), no cost, and fast.

*   **3.10 Figure Handling:** 
    *   *Decision:* Only figure captions are indexed; images are ignored.
    *   *Why:* Text-only pipelines cannot embed raw images. (Future work: Multimodal CLIP embeddings).

---

## 4. Constraints Satisfied

*   **Language:** Python 3.11, FastAPI, async.
*   **Vector store:** Open-source Qdrant (mocked in tests).
*   **Embeddings:** Open model `all-MiniLM-L6-v2` (mocked in tests).
*   **Parsers:** Both ADI (required) and Docling (bonus) supported.

---

## 5. What I Would Tackle Next

1.  **Queue-Based Async:** Move ingestion to a background task queue (Celery/RQ) to decouple from HTTP timeouts.
2.  **OCR for Scanned PDFs:** Add an AWS Textract/Tesseract pre-processing stage.
3.  **Hybrid Search:** Add BM25 sparse vectors to Qdrant to improve keyword matching (e.g. policy numbers).
4.  **True Tokenizer:** Swap the 800-character approximation for HuggingFace `tokenizers` exact counts.
5.  **Observability:** Add Prometheus counters for ingestion latency and structured JSON logging.
6.  **Document Deletion:** Add a `DELETE /documents/{document_id}` API using Qdrant filters.
7.  **Chunk Overlap:** Implement a 50-token overlap between chunks to preserve context at boundaries.