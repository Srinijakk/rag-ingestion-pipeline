# RAG Ingestion Pipeline

A production-grade document ingestion pipeline for enterprise RAG systems.
Ingests parsed documents (ADI or Docling format), chunks them semantically,
embeds them, and stores vectors in Qdrant — with full multi-tenant isolation.

---

## Project Structure

```
rag-ingestion-pipeline/
├── app.py                     # FastAPI HTTP entry point
├── docker-compose.yml         # Qdrant service
├── requirements.txt
├── DESIGN.md                  # Architecture decisions and trade-offs
├── ingestor/
│   ├── pipeline.py            # ingest() — main orchestrator
│   ├── chunker.py             # Semantic chunker
│   ├── embedder.py            # Embedding interface + implementations
│   ├── vector_store.py        # Qdrant wrapper + mock
│   └── parsers/
│       ├── base.py            # BaseParser interface + data models
│       ├── adi.py             # Azure Document Intelligence adapter
│       └── docling.py         # Docling adapter
└── tests/
    ├── test_parsers.py
    ├── test_chunker.py
    └── test_pipeline.py
```

---

## Setup and Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-ingestion-pipeline.git
cd rag-ingestion-pipeline

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant (using Docker)
docker-compose up -d

# 5. Start the ingestor
uvicorn app:app --reload --port 8000

# 6. Check it's running
curl http://localhost:8000/healthz
# → {"status": "ok"}
```

---

## Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_adi.json" \
  -F "document_id=doc-001" \
  -F "tenant_id=acme-corp" \
  -F "collection_id=benefits-2024" \
  -F "document_type=insurance" \
  -F "tags=insurance,benefits" \
  -F 'metadata={"year":"2024","source":"upload"}'
```

**Response:**
```json
{
  "document_id": "doc-001",
  "tenant_id": "acme-corp",
  "collection_id": "benefits-2024",
  "document_type": "insurance",
  "chunks_ingested": 12,
  "status": "success",
  "error": null
}
```

---

## Run Tests

```bash
# Run all tests
pytest tests/ -v

```
## Supported Parser Formats

| Parser | Format | Auto-detected by |
|--------|--------|-----------------|
| Azure Document Intelligence (ADI) | `paragraphs`, `tables`, `figures` keys | JSON structure |
| Docling | `schema_name: "DoclingDocument"` | `schema_name` field |

Parser format is auto-detected from JSON structure. `document_type` is a business category (e.g. `insurance`, `clinical_protocol`), not a parser selector.

---

---

## Architecture

See [DESIGN.md](./DESIGN.md) for full architectural decisions, trade-offs, and what would be built next.
