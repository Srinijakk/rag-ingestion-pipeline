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

Each stage catches specific operational exceptions only — not bare
`except Exception`. This is an important distinction:

```
Expected operational errors (malformed JSON, network timeout, OOM)
        → captured as structured IngestResult with status="failed"

Unexpected programming errors (AttributeError, logic bugs)
        → allowed to propagate to FastAPI's error handler → HTTP 500
```

Swallowing all exceptions with `except Exception` would silently convert
programming bugs into normal ingestion failures — making them invisible
in production logs. Specific exception types per stage ensure that only
genuine operational failures are handled gracefully.

---

## 2. Primary Design Driver

The assignment states:

> *"A malformed chunk or a missed table can produce a legally significant wrong answer."*

This single sentence drove every major decision in this pipeline. Enterprise
customers upload insurance benefit books, regulatory filings, and clinical
protocols. A wrong answer in these domains has legal and financial
consequences.

Every architectural decision below traces back to this constraint:
**correctness of chunk boundaries is more important than simplicity.**

---

## 3. Key Architectural Decisions

### 3.1 Parser Adapter Pattern → Canonical ParsedDocument

**Decision:** ADI and Docling parsers both implement `BaseParser` and produce
a normalized `ParsedDocument` (list of `RawBlock`). Nothing downstream knows
or cares which parser was used.

**Why:**
Parser output formats change frequently. ADI schemas evolve; teams adopt new
parsers (Docling, Unstructured, LlamaParse). Without the adapter, a parser
change cascades into chunking, embedding, and storage code.

With the adapter, adding a third parser is one new file with zero changes
elsewhere. The chunker, embedder, and vector store are completely decoupled
from parser specifics.

**`document_type` vs. parser detection — orthogonal dimensions:**

`document_type` is a **business category** — "insurance", "clinical_protocol",
"regulatory_filing". It describes what kind of document the business is
processing, and is stored in every chunk's Qdrant payload so the retrieval
layer can filter by document category.

Parser detection is **automatic** from the JSON structure:
- `schema_name: "DoclingDocument"` → Docling adapter
- `paragraphs` / `content` keys → ADI adapter

The same insurance document could be parsed by ADI or Docling. These are
two independent dimensions:

```
document_type = "insurance"     ← business category (stored in payload)
parser format = ADI or Docling  ← auto-detected from JSON (never exposed to caller)
```

**Trade-off considered:**
We could parse directly in the pipeline function — simpler initially but
creates tight coupling. The adapter boundary makes each layer independently
testable and replaceable.

---

### 3.2 Table Chunking — Row Boundary, Not Atomic

**Decision:** Tables split only at row boundaries. Never mid-row. Header
row is repeated in every chunk so each chunk is self-contained.

**Why not "one table = one chunk":**
A 100-row insurance benefit table as one chunk would exceed the embedding
model's context window. The embedding of a 5,000-token input is meaningless
— the model truncates or produces a degraded vector. The retrieval quality
collapses entirely for large tables.

**Why not arbitrary character split:**
Splitting mid-row separates `$20 copay` from `Primary Care Visit`. The
downstream answer-generation model sees `$20 copay` with no service context
and produces a wrong answer. In an insurance context, that is a legally
significant error.

**The correct rule:**
> Never split a table row. Split only between rows when accumulated
> size exceeds MAX_CHARS. Repeat the header in every chunk.

```
Chunk 1:                       Chunk 2:
| Service      | Cost |        | Service      | Cost |
| ---          | ---  |        | ---          | ---  |
| PCP Visit    | $20  |        | Specialist   | $40  |
| ER Visit     | $150 |        | Urgent Care  | $50  |
```

Every chunk knows its column names. No chunk is ambiguous in isolation.

**Trade-off considered:**
Repeating the header row increases storage slightly. That cost is trivially
small compared to the retrieval quality improvement.

---

### 3.3 Section Heading Propagation

**Decision:** Headings are NOT emitted as standalone chunks. Instead they
update a running `current_heading` variable that is attached as
`section_heading` metadata to every subsequent chunk until the next
heading appears.

**Why not heading chunks:**
A heading-only chunk ("Annual Deductible") has almost no semantic content
— it will match too broadly against any query mentioning deductibles,
regardless of context. Worse, it gives the retrieval layer no information
about what content follows.

**Why metadata propagation:**
Every content chunk carries `section_heading: "Annual Deductible"` in its
payload. This allows:
- The retrieval layer to filter chunks by section
- The answer-generation model to always know which section a chunk came from
- Accurate attribution in generated answers

**Trade-off considered:**
We could embed headings as separate chunks and rely on the model to reason
about document proximity. This is simpler but shifts correctness burden
onto the model rather than encoding structure explicitly.

---

### 3.4 Dynamic, Structure-Aware Chunking — Not Fixed-Size

**Decision:** Chunks are variable size. `MAX_CHARS = 800` (~200 tokens) is
a configurable safety ceiling — not a target. Structure determines chunk
boundaries; the character limit is the constraint of last resort.

**Why 800 chars (~200 tokens) — not 512:**
`all-MiniLM-L6-v2` has `max_seq_length = 256` word pieces. Inputs longer
than 256 word pieces are silently truncated. A chunk of 480 "tokens" by
naive approximation could easily exceed 256 real word pieces — the model
would silently discard half the content. This is exactly the kind of
invisible correctness failure this pipeline must prevent.

We target ~200 tokens (800 chars), leaving a safety margin for:
- Tokenizer differences (word pieces ≠ words)
- Special tokens added by the model (`[CLS]`, `[SEP]`)
- Variation in character-to-token ratio across document types

**The limit is chosen from the model's actual constraint, not a generic
"512 is standard" assumption.**

**Our algorithm:**
```
1. Identify structural type of next block
   (heading / text paragraph / table / figure)

2. Headings → update section_heading context, emit no chunk

3. Tables → accumulate rows greedily
   Adding next row within MAX_CHARS → add it
   Adding next row exceeds MAX_CHARS → flush chunk, repeat header, continue

4. Text → accumulate sentences greedily
   Adding next sentence within MAX_CHARS → add it
   Adding next sentence exceeds MAX_CHARS → flush chunk, continue

5. Every chunk carries: section_heading, page_number, block_type

6. Result: variable-size chunks — e.g. 320, 470, 290, 600 chars
   All within the ceiling. None forced to be exactly MAX_CHARS.
```

**Boundary priority:**
```
Semantic boundary (section / block type)
              ↓
    Sentence boundary (. ? !)
              ↓
    Character ceiling (MAX_CHARS = 800)
```

**Trade-off considered:**
Using the model's actual tokenizer would give exact word-piece counts.
We use character approximation to avoid importing sentence-transformers
into the chunker module — keeping chunker independently testable. In
production, replace `len(text)` with `len(tokenizer.encode(text))` in
`_split_text()` — the interface is unchanged.

---

### 3.5 Tenant Isolation — Payload Filter, Not Separate Collections

**Decision:** Single Qdrant collection `"documents"`. Tenant isolation via
`tenant_id` + `collection_id` payload filters on every vector point.

**Why not separate collections per tenant:**
The assignment specifies hundreds of enterprise tenants and millions of
documents. At that scale, one collection per tenant creates serious
operational overhead — each collection maintains its own HNSW index,
memory segments, WAL, and monitoring surface. Managing thousands of
collections is operationally fragile and expensive.

**Why payload filter:**
- Single collection to monitor, back up, and scale horizontally
- Payload indexes on `tenant_id`, `collection_id`, `document_id` are
  created at collection setup — Qdrant uses these to optimize filtered
  queries without scanning all vectors
- GDPR deletion: `delete_by_filter(tenant_id=X)` with no collection drop
- Simpler application-layer access control

**Trade-off acknowledged:**
For tenants with strict regulatory isolation requirements (healthcare,
government contracts), dedicated collections or separate Qdrant deployments
may be contractually required. The `BaseVectorStore` abstraction makes
this a one-file change — no pipeline code changes needed.

---

### 3.6 Idempotent Ingestion — Collision-Safe Point IDs

**Decision:**
```
Point ID = uuid5(tenant_id + collection_id + document_id + chunk_index)
```

**Why include tenant_id:**
Two tenants using the same `document_id` string (both use `"doc-001"`)
would produce identical IDs without tenant scoping — one tenant's data
would silently overwrite another's. Including `tenant_id` makes every
point ID globally unique and collision-proof.

**Why uuid5 (deterministic):**
Re-ingesting the same document produces the same point IDs. Qdrant upsert
on an existing ID = overwrite. No duplicate vectors, no cleanup logic,
safe to retry any failed ingestion job.

**Known limitation — document versioning:**
If a document is updated and the new version has fewer chunks than the
original (e.g. version 1 had 20 chunks, version 2 has 18), chunks 1–18
are overwritten correctly but old chunks 19–20 remain in the store.
Current idempotency covers repeated ingestion of the same document
version. Version-aware replacement (delete old chunks, ingest new) is
future work — see Section 5.6.

**Trade-off considered:**
Random UUIDs (uuid4) are simpler but break idempotency — every re-ingest
creates duplicates that silently degrade retrieval quality.

---

### 3.7 Async Pipeline — Ready for Scale

**Decision:** The public ingestion interface is `async def` to support
non-blocking HTTP handling and vector-store I/O.

**What async actually helps here:**
- FastAPI request handling — non-blocking, concurrent HTTP requests
- Qdrant upsert — network I/O, genuinely benefits from async
- File reading — I/O bound

**What async does NOT help here:**
Local embedding inference with `sentence-transformers` is CPU-bound, not
I/O-bound. Writing `async def embed()` would not make CPU inference
non-blocking. The `embed()` method is intentionally synchronous.

For high-concurrency production deployments, embedding computation should
be isolated into worker processes or a dedicated embedding service
(e.g. Triton Inference Server). The pipeline is separated from the HTTP
layer specifically so it can be invoked by a task queue (Celery/RQ)
without changing the core parsing, chunking, embedding, or storage
interfaces.

**Trade-off considered:**
Async adds slight complexity in tests. We use
`asyncio.get_event_loop().run_until_complete()` — no pytest-asyncio
dependency required.

---

### 3.8 Dependency Injection for Embedder and Vector Store

**Decision:** `ingest()` accepts `embedder` and `vector_store` as optional
injected parameters. Production defaults are used when not provided.

**Why:**
- Tests inject `MockEmbedder` + `MockVectorStore` — no Qdrant process,
  no model download, instant CI
- Swapping `sentence-transformers` for OpenAI is a one-line change
- No `unittest.mock.patch` magic — tests use real mock class instances,
  making tests readable and non-brittle
- Each mock captures all upserted points for assertion

---

### 3.9 Embedding Model Choice

**Decision:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions,
runs on CPU, fully offline).

**Why:**
- Fully offline — no data leaves the instance (critical for insurance and
  healthcare regulated environments)
- No API key, no cost, no rate limits, no external dependency
- Strong performance on semantic similarity benchmarks for English text
- 384 dimensions: compact Qdrant storage, fast HNSW retrieval

**Trade-off:** OpenAI `text-embedding-3-small` gives higher retrieval
quality but introduces an external API dependency, per-token cost, network
latency, and data egress risk. For a regulated-industry RAG platform,
local embeddings are the safer production default.

---

### 3.10 Figure Handling — Explicit Limitation

**Current:** Figure captions are indexed as text chunks. A figure without
a caption produces no chunk.

**Why caption-only:**
The parser outputs (ADI and Docling) both expose figure captions as
structured text fields. Indexing the caption gives the retrieval layer
a meaningful text handle for figures that do have descriptions.

**Explicit limitation:**
Figures without captions are not indexed. Visual content (charts,
diagrams) is not embedded — only caption text is.

**Future:** A multimodal pipeline could generate image embeddings (CLIP)
or visual descriptions (GPT-4V / Claude Vision) for uncaptioned figures.
The `RawBlock` dataclass already supports adding an `image_bytes` field
without any downstream changes.

---

## 4. Constraints Satisfied

| Constraint | How Satisfied |
|---|---|
| Language: Python | Python 3.11, FastAPI, async throughout |
| Vector store: open-source | Qdrant (Apache 2.0), runs via Docker |
| Embeddings: open model | `sentence-transformers/all-MiniLM-L6-v2` |
| Mock in tests | `MockEmbedder`, `MockVectorStore` — zero real dependencies in tests |
| Both parser adapters | ADI (required) + Docling (bonus) both implemented |
| Production scale design | Single collection, payload filter, async, idempotent UUIDs |

---

## 5. What I Would Tackle Next

### 5.1 Queue-Based Async Processing
Move from synchronous HTTP ingestion to a task queue (Celery + Redis or
RQ). The endpoint returns a `job_id` immediately; clients poll for
completion. This decouples ingestion latency from HTTP response time and
enables horizontal worker scaling across machines.

### 5.2 OCR for Scanned Documents
The current pipeline assumes the parser has already extracted text from
the document. For scanned PDFs (very common in insurance and legal
workflows), an OCR stage (Tesseract or AWS Textract) needs to run before
parsing. The `BaseParser` interface already accommodates this — add a
pre-processing step before `parser.parse()`.

### 5.3 Hybrid Search (BM25 + Dense)
Dense vector search alone misses exact keyword matches — policy numbers,
drug names, ICD codes, specific dollar amounts. Production RAG systems for
enterprise use BM25 + dense hybrid search with cross-encoder re-ranking.
Qdrant supports sparse vectors natively for hybrid search — this is the
highest-impact retrieval quality improvement after the current baseline.

### 5.4 Proper Tokenizer for Chunking
Replace the character-count approximation (`1 token ≈ 4 chars`) with the
embedding model's actual tokenizer from HuggingFace `tokenizers`. This
guarantees chunks never exceed the model's true context window and gives
accurate overlap calculations. The `_split_text()` function interface is
unchanged — this is a drop-in replacement.

### 5.5 Observability
Add structured JSON logging per ingestion event:
`{document_id, tenant_id, stage, duration_ms, chunk_count, status}`.
Add Prometheus counters for ingestion success/failure rates and p95
latency per pipeline stage. Without these, debugging production failures
at scale across hundreds of tenants is extremely difficult.

### 5.6 Document Deletion and Versioning
Add `DELETE /documents/{document_id}` using Qdrant's `delete_by_filter`
for GDPR compliance. Add `document_version` to point IDs for clean
version management — re-ingesting an updated document creates new points
while old version points can be deleted atomically.

### 5.7 Chunk Overlap
Currently there is no overlap between consecutive text chunks. For long
paragraphs split at sentence boundaries, context at the boundary can be
lost. Adding a configurable `CHUNK_OVERLAP` (e.g. 50 tokens) where the
last sentence of chunk N is repeated as the first sentence of chunk N+1
improves retrieval for queries that span chunk boundaries.