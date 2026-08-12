"""
Semantic chunker — converts ParsedDocument blocks into embeddable Chunks.

Chunking rules (critical for RAG correctness):

  1. Tables  → Split ONLY at row boundaries, never mid-row.
               A split table cell produces a legally wrong answer.
               e.g. "$20 copay" separated from "Primary Care Visit" = wrong retrieval.
               But a 100-row table as one chunk exceeds embedding model limits.
               Rule: accumulate rows until MAX_CHARS; flush at row boundary.

  2. Figures → Always one chunk (caption text only).

  3. Text    → Split at sentence boundaries to stay within MAX_CHARS.
               Never cut mid-sentence.

  4. Headings → NOT emitted as chunks. Update running context.
                Every subsequent chunk carries section_heading as metadata.
                This lets the retrieval layer filter by section without
                polluting the vector space with low-signal heading chunks.
"""

import re
from dataclasses import dataclass, field

from .parsers.base import ParsedDocument, RawBlock

# ------------------------------------------------------------------
# Configuration — tune per deployment by changing MAX_CHARS only
# ------------------------------------------------------------------


MAX_CHARS: int = 800  # ~200 tokens — safe ceiling for all-MiniLM-L6-v2


@dataclass
class Chunk:
    """A single embeddable unit — ready for embedding and vector store upsert."""
    text: str
    document_id: str
    tenant_id: str
    collection_id: str
    page_number: int
    block_type: str           # "text" | "table" | "figure"
    section_heading: str      # running heading at point of this chunk
    chunk_index: int          # 0-based position within document
    table_id: str | None = None      # set for table chunks
    table_caption: str | None = None # set for table chunks
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def chunk_document(
    parsed: ParsedDocument,
    document_id: str,
    tenant_id: str,
    collection_id: str,
    tags: list[str],
    metadata: dict,
) -> list[Chunk]:
    """
    Convert a ParsedDocument into an ordered list of Chunks.

    This is the core of the ingestion pipeline — correctness here
    directly determines RAG answer quality.
    """
    chunks: list[Chunk] = []
    current_heading: str = ""
    chunk_index: int = 0
    table_counter: int = 0

    for block in parsed.blocks:

        # Headings update context — NOT emitted as chunks.
        if block.role == "sectionHeading":
            current_heading = block.content
            continue

        if block.type == "table":
            # Split table at row boundaries when needed.
            # Never split mid-row — that's the correctness guarantee.
            table_id = f"table-{table_counter}"
            table_counter += 1

            # Extract caption from content (first line if it's a caption)
            content_lines = block.content.split("\n")
            table_caption = ""
            table_body_lines = content_lines

            # If first line has no pipe, it's a prepended caption
            if content_lines and "|" not in content_lines[0]:
                table_caption = content_lines[0].strip()
                table_body_lines = content_lines[1:]

            row_chunks = _split_table_by_rows(
                lines=table_body_lines,
                caption=table_caption,
                block=block,
                table_id=table_id,
                section_heading=current_heading,
                chunk_index_start=chunk_index,
                document_id=document_id,
                tenant_id=tenant_id,
                collection_id=collection_id,
                tags=tags,
                metadata=metadata,
            )
            chunks.extend(row_chunks)
            chunk_index += len(row_chunks)

        elif block.type == "figure":
            # Figures are always atomic — caption text only.
            chunks.append(_build_chunk(
                text=block.content,
                block_type="figure",
                page_number=block.page_number,
                section_heading=current_heading,
                chunk_index=chunk_index,
                document_id=document_id,
                tenant_id=tenant_id,
                collection_id=collection_id,
                tags=tags,
                metadata=metadata,
            ))
            chunk_index += 1

        else:
            # Text — split at sentence boundaries if over limit.
            for segment in _split_text(block.content):
                chunks.append(_build_chunk(
                    text=segment,
                    block_type="text",
                    page_number=block.page_number,
                    section_heading=current_heading,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    tags=tags,
                    metadata=metadata,
                ))
                chunk_index += 1

    return chunks


# ------------------------------------------------------------------
# Table chunking — row-boundary aware
# ------------------------------------------------------------------

def _split_table_by_rows(
    lines: list[str],
    caption: str,
    block: RawBlock,
    table_id: str,
    section_heading: str,
    chunk_index_start: int,
    document_id: str,
    tenant_id: str,
    collection_id: str,
    tags: list[str],
    metadata: dict,
) -> list[Chunk]:
    """
    Split a markdown table into chunks at row boundaries.

    Rules:
      - Header row (first row) is always included in every chunk
        so each chunk is self-contained and interpretable.
      - Separator row (---) is always included after header.
      - Data rows are accumulated until MAX_CHARS; flush at boundary.
      - If a single row exceeds MAX_CHARS, it gets its own chunk.

    This ensures: no row is ever split, every chunk has column context.
    """
    if not lines:
        return []

    # Identify header, separator, and data rows
    header_lines: list[str] = []
    data_rows: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not header_lines:
            header_lines.append(line)       # first row = header
        elif set(stripped.replace("|", "").replace("-", "").replace(" ", "")) == set():
            header_lines.append(line)       # separator row
        else:
            data_rows.append(line)          # data row

    header_text = "\n".join(header_lines)
    header_len = len(header_text)

    # If entire table fits, one chunk
    full_table = (caption + "\n\n" if caption else "") + "\n".join(lines)
    if len(full_table) <= MAX_CHARS:
        return [_build_chunk(
            text=full_table,
            block_type="table",
            page_number=block.page_number,
            section_heading=section_heading,
            chunk_index=chunk_index_start,
            document_id=document_id,
            tenant_id=tenant_id,
            collection_id=collection_id,
            tags=tags,
            metadata={**metadata, "table_id": table_id, "table_caption": caption},
        )]

    # Large table — split at row boundaries
    chunks: list[Chunk] = []
    current_rows: list[str] = []
    current_len: int = header_len

    for row in data_rows:
        row_len = len(row) + 1  # +1 for newline

        if current_len + row_len > MAX_CHARS and current_rows:
            # Flush current group — prepend header so chunk is self-contained
            chunk_text = _assemble_table_chunk(caption, header_text, current_rows)
            chunks.append(_build_chunk(
                text=chunk_text,
                block_type="table",
                page_number=block.page_number,
                section_heading=section_heading,
                chunk_index=chunk_index_start + len(chunks),
                document_id=document_id,
                tenant_id=tenant_id,
                collection_id=collection_id,
                tags=tags,
                metadata={**metadata, "table_id": table_id, "table_caption": caption},
            ))
            current_rows = []
            current_len = header_len

        current_rows.append(row)
        current_len += row_len

    # Flush remaining rows
    if current_rows:
        chunk_text = _assemble_table_chunk(caption, header_text, current_rows)
        chunks.append(_build_chunk(
            text=chunk_text,
            block_type="table",
            page_number=block.page_number,
            section_heading=section_heading,
            chunk_index=chunk_index_start + len(chunks),
            document_id=document_id,
            tenant_id=tenant_id,
            collection_id=collection_id,
            tags=tags,
            metadata={**metadata, "table_id": table_id, "table_caption": caption},
        ))

    return chunks if chunks else [_build_chunk(
        text=full_table,
        block_type="table",
        page_number=block.page_number,
        section_heading=section_heading,
        chunk_index=chunk_index_start,
        document_id=document_id,
        tenant_id=tenant_id,
        collection_id=collection_id,
        tags=tags,
        metadata={**metadata, "table_id": table_id, "table_caption": caption},
    )]


def _assemble_table_chunk(caption: str, header: str, rows: list[str]) -> str:
    """Assemble a self-contained table chunk with caption + header + rows."""
    parts = []
    if caption:
        parts.append(caption)
        parts.append("")  # blank line
    parts.append(header)
    parts.extend(rows)
    return "\n".join(parts)


# ------------------------------------------------------------------
# Text splitting — sentence boundary aware
# ------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """
    Split text at sentence boundaries to stay within MAX_CHARS.

    Structure first, size second:
      1. Split on sentence terminators (. ? !)
      2. Accumulate greedily until limit
      3. Flush at sentence boundary — never mid-sentence
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= MAX_CHARS:
        return [text]

    sentences = _tokenize_sentences(text)
    segments: list[str] = []
    current: list[str] = []
    current_len: int = 0

    for sentence in sentences:
        s_len = len(sentence)

        # Single oversized sentence — hard split as last resort
        if s_len > MAX_CHARS:
            if current:
                segments.append(" ".join(current).strip())
                current, current_len = [], 0
            for i in range(0, s_len, MAX_CHARS):
                segments.append(sentence[i:i + MAX_CHARS])
            continue

        if current_len + s_len > MAX_CHARS and current:
            segments.append(" ".join(current).strip())
            current, current_len = [], 0

        current.append(sentence)
        current_len += s_len + 1

    if current:
        segments.append(" ".join(current).strip())

    return [s for s in segments if s]


def _tokenize_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ------------------------------------------------------------------
# Chunk builder
# ------------------------------------------------------------------

def _build_chunk(
    text: str,
    block_type: str,
    page_number: int,
    section_heading: str,
    chunk_index: int,
    document_id: str,
    tenant_id: str,
    collection_id: str,
    tags: list[str],
    metadata: dict,
) -> Chunk:
    return Chunk(
        text=text,
        document_id=document_id,
        tenant_id=tenant_id,
        collection_id=collection_id,
        page_number=page_number,
        block_type=block_type,
        section_heading=section_heading,
        chunk_index=chunk_index,
        table_id=metadata.get("table_id"),
        table_caption=metadata.get("table_caption"),
        tags=list(tags),
        metadata=dict(metadata),
    )
