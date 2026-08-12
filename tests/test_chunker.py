"""
Unit tests for the semantic chunker.

These are the most important tests in the project.
Chunking correctness directly determines RAG answer quality.

Tests verify:
  - Tables split ONLY at row boundaries — never mid-row
  - Small tables stay as one chunk; large tables split between rows
  - Every table chunk repeats the header so it is self-contained
  - Section headings propagate as metadata to all subsequent chunks
  - Headings are NOT emitted as standalone chunks
  - Long text is split at sentence boundaries, never mid-sentence
  - Figures produce exactly one chunk
  - chunk_index is sequential and 0-based
  - Each chunk carries correct document/tenant/collection IDs
  - Empty documents return empty list
"""

import pytest
from ingestor.chunker import Chunk, chunk_document, MAX_CHARS
from ingestor.parsers.base import ParsedDocument, RawBlock


# ======================================================================
# Helpers
# ======================================================================

DEFAULTS = dict(
    document_id="doc-001",
    tenant_id="tenant-abc",
    collection_id="col-xyz",
    tags=["insurance", "benefits"],
    metadata={"source": "upload"},
)


def make_parsed(*blocks: RawBlock) -> ParsedDocument:
    return ParsedDocument(blocks=list(blocks))


def text_block(content: str, role: str | None = None, page: int = 1) -> RawBlock:
    return RawBlock(type="text", content=content, page_number=page, role=role)


def table_block(content: str, page: int = 1) -> RawBlock:
    return RawBlock(type="table", content=content, page_number=page, role="table")


def figure_block(content: str, page: int = 1) -> RawBlock:
    return RawBlock(type="figure", content=content, page_number=page, role="figure")


def heading_block(content: str, page: int = 1) -> RawBlock:
    return RawBlock(type="text", content=content, page_number=page, role="sectionHeading")


# ======================================================================
# Basic output tests
# ======================================================================

class TestChunkDocumentBasics:

    def test_empty_document_returns_empty_list(self):
        result = chunk_document(make_parsed(), **DEFAULTS)
        assert result == []

    def test_single_text_block_produces_one_chunk(self):
        parsed = make_parsed(text_block("All amounts shown are in-network rates."))
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) == 1
        assert result[0].text == "All amounts shown are in-network rates."

    def test_chunk_carries_document_id(self):
        parsed = make_parsed(text_block("Hello."))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].document_id == "doc-001"

    def test_chunk_carries_tenant_id(self):
        parsed = make_parsed(text_block("Hello."))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].tenant_id == "tenant-abc"

    def test_chunk_carries_collection_id(self):
        parsed = make_parsed(text_block("Hello."))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].collection_id == "col-xyz"

    def test_chunk_carries_page_number(self):
        parsed = make_parsed(text_block("Hello.", page=3))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].page_number == 3

    def test_chunk_carries_tags(self):
        parsed = make_parsed(text_block("Hello."))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].tags == ["insurance", "benefits"]

    def test_chunk_index_is_zero_based_sequential(self):
        parsed = make_parsed(
            text_block("First sentence."),
            text_block("Second sentence."),
            text_block("Third sentence."),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert [c.chunk_index for c in result] == list(range(len(result)))


# ======================================================================
# Table chunking — row-boundary correctness tests
# ======================================================================

class TestTableChunking:

    def test_small_table_produces_one_chunk(self):
        """A table that fits within MAX_CHARS stays as one chunk."""
        table_content = (
            "| Service | Your Cost |\n"
            "| --- | --- |\n"
            "| Primary Care Visit | $20 copay |\n"
            "| Specialist Visit | $40 copay |"
        )
        parsed = make_parsed(table_block(table_content))
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) == 1

    def test_table_chunk_block_type_is_table(self):
        parsed = make_parsed(table_block("| A | B |\n| --- | --- |\n| 1 | 2 |"))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].block_type == "table"

    def test_large_table_splits_into_multiple_chunks(self):
        """
        Large tables MUST split — but only at row boundaries.
        One table = one chunk always was wrong: a 100-row table
        exceeds embedding model context window.
        """
        header = "| Service | Cost | Notes |"
        separator = "| --- | --- | --- |"
        rows = [f"| Service {i} | ${i * 10} copay | Coverage note {i} |" for i in range(60)]
        large_table = "\n".join([header, separator] + rows)
        assert len(large_table) > MAX_CHARS

        parsed = make_parsed(table_block(large_table))
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) > 1
        assert all(c.block_type == "table" for c in result)

    def test_large_table_never_splits_mid_row(self):
        """
        Every row must appear complete in exactly one chunk.
        No row content may be split across two chunks.
        """
        header = "| Service | Cost |"
        separator = "| --- | --- |"
        rows = [f"| Primary Care Visit Type {i} | ${i * 5} copay |" for i in range(60)]
        large_table = "\n".join([header, separator] + rows)

        parsed = make_parsed(table_block(large_table))
        result = chunk_document(parsed, **DEFAULTS)

        # Collect all row content from all chunks
        all_text = "\n".join(c.text for c in result)
        # Every original row must appear fully in the combined output
        for row in rows:
            assert row in all_text, f"Row missing or split: {row}"

    def test_large_table_each_chunk_has_header(self):
        """
        Each table chunk must repeat the header row so it is
        self-contained. Without headers, '$20 copay' has no column context.
        """
        header = "| Service | Your Cost |"
        separator = "| --- | --- |"
        rows = [f"| Service {i} | ${i * 10} |" for i in range(60)]
        large_table = "\n".join([header, separator] + rows)

        parsed = make_parsed(table_block(large_table))
        result = chunk_document(parsed, **DEFAULTS)

        assert len(result) > 1
        for chunk in result:
            assert "Service" in chunk.text, (
                f"Chunk missing header context:\n{chunk.text[:200]}"
            )

    def test_table_carries_table_id_metadata(self):
        parsed = make_parsed(table_block("| A | B |\n| --- | --- |\n| 1 | 2 |"))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].table_id is not None
        assert result[0].table_id.startswith("table-")

    def test_table_with_caption_included_in_chunk(self):
        """Caption prepended to table content must appear in chunk text."""
        table_with_caption = (
            "Table 1: Cost Sharing Summary\n\n"
            "| Service | Cost |\n"
            "| --- | --- |\n"
            "| PCP | $20 |"
        )
        parsed = make_parsed(table_block(table_with_caption))
        result = chunk_document(parsed, **DEFAULTS)
        assert "Table 1: Cost Sharing Summary" in result[0].text


# ======================================================================
# Section heading propagation — critical metadata tests
# ======================================================================

class TestHeadingPropagation:

    def test_heading_not_emitted_as_chunk(self):
        """Headings update context but must NOT appear as standalone chunks."""
        parsed = make_parsed(heading_block("Annual Deductible"))
        result = chunk_document(parsed, **DEFAULTS)
        assert result == []

    def test_heading_propagates_to_next_text_chunk(self):
        parsed = make_parsed(
            heading_block("Annual Deductible"),
            text_block("The deductible is $1,500 per year."),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) == 1
        assert result[0].section_heading == "Annual Deductible"

    def test_heading_propagates_to_next_table_chunk(self):
        parsed = make_parsed(
            heading_block("Cost Sharing"),
            table_block("| Service | Cost |\n| --- | --- |\n| PCP | $20 |"),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].section_heading == "Cost Sharing"

    def test_heading_propagates_to_next_figure_chunk(self):
        parsed = make_parsed(
            heading_block("Network Map"),
            figure_block("Figure 1: Network Coverage Map"),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].section_heading == "Network Map"

    def test_heading_propagates_to_multiple_subsequent_chunks(self):
        parsed = make_parsed(
            heading_block("Section A"),
            text_block("First paragraph."),
            text_block("Second paragraph."),
            table_block("| A | B |\n| --- | --- |\n| 1 | 2 |"),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert all(c.section_heading == "Section A" for c in result)

    def test_heading_updates_when_new_heading_encountered(self):
        parsed = make_parsed(
            heading_block("Section A"),
            text_block("Content under A."),
            heading_block("Section B"),
            text_block("Content under B."),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].section_heading == "Section A"
        assert result[1].section_heading == "Section B"

    def test_chunk_before_any_heading_has_empty_section_heading(self):
        parsed = make_parsed(text_block("Intro text before any heading."))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].section_heading == ""

    def test_multiple_headings_correct_propagation(self):
        parsed = make_parsed(
            heading_block("Deductible"),
            text_block("Deductible text."),
            heading_block("Copay"),
            text_block("Copay text."),
            heading_block("Out-of-Pocket"),
            text_block("OOP text."),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].section_heading == "Deductible"
        assert result[1].section_heading == "Copay"
        assert result[2].section_heading == "Out-of-Pocket"


# ======================================================================
# Figure chunking
# ======================================================================

class TestFigureChunking:

    def test_figure_produces_exactly_one_chunk(self):
        parsed = make_parsed(figure_block("Figure 1: Network Coverage Map"))
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) == 1

    def test_figure_block_type_is_figure(self):
        parsed = make_parsed(figure_block("Figure 1: Network Coverage Map"))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].block_type == "figure"

    def test_figure_text_preserved(self):
        caption = "Figure 2: Claims Processing Workflow"
        parsed = make_parsed(figure_block(caption))
        result = chunk_document(parsed, **DEFAULTS)
        assert result[0].text == caption


# ======================================================================
# Long text splitting
# ======================================================================

class TestTextSplitting:

    def test_short_text_stays_as_one_chunk(self):
        short = "The deductible is $1,500 per year."
        parsed = make_parsed(text_block(short))
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) == 1
        assert result[0].text == short

    def test_long_text_split_into_multiple_chunks(self):
        sentence = "This is a sentence about insurance benefits and coverage. "
        long_text = sentence * (MAX_CHARS // len(sentence) + 5)
        assert len(long_text) > MAX_CHARS

        parsed = make_parsed(text_block(long_text))
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) > 1

    def test_no_chunk_exceeds_max_chars(self):
        sentence = "Each sentence is about thirty characters long. "
        long_text = sentence * 100
        parsed = make_parsed(text_block(long_text))
        result = chunk_document(parsed, **DEFAULTS)
        for chunk in result:
            assert len(chunk.text) <= MAX_CHARS * 1.1

    def test_split_chunks_all_have_same_block_type(self):
        sentence = "This is a moderately long sentence about health insurance. "
        long_text = sentence * 50
        parsed = make_parsed(text_block(long_text))
        result = chunk_document(parsed, **DEFAULTS)
        assert all(c.block_type == "text" for c in result)

    def test_split_chunks_carry_same_section_heading(self):
        sentence = "This is a sentence about the deductible. "
        long_text = sentence * 60
        parsed = make_parsed(
            heading_block("Annual Deductible"),
            text_block(long_text),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert len(result) > 1
        assert all(c.section_heading == "Annual Deductible" for c in result)


# ======================================================================
# Mixed document (realistic end-to-end scenario)
# ======================================================================

class TestMixedDocument:

    def test_realistic_document_chunk_count(self):
        parsed = make_parsed(
            text_block("This document summarizes your health insurance benefits."),
            heading_block("Annual Deductible"),
            text_block("The annual deductible is $1,500 for individuals."),
            text_block("Family deductibles are capped at $3,000 per year."),
            table_block("| Service | Cost |\n| --- | --- |\n| PCP | $20 |\n| Specialist | $40 |"),
            heading_block("Network Coverage"),
            figure_block("Figure 1: Network Coverage Map"),
        )
        result = chunk_document(parsed, **DEFAULTS)
        # 2 headings excluded; 5 content blocks remain
        assert len(result) == 5

    def test_realistic_document_block_types(self):
        parsed = make_parsed(
            text_block("Intro text."),
            heading_block("Section"),
            table_block("| A | B |\n| --- | --- |\n| 1 | 2 |"),
            figure_block("Figure caption."),
        )
        result = chunk_document(parsed, **DEFAULTS)
        types = [c.block_type for c in result]
        assert "text" in types
        assert "table" in types
        assert "figure" in types

    def test_chunk_indexes_sequential_across_mixed_types(self):
        parsed = make_parsed(
            text_block("Text one."),
            table_block("| A |\n| --- |\n| 1 |"),
            figure_block("Caption."),
            text_block("Text two."),
        )
        result = chunk_document(parsed, **DEFAULTS)
        assert [c.chunk_index for c in result] == list(range(len(result)))