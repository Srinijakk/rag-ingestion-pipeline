"""
Unit tests for ADI and Docling parser adapters.

Tests verify:
  - Correct RawBlock type, content, page_number, role extraction
  - Table serialization to markdown
  - Caption prepended to table content
  - Figure caption extraction
  - Section heading role preserved
  - Empty / missing fields handled gracefully
  - Both parsers produce the same normalized ParsedDocument shape
"""

import pytest
from ingestor.parsers.adi import ADIParser
from ingestor.parsers.docling import DoclingParser
from ingestor.parsers.base import ParsedDocument, RawBlock


# ======================================================================
# Fixtures — sample parser JSON matching the assignment spec exactly
# ======================================================================

ADI_SAMPLE = {
    "paragraphs": [
        {
            "content": "All amounts shown are in-network rates.",
            "role": None,
            "bounding_regions": [{"page_number": 1, "polygon": [0.5, 0.7, 4.2, 0.9]}],
            "spans": [{"offset": 0, "length": 40}],
        },
        {
            "content": "Annual Deductible",
            "role": "sectionHeading",
            "bounding_regions": [{"page_number": 1, "polygon": [0.5, 1.0, 3.0, 1.2]}],
            "spans": [{"offset": 41, "length": 17}],
        },
    ],
    "tables": [
        {
            "row_count": 3,
            "column_count": 2,
            "cells": [
                {"row_index": 0, "column_index": 0, "content": "Service",          "kind": "columnHeader"},
                {"row_index": 0, "column_index": 1, "content": "Your Cost",        "kind": "columnHeader"},
                {"row_index": 1, "column_index": 0, "content": "Primary Care Visit","kind": "content"},
                {"row_index": 1, "column_index": 1, "content": "$20 copay",        "kind": "content"},
                {"row_index": 2, "column_index": 0, "content": "Specialist Visit", "kind": "content"},
                {"row_index": 2, "column_index": 1, "content": "$40 copay",        "kind": "content"},
            ],
            "bounding_regions": [{"page_number": 1}],
            "spans": [{"offset": 58, "length": 200}],
            "caption": {"content": "Table 1: Cost Sharing Summary"},
        }
    ],
    "figures": [
        {
            "bounding_regions": [{"page_number": 2, "polygon": [1.0, 3.0, 5.0, 6.5]}],
            "spans": [{"offset": 350, "length": 0}],
            "caption": {"content": "Figure 1: Network Coverage Map"},
        }
    ],
    "pages": [{"page_number": 1, "width": 8.5, "height": 11.0}],
    "content": "All amounts shown are in-network rates.\nAnnual Deductible\n...",
}

DOCLING_SAMPLE = {
    "schema_name": "DoclingDocument",
    "version": "1.3.0",
    "name": "benefit_summary",
    "origin": {
        "mimetype": "application/pdf",
        "binary_hash": "a3f5...",
        "filename": "benefit_summary.pdf",
    },
    "body": {
        "self_ref": "#/body",
        "children": [
            {"$ref": "#/texts/0"},
            {"$ref": "#/texts/1"},
            {"$ref": "#/tables/0"},
        ],
        "label": "unspecified",
    },
    "texts": [
        {
            "self_ref": "#/texts/0",
            "parent": {"$ref": "#/body"},
            "label": "section_header",
            "prov": [{"page_no": 1, "bbox": {"l": 36, "t": 700, "r": 400, "b": 720}, "coord_origin": "BOTTOMLEFT", "charspan": [0, 17]}],
            "text": "Annual Deductible",
        },
        {
            "self_ref": "#/texts/1",
            "parent": {"$ref": "#/body"},
            "label": "text",
            "prov": [{"page_no": 1, "bbox": {"l": 36, "t": 670, "r": 540, "b": 690}, "coord_origin": "BOTTOMLEFT", "charspan": [18, 95]}],
            "text": "All amounts shown are in-network rates.",
        },
        {
            "self_ref": "#/texts/5",
            "parent": {"$ref": "#/body"},
            "label": "caption",
            "prov": [{"page_no": 1, "bbox": {"l": 36, "t": 390, "r": 540, "b": 400}}],
            "text": "Table 1: Cost Sharing Summary",
        },
        {
            "self_ref": "#/texts/6",
            "parent": {"$ref": "#/body"},
            "label": "caption",
            "prov": [{"page_no": 2, "bbox": {"l": 72, "t": 290, "r": 540, "b": 300}}],
            "text": "Figure 1: Network Coverage Map",
        },
    ],
    "tables": [
        {
            "self_ref": "#/tables/0",
            "parent": {"$ref": "#/body"},
            "label": "table",
            "prov": [{"page_no": 1, "bbox": {"l": 36, "t": 400, "r": 540, "b": 650}, "coord_origin": "BOTTOMLEFT"}],
            "data": {
                "table_cells": [
                    {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Service",           "column_header": True},
                    {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "Your Cost",         "column_header": True},
                    {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Primary Care Visit","column_header": False},
                    {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "$20 copay",         "column_header": False},
                ],
                "num_rows": 2,
                "num_cols": 2,
            },
            "captions": [{"$ref": "#/texts/5"}],
        }
    ],
    "pictures": [
        {
            "self_ref": "#/pictures/0",
            "parent": {"$ref": "#/body"},
            "label": "picture",
            "prov": [{"page_no": 2, "bbox": {"l": 72, "t": 300, "r": 540, "b": 580}, "coord_origin": "BOTTOMLEFT"}],
            "captions": [{"$ref": "#/texts/6"}],
        }
    ],
    "pages": {
        "1": {"size": {"width": 612, "height": 792}},
        "2": {"size": {"width": 612, "height": 792}},
    },
}


# ======================================================================
# ADI Parser Tests
# ======================================================================

class TestADIParser:

    def setup_method(self):
        self.parser = ADIParser()

    def test_returns_parsed_document(self):
        result = self.parser.parse(ADI_SAMPLE)
        assert isinstance(result, ParsedDocument)

    def test_paragraph_block_extracted(self):
        result = self.parser.parse(ADI_SAMPLE)
        text_blocks = [b for b in result.blocks if b.type == "text"]
        contents = [b.content for b in text_blocks]
        assert "All amounts shown are in-network rates." in contents

    def test_section_heading_role_preserved(self):
        result = self.parser.parse(ADI_SAMPLE)
        heading_blocks = [b for b in result.blocks if b.role == "sectionHeading"]
        assert len(heading_blocks) == 1
        assert heading_blocks[0].content == "Annual Deductible"

    def test_table_serialized_as_markdown(self):
        result = self.parser.parse(ADI_SAMPLE)
        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert len(table_blocks) == 1
        table_text = table_blocks[0].content
        # Markdown table must contain header separator
        assert "---" in table_text
        # Cell values must be present
        assert "Primary Care Visit" in table_text
        assert "$20 copay" in table_text
        assert "Specialist Visit" in table_text
        assert "$40 copay" in table_text

    def test_table_caption_prepended(self):
        result = self.parser.parse(ADI_SAMPLE)
        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert "Table 1: Cost Sharing Summary" in table_blocks[0].content

    def test_figure_caption_extracted(self):
        result = self.parser.parse(ADI_SAMPLE)
        figure_blocks = [b for b in result.blocks if b.type == "figure"]
        assert len(figure_blocks) == 1
        assert figure_blocks[0].content == "Figure 1: Network Coverage Map"

    def test_figure_page_number_correct(self):
        result = self.parser.parse(ADI_SAMPLE)
        figure_blocks = [b for b in result.blocks if b.type == "figure"]
        assert figure_blocks[0].page_number == 2

    def test_table_page_number_correct(self):
        result = self.parser.parse(ADI_SAMPLE)
        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert table_blocks[0].page_number == 1

    def test_empty_document_returns_empty_blocks(self):
        result = self.parser.parse({})
        assert result.blocks == []

    def test_empty_paragraph_content_skipped(self):
        data = {"paragraphs": [{"content": "   ", "role": None, "bounding_regions": []}]}
        result = self.parser.parse(data)
        assert result.blocks == []

    def test_figure_without_caption_skipped(self):
        data = {"figures": [{"bounding_regions": [{"page_number": 1}], "caption": {}}]}
        result = self.parser.parse(data)
        figure_blocks = [b for b in result.blocks if b.type == "figure"]
        assert figure_blocks == []

    def test_table_markdown_has_pipe_delimiters(self):
        result = self.parser.parse(ADI_SAMPLE)
        table_blocks = [b for b in result.blocks if b.type == "table"]
        # Every row should start and end with |
        table_lines = [
            line for line in table_blocks[0].content.splitlines()
            if line.strip().startswith("|")
        ]
        assert len(table_lines) > 0
        for line in table_lines:
            assert line.strip().startswith("|")
            assert line.strip().endswith("|")


# ======================================================================
# Docling Parser Tests
# ======================================================================

class TestDoclingParser:

    def setup_method(self):
        self.parser = DoclingParser()

    def test_returns_parsed_document(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        assert isinstance(result, ParsedDocument)

    def test_text_block_extracted(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        text_blocks = [b for b in result.blocks if b.type == "text"]
        contents = [b.content for b in text_blocks]
        assert "All amounts shown are in-network rates." in contents

    def test_section_header_label_mapped_to_heading_role(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        heading_blocks = [b for b in result.blocks if b.role == "sectionHeading"]
        assert len(heading_blocks) >= 1
        assert any(b.content == "Annual Deductible" for b in heading_blocks)

    def test_table_serialized_as_markdown(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert len(table_blocks) == 1
        table_text = table_blocks[0].content
        assert "Service" in table_text
        assert "Your Cost" in table_text
        assert "Primary Care Visit" in table_text
        assert "$20 copay" in table_text

    def test_table_caption_resolved_via_ref(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert "Table 1: Cost Sharing Summary" in table_blocks[0].content

    def test_picture_caption_resolved_via_ref(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        figure_blocks = [b for b in result.blocks if b.type == "figure"]
        assert len(figure_blocks) == 1
        assert figure_blocks[0].content == "Figure 1: Network Coverage Map"

    def test_picture_page_number_correct(self):
        result = self.parser.parse(DOCLING_SAMPLE)
        figure_blocks = [b for b in result.blocks if b.type == "figure"]
        assert figure_blocks[0].page_number == 2

    def test_empty_document_returns_empty_blocks(self):
        result = self.parser.parse({"schema_name": "DoclingDocument"})
        assert result.blocks == []

    def test_picture_without_caption_skipped(self):
        data = {
            "schema_name": "DoclingDocument",
            "texts": [],
            "pictures": [{"self_ref": "#/pictures/0", "prov": [{"page_no": 1}], "captions": []}],
        }
        result = self.parser.parse(data)
        figure_blocks = [b for b in result.blocks if b.type == "figure"]
        assert figure_blocks == []


# ======================================================================
# Cross-parser contract tests
# Both parsers must produce the same normalized output shape
# ======================================================================

class TestParserContract:

    def test_adi_blocks_are_rawblock_instances(self):
        result = ADIParser().parse(ADI_SAMPLE)
        for block in result.blocks:
            assert isinstance(block, RawBlock)

    def test_docling_blocks_are_rawblock_instances(self):
        result = DoclingParser().parse(DOCLING_SAMPLE)
        for block in result.blocks:
            assert isinstance(block, RawBlock)

    def test_adi_block_types_valid(self):
        result = ADIParser().parse(ADI_SAMPLE)
        valid_types = {"text", "table", "figure"}
        for block in result.blocks:
            assert block.type in valid_types

    def test_docling_block_types_valid(self):
        result = DoclingParser().parse(DOCLING_SAMPLE)
        valid_types = {"text", "table", "figure"}
        for block in result.blocks:
            assert block.type in valid_types

    def test_both_parsers_extract_table_and_figure(self):
        adi_result = ADIParser().parse(ADI_SAMPLE)
        docling_result = DoclingParser().parse(DOCLING_SAMPLE)

        adi_types = {b.type for b in adi_result.blocks}
        docling_types = {b.type for b in docling_result.blocks}

        assert "table" in adi_types
        assert "figure" in adi_types
        assert "table" in docling_types
        assert "figure" in docling_types