"""
Adapter for Azure Document Intelligence (ADI) parser output.

ADI JSON shape:
  {
    "paragraphs": [ { "content", "role", "bounding_regions": [{"page_number"}] } ],
    "tables":     [ { "row_count", "column_count", "cells": [...], "bounding_regions", "caption" } ],
    "figures":    [ { "bounding_regions", "caption" } ],
    "content":    "full raw text string"
  }
"""

from .base import BaseParser, ParsedDocument, RawBlock


class ADIParser(BaseParser):

    def parse(self, data: dict) -> ParsedDocument:
        blocks: list[RawBlock] = []

        # ----------------------------------------------------------------
        # 1. Paragraphs
        # ----------------------------------------------------------------
        for para in data.get("paragraphs", []):
            content = para.get("content", "").strip()
            if not content:
                continue

            page = self._first_page(para.get("bounding_regions", []))
            role = para.get("role")  # "sectionHeading" | None | etc.

            blocks.append(RawBlock(
                type="text",
                content=content,
                page_number=page,
                role=role,
            ))

        # ----------------------------------------------------------------
        # 2. Tables  — serialized as markdown, never split downstream
        # ----------------------------------------------------------------
        for table in data.get("tables", []):
            content = self._serialize_table(table)
            if not content:
                continue

            page = self._first_page(table.get("bounding_regions", []))

            # Prepend caption so it's searchable alongside cell content
            caption = table.get("caption", {}).get("content", "")
            if caption:
                content = f"{caption}\n\n{content}"

            blocks.append(RawBlock(
                type="table",
                content=content,
                page_number=page,
                role="table",
            ))

        # ----------------------------------------------------------------
        # 3. Figures — only caption is useful for embedding
        # ----------------------------------------------------------------
        for figure in data.get("figures", []):
            caption = figure.get("caption", {}).get("content", "")
            if not caption:
                continue  # nothing to embed without a caption

            page = self._first_page(figure.get("bounding_regions", []))

            blocks.append(RawBlock(
                type="figure",
                content=caption,
                page_number=page,
                role="figure",
            ))

        return ParsedDocument(blocks=blocks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _first_page(self, bounding_regions: list) -> int:
        """Extract page number from the first bounding region. Default 1."""
        if bounding_regions:
            return bounding_regions[0].get("page_number", 1)
        return 1

    def _serialize_table(self, table: dict) -> str:
        """
        Convert ADI table cells into a markdown table string.

        Why markdown?
          - Human-readable in payloads / logs
          - Embedding models handle markdown well
          - Preserves row/column structure without custom parsing

        We never split tables — the whole table is one chunk.
        A split table cell can produce a legally wrong answer
        (e.g. "$20 copay" separated from "Primary Care Visit").
        """
        row_count = table.get("row_count", 0)
        col_count = table.get("column_count", 0)
        cells = table.get("cells", [])

        if not cells or row_count == 0 or col_count == 0:
            return ""

        # Build a 2D grid filled with empty strings
        grid: list[list[str]] = [
            [""] * col_count for _ in range(row_count)
        ]

        for cell in cells:
            r = cell.get("row_index", 0)
            c = cell.get("column_index", 0)
            if r < row_count and c < col_count:
                grid[r][c] = cell.get("content", "")

        # Render as markdown table
        lines = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                # Separator row after header
                lines.append("| " + " | ".join(["---"] * col_count) + " |")

        return "\n".join(lines)