"""
Adapter for Docling parser output.

Docling JSON shape:
  {
    "schema_name": "DoclingDocument",
    "texts":    [ { "self_ref", "label", "text", "prov": [{"page_no", "bbox"}] } ],
    "tables":   [ { "self_ref", "data": {"table_cells", "num_rows", "num_cols"}, "captions", "prov" } ],
    "pictures": [ { "self_ref", "captions", "prov" } ],
    "pages":    { "1": { "size": { "width", "height" } } }
  }
"""

from .base import BaseParser, ParsedDocument, RawBlock


# Map Docling labels to our normalized role names
LABEL_TO_ROLE: dict[str, str] = {
    "section_header": "sectionHeading",
    "title":          "sectionHeading",
    "text":           "paragraph",
    "paragraph":      "paragraph",
    "list_item":      "paragraph",
    "table":          "table",
    "picture":        "figure",
    "caption":        "caption",
}


class DoclingParser(BaseParser):

    def parse(self, data: dict) -> ParsedDocument:
        blocks: list[RawBlock] = []

        # Build a ref → text lookup so we can resolve caption $refs
        ref_to_text: dict[str, str] = {
            item["self_ref"]: item.get("text", "")
            for item in data.get("texts", [])
            if "self_ref" in item
        }

        # ----------------------------------------------------------------
        # 1. Text blocks
        # ----------------------------------------------------------------
        for text_item in data.get("texts", []):
            content = text_item.get("text", "").strip()
            if not content:
                continue

            label = text_item.get("label", "text")
            role = LABEL_TO_ROLE.get(label, "paragraph")
            page = self._first_page(text_item.get("prov", []))

            blocks.append(RawBlock(
                type="text",
                content=content,
                page_number=page,
                role=role,
            ))

        # ----------------------------------------------------------------
        # 2. Tables
        # ----------------------------------------------------------------
        for table in data.get("tables", []):
            table_data = table.get("data", {})
            content = self._serialize_table(table_data)
            if not content:
                continue

            page = self._first_page(table.get("prov", []))

            # Resolve caption via $ref
            caption = self._resolve_caption(table.get("captions", []), ref_to_text)
            if caption:
                content = f"{caption}\n\n{content}"

            blocks.append(RawBlock(
                type="table",
                content=content,
                page_number=page,
                role="table",
            ))

        # ----------------------------------------------------------------
        # 3. Pictures — only caption is embeddable
        # ----------------------------------------------------------------
        for picture in data.get("pictures", []):
            caption = self._resolve_caption(picture.get("captions", []), ref_to_text)
            if not caption:
                continue

            page = self._first_page(picture.get("prov", []))

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

    def _first_page(self, prov: list) -> int:
        """Extract page number from first provenance entry."""
        if prov:
            return prov[0].get("page_no", 1)
        return 1

    def _resolve_caption(self, captions: list, ref_to_text: dict[str, str]) -> str:
        """Resolve a list of caption $refs to the first non-empty text."""
        for cap in captions:
            ref = cap.get("$ref", "")
            text = ref_to_text.get(ref, "").strip()
            if text:
                return text
        return ""

    def _serialize_table(self, table_data: dict) -> str:
        """
        Convert Docling table_cells into a markdown table string.

        Docling uses start/end row+col offsets (0-indexed, exclusive end).
        column_header=true marks header cells.
        """
        num_rows = table_data.get("num_rows", 0)
        num_cols = table_data.get("num_cols", 0)
        cells = table_data.get("table_cells", [])

        if not cells or num_rows == 0 or num_cols == 0:
            return ""

        # Build 2D grid
        grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
        header_rows: set[int] = set()

        for cell in cells:
            r = cell.get("start_row_offset_idx", 0)
            c = cell.get("start_col_offset_idx", 0)
            if r < num_rows and c < num_cols:
                grid[r][c] = cell.get("text", "")
                if cell.get("column_header", False):
                    header_rows.add(r)

        # Render as markdown
        lines = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i in header_rows:
                lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        return "\n".join(lines)
