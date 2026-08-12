"""
Base parser interface and normalized document models.

Every parser adapter (ADI, Docling, etc.) must output a ParsedDocument.
Nothing downstream knows or cares which parser was used.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RawBlock:
    """
    A single semantic unit extracted from a document.

    type:
        "text"    — a paragraph or heading
        "table"   — a full table (serialized as markdown)
        "figure"  — an image/chart, represented by its caption

    role:
        "sectionHeading", "paragraph", "caption", None, etc.
        Used by the chunker to propagate heading context.
    """
    type: str
    content: str
    page_number: int
    role: str | None = None


@dataclass
class ParsedDocument:
    """
    Normalized output produced by any parser adapter.
    This is the contract between parsing and chunking.
    """
    blocks: list[RawBlock] = field(default_factory=list)


class BaseParser(ABC):
    """
    Abstract adapter — all parsers implement this interface.

    Why adapter pattern?
    Parser output formats change (ADI today, Docling tomorrow).
    Isolating parser logic here means chunking/embedding code
    never needs to change when we add a new document source.
    """

    @abstractmethod
    def parse(self, data: dict) -> ParsedDocument:
        """
        Accept raw parser JSON as a dict.
        Return a normalized ParsedDocument.
        """
        ...