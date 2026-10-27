"""Extractors turn a loaded document into a ClaimForm."""

from __future__ import annotations

from typing import Protocol

from claimextract.documents import LoadedDocument
from claimextract.schemas import ExtractionResult


class ExtractionError(RuntimeError):
    pass


class Extractor(Protocol):
    name: str

    def extract(self, document: LoadedDocument) -> ExtractionResult: ...


def get_extractor(name: str) -> Extractor:
    if name == "ocr":
        from claimextract.extractors.ocr import OcrRulesExtractor

        return OcrRulesExtractor()
    if name == "llm":
        from claimextract.extractors.llm import ClaudeVisionExtractor

        return ClaudeVisionExtractor()
    raise ValueError(f"unknown extractor {name!r}; choose 'ocr' or 'llm'")
