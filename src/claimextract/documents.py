"""Loading uploaded files: PDFs (digital or scanned) and images."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

PDF_TYPES = {"application/pdf"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/webp"}
SUPPORTED_TYPES = PDF_TYPES | IMAGE_TYPES
MAX_PAGES = 10
# A page with fewer embedded characters than this is treated as a scan.
MIN_TEXT_CHARS = 50


class UnsupportedDocument(ValueError):
    pass


@dataclass
class LoadedDocument:
    content_type: str
    data: bytes
    pages: list[Image.Image]
    embedded_text: str | None  # text layer of a digital PDF, None for scans/images

    @property
    def is_digital(self) -> bool:
        return self.embedded_text is not None


def sniff_content_type(data: bytes, declared: str | None = None) -> str:
    """Trust magic bytes over the client-declared content type."""
    if data.startswith(b"%PDF"):
        return "application/pdf"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise UnsupportedDocument(f"unsupported file type (declared {declared or 'unknown'})")


def load_document(data: bytes, declared_type: str | None = None) -> LoadedDocument:
    content_type = sniff_content_type(data, declared_type)
    if content_type in PDF_TYPES:
        return _load_pdf(data)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # Pillow raises many types for corrupt files
        raise UnsupportedDocument(f"could not read image: {exc}") from exc
    return LoadedDocument(content_type, data, [image.convert("RGB")], None)


def _load_pdf(data: bytes) -> LoadedDocument:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise UnsupportedDocument(f"could not read PDF: {exc}") from exc
    try:
        if len(pdf) == 0:
            raise UnsupportedDocument("PDF has no pages")
        if len(pdf) > MAX_PAGES:
            raise UnsupportedDocument(f"PDF has {len(pdf)} pages; the limit is {MAX_PAGES}")
        texts, pages = [], []
        for page in pdf:
            texts.append(page.get_textpage().get_text_bounded())
            pages.append(page.render(scale=2.0).to_pil().convert("RGB"))
    finally:
        pdf.close()
    text = "\n".join(texts).replace("\r\n", "\n")
    embedded = text if len(text.strip()) >= MIN_TEXT_CHARS else None
    return LoadedDocument("application/pdf", data, pages, embedded)
