"""Baseline extractor: text (embedded PDF text or Tesseract OCR) + label rules.

This is the "what a team would hand-build" baseline. It is fast and free, and
it breaks on layouts it wasn't written for, which is the point of comparing it
with the vision-LLM extractor.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from claimextract.documents import LoadedDocument
from claimextract.extractors import ExtractionError
from claimextract.schemas import ClaimForm, ExtractionResult, ServiceLine

# Labels as printed on the form. Each value runs until the next label or end of line.
LABELS = {
    "payer_name": r"1\.\s*PAYER",
    "member_id": r"1a\.\s*INSURED\s*ID",
    "patient_name": r"2\.\s*PATIENT\s*NAME",
    "patient_dob": r"3\.\s*PATIENT\s*DOB",
    "claim_id": r"26\.\s*PATIENT\s*ACCOUNT\s*NO",
    "provider_name": r"33\.\s*BILLING\s*PROVIDER",
    "provider_npi": r"33a\.\s*NPI",
    "amount_paid": r"29\.\s*AMOUNT\s*PAID",
    "diagnosis_codes": r"21\.\s*DIAGNOSIS\s*CODES",
    "total_charge": r"28\.\s*TOTAL\s*CHARGE",
}
# Horizontal whitespace only: an empty field must not swallow the next line.
NEXT_LABEL = r"(?=[ \t]+\d{1,2}[a-z]?\.[ \t]*[A-Z][A-Z ]+:|$)"
SERVICE_LINE_RE = re.compile(
    r"(?P<dos>\d{2}/\d{2}/\d{4})\s+(?P<cpt>[0-9A-Z]{5})\s+(?P<mod>[0-9A-Z]{2}|-)\s+"
    r"(?P<ptr>[A-L]{1,4})\s+(?P<units>\d+)\s+(?P<charge>[\d,]+\.\d{2})"
)
DX_RE = re.compile(r"[A-L]\.\s*([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)")


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def _parse_money(value: str | None) -> Decimal | None:
    if not value:
        return None
    try:
        return Decimal(value.replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return None


def _field(text: str, label: str) -> str | None:
    match = re.search(label + r"[ \t]*:[ \t]*(.*?)" + NEXT_LABEL, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip() or None


def parse_claim_text(text: str) -> tuple[ClaimForm, list[str]]:
    """Parse form text into a claim. Returns the claim and the fields not found."""
    raw = {name: _field(text, label) for name, label in LABELS.items()}
    lines = [
        ServiceLine(
            date_of_service=_parse_date(m["dos"]),
            cpt_code=m["cpt"],
            modifier=None if m["mod"] == "-" else m["mod"],
            diagnosis_pointer=m["ptr"],
            units=int(m["units"]),
            charge=_parse_money(m["charge"]),
        )
        for m in SERVICE_LINE_RE.finditer(text)
    ]
    claim = ClaimForm(
        claim_id=raw["claim_id"],
        patient_name=raw["patient_name"],
        patient_dob=_parse_date(raw["patient_dob"]),
        member_id=raw["member_id"],
        payer_name=raw["payer_name"],
        provider_name=raw["provider_name"],
        provider_npi=re.sub(r"\D", "", raw["provider_npi"]) if raw["provider_npi"] else None,
        diagnosis_codes=DX_RE.findall(raw["diagnosis_codes"] or ""),
        service_lines=lines,
        total_charge=_parse_money(raw["total_charge"]),
        amount_paid=_parse_money(raw["amount_paid"]),
    )
    missing = [name for name in LABELS if getattr(claim, name) in (None, "", [])]
    if not lines:
        missing.append("service_lines")
    return claim, missing


class OcrRulesExtractor:
    name = "ocr"

    def extract(self, document: LoadedDocument) -> ExtractionResult:
        text = document.embedded_text if document.is_digital else self._ocr(document)
        claim, missing = parse_claim_text(text)
        return ExtractionResult(claim=claim, extractor=self.name, uncertain_fields=missing)

    @staticmethod
    def _ocr(document: LoadedDocument) -> str:
        import pytesseract

        try:
            return "\n".join(pytesseract.image_to_string(page) for page in document.pages)
        except pytesseract.TesseractNotFoundError as exc:
            raise ExtractionError(
                "Tesseract is not installed; scanned documents need it (it is included in the Docker image)"
            ) from exc
