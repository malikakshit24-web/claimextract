"""The processing pipeline: load -> extract -> validate -> route."""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from claimextract.db import Document
from claimextract.documents import load_document
from claimextract.extractors import ExtractionError, Extractor
from claimextract.schemas import (
    ClaimForm,
    DocumentStatus,
    ReviewAction,
    ReviewRequest,
    ValidationIssue,
)
from claimextract.validation import validate_claim


def route(issues: list[ValidationIssue], uncertain_fields: list[str]) -> DocumentStatus:
    """Only documents with no issues and nothing the extractor was unsure of skip review."""
    if issues or uncertain_fields:
        return DocumentStatus.NEEDS_REVIEW
    return DocumentStatus.VALIDATED


def process_upload(
    session: Session, extractor: Extractor, filename: str, data: bytes, declared_type: str | None
) -> Document:
    """Store and process one upload. Raises UnsupportedDocument for unreadable files."""
    document = load_document(data, declared_type)
    row = Document(
        filename=filename,
        content_type=document.content_type,
        sha256=hashlib.sha256(data).hexdigest(),
        status=DocumentStatus.FAILED,
        extractor=extractor.name,
        issues=[],
        uncertain_fields=[],
    )
    try:
        result = extractor.extract(document)
    except ExtractionError as exc:
        row.error = str(exc)
    else:
        claim_json = result.claim.model_dump(mode="json")
        issues = validate_claim(result.claim)
        row.extracted = claim_json
        row.claim = claim_json
        row.issues = [issue.model_dump(mode="json") for issue in issues]
        row.uncertain_fields = result.uncertain_fields
        row.status = route(issues, result.uncertain_fields)
    session.add(row)
    session.commit()
    return row


class ReviewError(ValueError):
    pass


def apply_review(session: Session, row: Document, review: ReviewRequest) -> Document:
    if row.status != DocumentStatus.NEEDS_REVIEW:
        raise ReviewError(f"document is {row.status}, not waiting for review")

    if review.action == ReviewAction.REJECT:
        row.status = DocumentStatus.REJECTED
    else:
        if review.action == ReviewAction.CORRECT:
            if review.claim is None:
                raise ReviewError("a corrected claim is required for 'correct'")
            claim = review.claim
        else:
            claim = ClaimForm.model_validate(row.claim)
        # A human can approve over warnings, but not over hard errors.
        issues = validate_claim(claim)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise ReviewError("claim still has errors: " + "; ".join(f"{i.field}: {i.message}" for i in errors))
        row.claim = claim.model_dump(mode="json")
        row.issues = [i.model_dump(mode="json") for i in issues]
        row.status = DocumentStatus.APPROVED

    row.reviewer = review.reviewer
    row.review_note = review.note
    session.commit()
    return row
