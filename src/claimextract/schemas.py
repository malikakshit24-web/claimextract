"""Pydantic schemas for extracted claims and API payloads.

`ClaimForm` is the single contract every extractor must produce, whether it
reads OCR text with rules or asks a vision LLM. Money is `Decimal`, never float.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ServiceLine(BaseModel):
    """One row of the service table (CMS-1500 box 24)."""

    date_of_service: date | None = Field(None, description="Date the service was provided")
    cpt_code: str | None = Field(None, description="5-character CPT/HCPCS procedure code")
    modifier: str | None = Field(None, description="Optional 2-character CPT modifier")
    diagnosis_pointer: str | None = Field(
        None, description="Letters pointing at the claim's diagnosis codes, e.g. 'A' or 'AB'"
    )
    units: int | None = Field(None, description="Number of units billed")
    charge: Decimal | None = Field(None, description="Line charge in dollars, e.g. 125.00")


class ClaimForm(BaseModel):
    """Structured fields of a CMS-1500-style professional claim."""

    claim_id: str | None = Field(None, description="Patient account / claim number")
    patient_name: str | None = Field(None, description="Patient full name as 'Last, First'")
    patient_dob: date | None = Field(None, description="Patient date of birth")
    member_id: str | None = Field(None, description="Insured's member/ID number")
    payer_name: str | None = Field(None, description="Insurance company name")
    provider_name: str | None = Field(None, description="Billing provider name")
    provider_npi: str | None = Field(None, description="Billing provider 10-digit NPI")
    diagnosis_codes: list[str] = Field(default_factory=list, description="ICD-10-CM codes in order (A, B, C, ...)")
    service_lines: list[ServiceLine] = Field(default_factory=list)
    total_charge: Decimal | None = Field(None, description="Total charge in dollars")
    amount_paid: Decimal | None = Field(None, description="Amount already paid in dollars")


class LLMClaimExtraction(ClaimForm):
    """What the vision LLM returns: the claim plus fields it was unsure about."""

    uncertain_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields that were illegible, ambiguous or guessed",
    )


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    field: str
    severity: Severity
    message: str


class DocumentStatus(StrEnum):
    VALIDATED = "validated"  # passed every check; ready to post downstream
    NEEDS_REVIEW = "needs_review"  # has issues; waiting for a human
    APPROVED = "approved"  # a human approved or corrected it
    REJECTED = "rejected"  # a human rejected it
    FAILED = "failed"  # extraction itself failed


class ExtractionResult(BaseModel):
    claim: ClaimForm
    extractor: str
    uncertain_fields: list[str] = Field(default_factory=list)


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    sha256: str
    status: DocumentStatus
    extractor: str | None
    claim: ClaimForm | None
    issues: list[ValidationIssue]
    uncertain_fields: list[str]
    error: str | None
    reviewer: str | None
    review_note: str | None
    created_at: datetime
    updated_at: datetime


class ReviewAction(StrEnum):
    APPROVE = "approve"
    CORRECT = "correct"
    REJECT = "reject"


class ReviewRequest(BaseModel):
    action: ReviewAction
    reviewer: str = Field(min_length=1)
    claim: ClaimForm | None = Field(None, description="Corrected claim; required for 'correct'")
    note: str | None = None
