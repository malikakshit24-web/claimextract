"""Rule-based checks that don't trust the extractor.

Errors mean the claim can't be posted as-is; warnings mean a human should look.
Either one sends the document to the review queue.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from claimextract.schemas import ClaimForm, Severity, ValidationIssue

CPT_RE = re.compile(r"^(\d{4}[0-9A-Z]|[A-Z]\d{4})$")
ICD10_RE = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
REQUIRED_FIELDS = (
    "claim_id",
    "patient_name",
    "patient_dob",
    "member_id",
    "payer_name",
    "provider_name",
    "provider_npi",
    "total_charge",
)


def npi_is_valid(npi: str) -> bool:
    """NPI check digit: Luhn over the 80840 prefix + first 9 digits."""
    if not re.fullmatch(r"\d{10}", npi):
        return False
    digits = [int(d) for d in "80840" + npi[:9]]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10 == int(npi[9])


def validate_claim(claim: ClaimForm, today: date | None = None) -> list[ValidationIssue]:
    today = today or date.today()
    issues: list[ValidationIssue] = []

    def add(field: str, severity: Severity, message: str) -> None:
        issues.append(ValidationIssue(field=field, severity=severity, message=message))

    for name in REQUIRED_FIELDS:
        if getattr(claim, name) in (None, ""):
            add(name, Severity.ERROR, "missing required field")

    if claim.provider_npi and not npi_is_valid(claim.provider_npi):
        add("provider_npi", Severity.ERROR, f"invalid NPI {claim.provider_npi!r} (check digit)")

    if claim.patient_dob and claim.patient_dob > today:
        add("patient_dob", Severity.ERROR, "date of birth is in the future")

    if not claim.diagnosis_codes:
        add("diagnosis_codes", Severity.ERROR, "at least one diagnosis code is required")
    for i, code in enumerate(claim.diagnosis_codes):
        if not ICD10_RE.match(code):
            add(f"diagnosis_codes[{i}]", Severity.ERROR, f"{code!r} is not a valid ICD-10 format")

    if not claim.service_lines:
        add("service_lines", Severity.ERROR, "at least one service line is required")

    pointer_letters = {chr(ord("A") + i) for i in range(len(claim.diagnosis_codes))}
    line_sum = Decimal("0")
    for i, line in enumerate(claim.service_lines):
        prefix = f"service_lines[{i}]"
        if not line.cpt_code:
            add(f"{prefix}.cpt_code", Severity.ERROR, "missing procedure code")
        elif not CPT_RE.match(line.cpt_code):
            add(f"{prefix}.cpt_code", Severity.ERROR, f"{line.cpt_code!r} is not a valid CPT format")
        if line.charge is None:
            add(f"{prefix}.charge", Severity.ERROR, "missing line charge")
        elif line.charge <= 0:
            add(f"{prefix}.charge", Severity.ERROR, "line charge must be positive")
        else:
            line_sum += line.charge
        if line.units is not None and line.units <= 0:
            add(f"{prefix}.units", Severity.ERROR, "units must be positive")
        if line.date_of_service is None:
            add(f"{prefix}.date_of_service", Severity.ERROR, "missing date of service")
        else:
            if line.date_of_service > today:
                add(f"{prefix}.date_of_service", Severity.ERROR, "date of service is in the future")
            if claim.patient_dob and line.date_of_service < claim.patient_dob:
                add(f"{prefix}.date_of_service", Severity.ERROR, "service before date of birth")
        if line.diagnosis_pointer:
            bad = set(line.diagnosis_pointer) - pointer_letters
            if bad:
                add(
                    f"{prefix}.diagnosis_pointer",
                    Severity.WARNING,
                    f"points at missing diagnosis {''.join(sorted(bad))}",
                )

    if claim.total_charge is not None and claim.service_lines and line_sum != claim.total_charge:
        add(
            "total_charge",
            Severity.ERROR,
            f"total {claim.total_charge} does not equal sum of lines {line_sum}",
        )
    if claim.amount_paid is not None:
        if claim.amount_paid < 0:
            add("amount_paid", Severity.ERROR, "amount paid is negative")
        elif claim.total_charge is not None and claim.amount_paid > claim.total_charge:
            add("amount_paid", Severity.WARNING, "amount paid exceeds total charge")

    return issues
