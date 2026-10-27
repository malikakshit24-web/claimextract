from datetime import date
from decimal import Decimal

import pytest

from claimextract.schemas import Severity
from claimextract.validation import npi_is_valid, validate_claim

TODAY = date(2026, 10, 26)


def fields(issues):
    return {issue.field for issue in issues}


def test_valid_claim_has_no_issues(valid_claim):
    assert validate_claim(valid_claim, TODAY) == []


def test_generated_claims_are_valid(generated_claims):
    for claim in generated_claims:
        assert validate_claim(claim, TODAY) == []


@pytest.mark.parametrize(("npi", "ok"), [("1234567893", True), ("1234567890", False), ("12345", False), ("", False)])
def test_npi_check_digit(npi, ok):
    assert npi_is_valid(npi) is ok


def test_missing_required_fields(valid_claim):
    valid_claim.member_id = None
    valid_claim.payer_name = ""
    assert {"member_id", "payer_name"} <= fields(validate_claim(valid_claim, TODAY))


def test_total_must_equal_sum_of_lines(valid_claim):
    valid_claim.total_charge = Decimal("260.00")
    issues = validate_claim(valid_claim, TODAY)
    assert fields(issues) == {"total_charge"}
    assert "250.00" in issues[0].message


def test_invalid_codes(valid_claim):
    valid_claim.diagnosis_codes[1] = "110"
    valid_claim.service_lines[0].cpt_code = "9921"
    assert fields(validate_claim(valid_claim, TODAY)) == {"diagnosis_codes[1]", "service_lines[0].cpt_code"}


def test_dates(valid_claim):
    valid_claim.service_lines[0].date_of_service = date(2027, 1, 1)
    valid_claim.service_lines[1].date_of_service = date(1970, 1, 1)
    assert fields(validate_claim(valid_claim, TODAY)) == {
        "service_lines[0].date_of_service",
        "service_lines[1].date_of_service",
    }


def test_negative_charge_is_error(valid_claim):
    valid_claim.service_lines[1].charge = Decimal("-65.00")
    assert "service_lines[1].charge" in fields(validate_claim(valid_claim, TODAY))


def test_dangling_diagnosis_pointer_is_warning(valid_claim):
    valid_claim.service_lines[1].diagnosis_pointer = "C"
    issues = validate_claim(valid_claim, TODAY)
    assert [(i.field, i.severity) for i in issues] == [("service_lines[1].diagnosis_pointer", Severity.WARNING)]


def test_overpayment_is_warning(valid_claim):
    valid_claim.amount_paid = Decimal("300.00")
    issues = validate_claim(valid_claim, TODAY)
    assert [(i.field, i.severity) for i in issues] == [("amount_paid", Severity.WARNING)]
