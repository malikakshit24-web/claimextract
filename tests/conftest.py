import random
from datetime import date
from decimal import Decimal

import pytest
from faker import Faker

from claimextract.generator import random_claim
from claimextract.schemas import ClaimForm, ServiceLine


@pytest.fixture
def valid_claim() -> ClaimForm:
    return ClaimForm(
        claim_id="CLM-123456",
        patient_name="Rivera, Ana",
        patient_dob=date(1980, 5, 17),
        member_id="B123456789",
        payer_name="Summit Commercial PPO",
        provider_name="Okafor Family Medicine",
        provider_npi="1234567893",  # valid check digit
        diagnosis_codes=["E11.9", "I10"],
        service_lines=[
            ServiceLine(
                date_of_service=date(2026, 3, 2),
                cpt_code="99214",
                diagnosis_pointer="AB",
                units=1,
                charge=Decimal("185.00"),
            ),
            ServiceLine(
                date_of_service=date(2026, 3, 2),
                cpt_code="85025",
                modifier="59",
                diagnosis_pointer="A",
                units=2,
                charge=Decimal("65.00"),
            ),
        ],
        total_charge=Decimal("250.00"),
        amount_paid=Decimal("0.00"),
    )


@pytest.fixture
def generated_claims():
    """Ten clean synthetic claims from a fixed seed."""
    rng, fake = random.Random(42), Faker()
    Faker.seed(42)
    return [random_claim(rng, fake).claim for _ in range(10)]
