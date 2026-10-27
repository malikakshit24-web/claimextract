import os
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from claimextract import db
from claimextract.api import app, extractor_dependency
from claimextract.extractors import ExtractionError
from claimextract.extractors.ocr import OcrRulesExtractor
from claimextract.generator import render_pdf


class BrokenExtractor:
    name = "broken"

    def extract(self, document):
        raise ExtractionError("model unavailable")


@pytest.fixture
def client():
    # CI sets this to a real PostgreSQL; locally the tests fall back to in-memory SQLite.
    url = os.environ.get("CLAIMEXTRACT_TEST_DATABASE_URL")
    if url:
        engine = db.init_engine(url)
    else:
        engine = db.init_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.Base.metadata.drop_all(engine)
    db.create_tables()
    app.dependency_overrides[extractor_dependency] = OcrRulesExtractor
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def upload(client, data: bytes, name: str = "claim.pdf"):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_clean_claim_is_validated(client, valid_claim):
    response = upload(client, render_pdf(valid_claim))
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "validated"
    assert body["issues"] == []
    assert body["claim"]["total_charge"] == "250.00"
    assert client.get(f"/documents/{body['id']}").json()["claim"] == body["claim"]


def test_inconsistent_claim_goes_to_review(client, valid_claim):
    valid_claim.total_charge = Decimal("999.00")
    body = upload(client, render_pdf(valid_claim)).json()
    assert body["status"] == "needs_review"
    assert [i["field"] for i in body["issues"]] == ["total_charge"]

    queue = client.get("/documents", params={"status": "needs_review"}).json()
    assert [d["id"] for d in queue] == [body["id"]]


def test_reviewer_corrects_claim(client, valid_claim):
    valid_claim.total_charge = Decimal("999.00")
    doc = upload(client, render_pdf(valid_claim)).json()

    # Approving as-is is refused while hard errors remain.
    refused = client.post(f"/documents/{doc['id']}/review", json={"action": "approve", "reviewer": "sam"})
    assert refused.status_code == 409

    corrected = doc["claim"] | {"total_charge": "250.00"}
    response = client.post(
        f"/documents/{doc['id']}/review",
        json={"action": "correct", "reviewer": "sam", "claim": corrected, "note": "total misprinted"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["claim"]["total_charge"] == "250.00"
    assert body["reviewer"] == "sam"

    # Reviewing twice is a conflict.
    again = client.post(f"/documents/{doc['id']}/review", json={"action": "reject", "reviewer": "sam"})
    assert again.status_code == 409


def test_reviewer_rejects(client, valid_claim):
    valid_claim.member_id = None
    doc = upload(client, render_pdf(valid_claim)).json()
    response = client.post(f"/documents/{doc['id']}/review", json={"action": "reject", "reviewer": "kim"})
    assert response.json()["status"] == "rejected"


def test_extraction_failure_is_recorded(client, valid_claim):
    app.dependency_overrides[extractor_dependency] = BrokenExtractor
    body = upload(client, render_pdf(valid_claim)).json()
    assert body["status"] == "failed"
    assert body["error"] == "model unavailable"
    assert body["claim"] is None


def test_unsupported_file(client):
    assert upload(client, b"not a document", "notes.txt").status_code == 415


def test_empty_file(client):
    assert upload(client, b"").status_code == 422


def test_unknown_extractor_param(client, valid_claim):
    app.dependency_overrides.clear()
    response = client.post(
        "/documents", params={"extractor": "gpt"}, files={"file": ("c.pdf", render_pdf(valid_claim), "application/pdf")}
    )
    assert response.status_code == 422
    assert "unknown extractor" in response.json()["detail"]


def test_missing_document(client):
    assert client.get("/documents/999").status_code == 404
