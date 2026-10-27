import random
from types import SimpleNamespace

import pytest

from claimextract.documents import UnsupportedDocument, load_document
from claimextract.evaluate import compare
from claimextract.extractors import ExtractionError
from claimextract.extractors.llm import ClaudeVisionExtractor, LLMClaim, LLMServiceLine, to_claim_form
from claimextract.extractors.ocr import OcrRulesExtractor, parse_claim_text
from claimextract.generator import pdf_to_scan, render_pdf
from claimextract.schemas import ClaimForm


def as_llm_output(claim: ClaimForm, uncertain: list[str] | None = None) -> LLMClaim:
    """What Claude would return for this claim in the flat schema."""
    s = lambda v: "" if v is None else str(v)  # noqa: E731
    return LLMClaim(
        claim_id=s(claim.claim_id),
        patient_name=s(claim.patient_name),
        patient_dob=s(claim.patient_dob),
        member_id=s(claim.member_id),
        payer_name=s(claim.payer_name),
        provider_name=s(claim.provider_name),
        provider_npi=s(claim.provider_npi),
        diagnosis_codes=claim.diagnosis_codes,
        service_lines=[
            LLMServiceLine(
                date_of_service=s(line.date_of_service),
                cpt_code=s(line.cpt_code),
                modifier=s(line.modifier),
                diagnosis_pointer=s(line.diagnosis_pointer),
                units=line.units or 0,
                charge=s(line.charge),
            )
            for line in claim.service_lines
        ],
        total_charge=s(claim.total_charge),
        amount_paid=s(claim.amount_paid),
        uncertain_fields=uncertain or [],
    )


def test_digital_pdf_round_trip(generated_claims):
    """Rules extractor reads every field of every generated digital PDF exactly."""
    extractor = OcrRulesExtractor()
    for claim in generated_claims:
        document = load_document(render_pdf(claim))
        assert document.is_digital
        result = extractor.extract(document)
        assert result.uncertain_fields == []
        assert all(compare(claim, result.claim).values()), compare(claim, result.claim)


def test_empty_field_does_not_swallow_next_line():
    text = "1. PAYER: Acme Health 1a. INSURED ID:\n2. PATIENT NAME: Doe, Jane 3. PATIENT DOB: 01/02/1990"
    claim, missing = parse_claim_text(text)
    assert claim.member_id is None
    assert "member_id" in missing
    assert claim.patient_name == "Doe, Jane"
    assert claim.payer_name == "Acme Health"


def test_scan_is_not_digital(generated_claims):
    png = pdf_to_scan(render_pdf(generated_claims[0]), random.Random(1))
    document = load_document(png, "image/png")
    assert not document.is_digital
    assert document.content_type == "image/png"
    assert len(document.pages) == 1


def test_content_type_comes_from_bytes_not_client(generated_claims):
    document = load_document(render_pdf(generated_claims[0]), "image/png")
    assert document.content_type == "application/pdf"


@pytest.mark.parametrize("data", [b"hello world", b"%PDF-1.7 truncated garbage", b"\x89PNG not really"])
def test_unreadable_files_are_rejected(data):
    with pytest.raises(UnsupportedDocument):
        load_document(data)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_llm_extractor_maps_structured_output(generated_claims):
    claim = generated_claims[0]
    parsed = as_llm_output(claim, ["service_lines[0].charge"])
    usage = SimpleNamespace(input_tokens=2000, output_tokens=500)
    response = SimpleNamespace(stop_reason="end_turn", parsed_output=parsed, model="claude-opus-5-5", usage=usage)
    messages = FakeMessages(response)
    extractor = ClaudeVisionExtractor(client=fake_client(messages))

    result = extractor.extract(load_document(render_pdf(claim)))

    assert result.claim == claim
    assert result.uncertain_fields == ["service_lines[0].charge"]
    assert result.cost_usd == pytest.approx((2000 * 4 + 500 * 20) / 1_000_000)
    call = messages.calls[0]
    assert call["output_format"] is LLMClaim
    assert call["messages"][0]["content"][0]["type"] == "document"


def test_llm_schema_has_no_unions_or_optional_fields():
    """Structured outputs cap nullable/optional fields; the flat schema must stay under it."""
    import json

    from anthropic.lib._parse._transform import transform_schema

    schema = transform_schema(LLMClaim)
    assert "anyOf" not in json.dumps(schema)
    for obj in [schema, *schema["$defs"].values()]:
        assert set(obj["required"]) == set(obj["properties"])


def test_flat_output_blanks_and_bad_values(valid_claim):
    llm = as_llm_output(valid_claim)
    llm.member_id = "  "
    llm.service_lines[0].modifier = ""
    llm.service_lines[1].charge = "6S.00"  # misread digit
    llm.patient_dob = "1980-13-40"
    claim, uncertain = to_claim_form(llm)
    assert claim.member_id is None
    assert claim.service_lines[0].modifier is None
    assert claim.service_lines[1].charge is None
    assert claim.patient_dob is None
    assert uncertain == ["service_lines[1].charge", "patient_dob"]


def test_llm_extractor_sends_images_for_scans(generated_claims):
    claim = generated_claims[0]
    parsed = as_llm_output(claim)
    messages = FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed))
    png = pdf_to_scan(render_pdf(claim), random.Random(3))

    ClaudeVisionExtractor(client=fake_client(messages)).extract(load_document(png))

    assert messages.calls[0]["messages"][0]["content"][0]["type"] == "image"


def test_llm_refusal_is_an_extraction_error(generated_claims):
    messages = FakeMessages(SimpleNamespace(stop_reason="refusal", parsed_output=None))
    extractor = ClaudeVisionExtractor(client=fake_client(messages))
    with pytest.raises(ExtractionError, match="declined"):
        extractor.extract(load_document(render_pdf(generated_claims[0])))


def _tesseract_available() -> bool:
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract not installed")
def test_ocr_reads_clean_scan(valid_claim):
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(render_pdf(valid_claim))
    buf = io.BytesIO()
    pdf[0].render(scale=3.0).to_pil().save(buf, format="PNG")
    pdf.close()

    result = OcrRulesExtractor().extract(load_document(buf.getvalue()))

    assert result.claim.claim_id == valid_claim.claim_id
    assert result.claim.total_charge == valid_claim.total_charge


def test_llm_with_unloadable_credentials_is_an_extraction_error(monkeypatch, tmp_path, generated_claims):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))  # empty: no profile to load
    with pytest.raises(ExtractionError, match="credentials"):
        ClaudeVisionExtractor().extract(load_document(render_pdf(generated_claims[0])))


def test_llm_with_no_credentials_is_an_extraction_error(generated_claims):
    class NoAuthMessages(FakeMessages):
        def parse(self, **kwargs):
            raise TypeError("Could not resolve authentication method")

    client = fake_client(NoAuthMessages())
    client.api_key = client.auth_token = None
    with pytest.raises(ExtractionError, match="credentials"):
        ClaudeVisionExtractor(client=client).extract(load_document(render_pdf(generated_claims[0])))


def test_evaluate_digital_set(tmp_path):
    from claimextract.evaluate import evaluate
    from claimextract.generator import generate

    generate(20, tmp_path, seed=5, scan_rate=0.0, defect_rate=0.3)
    report = evaluate(tmp_path, "ocr")

    assert report["failed"] == 0
    assert report["exact_match_pct"] == 100.0
    caught, total = map(int, report["defects_sent_to_review"].split("/"))
    assert caught == total > 0
    assert report["wrong_extractions_auto_validated"] == "0/0"
