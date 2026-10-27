"""Vision-LLM extractor: sends the document to Claude and gets back a validated ClaimForm.

Uses structured outputs (`messages.parse` with a Pydantic model), so the response
is guaranteed to match the schema. The model also lists fields it was unsure
about, which feed the review routing alongside the rule-based checks.

The schema sent to Claude is deliberately flat: every field is required and a
plain string or integer, with "" (or 0) meaning blank. Structured outputs cap
the number of optional and nullable fields per request, and a nullable version
of ClaimForm exceeds that cap ("Schema is too complex"). Values are converted to
ClaimForm here; anything that doesn't parse becomes None and is flagged uncertain.

Needs ANTHROPIC_API_KEY (or another Anthropic credential) in the environment.
"""

from __future__ import annotations

import base64
import io
from datetime import date
from decimal import Decimal, InvalidOperation

import anthropic
from pydantic import BaseModel, Field

from claimextract.config import get_settings
from claimextract.documents import LoadedDocument
from claimextract.extractors import ExtractionError
from claimextract.schemas import ClaimForm, ExtractionResult, ServiceLine

# USD per million tokens (input, output). Used to report cost per document.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = """\
You extract data from healthcare professional claim forms (CMS-1500 style) for a \
claims-processing system. Copy values exactly as printed: do not correct, compute \
or infer them. If the printed total doesn't match the line items, return the printed \
total anyway; downstream validation checks the arithmetic.

Every field must be present. If a field is blank or illegible, use "" (or 0 for \
units) and add its name to uncertain_fields. Also list any field whose value you \
had to guess between readings, for example a smudged digit. Use the schema's field \
names, with list indexes for line items, e.g. "service_lines[1].charge".

Dates are YYYY-MM-DD. Money is digits with two decimals and no currency symbol, \
e.g. "1250.00". A modifier printed as "-" means no modifier: use ""."""


class LLMServiceLine(BaseModel):
    date_of_service: str = Field(description="YYYY-MM-DD, or '' if blank")
    cpt_code: str = Field(description="5-character CPT/HCPCS code, or ''")
    modifier: str = Field(description="2-character modifier, or '' if none")
    diagnosis_pointer: str = Field(description="Letters such as 'A' or 'AB', or ''")
    units: int = Field(description="Units billed; 0 if blank")
    charge: str = Field(description="Line charge, e.g. '125.00', or ''")


class LLMClaim(BaseModel):
    """Flat, all-required schema sent to Claude."""

    claim_id: str = Field(description="Patient account / claim number")
    patient_name: str = Field(description="Patient name as printed, 'Last, First'")
    patient_dob: str = Field(description="Patient date of birth, YYYY-MM-DD")
    member_id: str = Field(description="Insured's member/ID number")
    payer_name: str = Field(description="Insurance company name")
    provider_name: str = Field(description="Billing provider name")
    provider_npi: str = Field(description="Billing provider NPI, 10 digits")
    diagnosis_codes: list[str] = Field(description="ICD-10-CM codes in printed order (A, B, C, ...)")
    service_lines: list[LLMServiceLine]
    total_charge: str = Field(description="Total charge, e.g. '1250.00'")
    amount_paid: str = Field(description="Amount paid, e.g. '0.00'")
    uncertain_fields: list[str] = Field(description="Fields that were blank, illegible, ambiguous or guessed")


def _text(value: str) -> str | None:
    return value.strip() or None


def _money(value: str, field: str, uncertain: list[str]) -> Decimal | None:
    if not value.strip():
        return None
    try:
        return Decimal(value.replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        uncertain.append(field)
        return None


def _date(value: str, field: str, uncertain: list[str]) -> date | None:
    if not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        uncertain.append(field)
        return None


def to_claim_form(llm: LLMClaim) -> tuple[ClaimForm, list[str]]:
    """Convert Claude's flat output to a ClaimForm; returns it with the uncertain fields."""
    uncertain = list(llm.uncertain_fields)
    lines = []
    for i, line in enumerate(llm.service_lines):
        prefix = f"service_lines[{i}]"
        lines.append(
            ServiceLine(
                date_of_service=_date(line.date_of_service, f"{prefix}.date_of_service", uncertain),
                cpt_code=_text(line.cpt_code),
                modifier=_text(line.modifier),
                diagnosis_pointer=_text(line.diagnosis_pointer),
                units=line.units or None,
                charge=_money(line.charge, f"{prefix}.charge", uncertain),
            )
        )
    claim = ClaimForm(
        claim_id=_text(llm.claim_id),
        patient_name=_text(llm.patient_name),
        patient_dob=_date(llm.patient_dob, "patient_dob", uncertain),
        member_id=_text(llm.member_id),
        payer_name=_text(llm.payer_name),
        provider_name=_text(llm.provider_name),
        provider_npi=_text(llm.provider_npi),
        diagnosis_codes=[code.strip() for code in llm.diagnosis_codes if code.strip()],
        service_lines=lines,
        total_charge=_money(llm.total_charge, "total_charge", uncertain),
        amount_paid=_money(llm.amount_paid, "amount_paid", uncertain),
    )
    return claim, list(dict.fromkeys(uncertain))


def _cost(model: str, usage) -> float | None:
    prices = PRICES.get(model)
    if prices is None or usage is None:
        return None
    input_tokens = (
        (usage.input_tokens or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * 1.25
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * 0.1
    )
    return (input_tokens * prices[0] + (usage.output_tokens or 0) * prices[1]) / 1_000_000


def _content_blocks(document: LoadedDocument) -> list[dict]:
    if document.content_type == "application/pdf":
        return [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(document.data).decode(),
                },
            }
        ]
    blocks = []
    for page in document.pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(buf.getvalue()).decode(),
                },
            }
        )
    return blocks


class ClaudeVisionExtractor:
    name = "llm"

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client
        settings = get_settings()
        self.model = settings.llm_model
        self.effort = settings.llm_effort

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def extract(self, document: LoadedDocument) -> ExtractionResult:
        content = _content_blocks(document)
        content.append({"type": "text", "text": "Extract the claim from this document."})
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_format=LLMClaim,
                output_config={"effort": self.effort},
                # If the model declines, the API retries on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.CredentialsError as exc:
            raise ExtractionError(f"Anthropic credentials could not be loaded: {exc}") from exc
        except TypeError as exc:
            # The SDK raises TypeError (not an API error) when no credential is configured.
            if self.client.api_key is None and self.client.auth_token is None:
                raise ExtractionError("no Anthropic credentials configured; set ANTHROPIC_API_KEY") from exc
            raise
        except anthropic.AuthenticationError as exc:
            raise ExtractionError("Anthropic credentials are missing or invalid") from exc
        except anthropic.BadRequestError as exc:
            raise ExtractionError(f"Claude rejected the request: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise ExtractionError("rate limited by the Claude API; retry later") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(f"Claude API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionError("could not reach the Claude API") from exc

        if response.stop_reason == "refusal":
            raise ExtractionError("Claude declined to process this document")
        if response.stop_reason == "max_tokens":
            raise ExtractionError("Claude's response was cut off (max_tokens)")
        parsed = response.parsed_output
        if parsed is None:
            raise ExtractionError("Claude returned no structured output")

        claim, uncertain = to_claim_form(parsed)
        return ExtractionResult(
            claim=claim,
            extractor=self.name,
            uncertain_fields=uncertain,
            cost_usd=_cost(getattr(response, "model", self.model), getattr(response, "usage", None)),
        )
