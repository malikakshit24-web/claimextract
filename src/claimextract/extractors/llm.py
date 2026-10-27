"""Vision-LLM extractor: sends the document to Claude and gets back a validated ClaimForm.

Uses structured outputs (`messages.parse` with a Pydantic model), so the response
is guaranteed to match the schema. The model also lists fields it was unsure
about, which feed the review routing alongside the rule-based checks.

Needs ANTHROPIC_API_KEY (or another Anthropic credential) in the environment.
"""

from __future__ import annotations

import base64
import io

import anthropic

from claimextract.config import get_settings
from claimextract.documents import LoadedDocument
from claimextract.extractors import ExtractionError
from claimextract.schemas import ClaimForm, ExtractionResult, LLMClaimExtraction

SYSTEM_PROMPT = """\
You extract data from healthcare professional claim forms (CMS-1500 style) for a \
claims-processing system. Copy values exactly as printed: do not correct, compute \
or infer them. If the printed total doesn't match the line items, return the printed \
total anyway; downstream validation checks the arithmetic. If a field is blank or \
illegible, return null and add its name to uncertain_fields. Also list any field \
whose value you had to guess between readings (for example a smudged digit). \
Use field names from the schema, with list indexes for line items, e.g. \
"service_lines[1].charge". Dates are ISO (YYYY-MM-DD); money is a plain number \
in dollars with no currency symbol."""


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
                output_format=LLMClaimExtraction,
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

        claim = ClaimForm.model_validate(parsed.model_dump(exclude={"uncertain_fields"}))
        return ExtractionResult(claim=claim, extractor=self.name, uncertain_fields=parsed.uncertain_fields)
