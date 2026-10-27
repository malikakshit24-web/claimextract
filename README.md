# ClaimExtract

**A document-processing API that extracts structured healthcare claim data from PDFs and scanned documents, validates it with accounting and coding rules, and routes anything doubtful to a human review queue.**

Python · FastAPI · Pydantic · PostgreSQL · Tesseract OCR · Claude vision · Docker Compose · GitHub Actions

> All data is synthetic. No real patient information (PHI) is used anywhere in this project.

---

## Why

Most healthcare payment data still arrives as documents: claim forms, remittance advice, letters. People re-type them, or template-based OCR breaks whenever a layout changes. Vision LLMs can read these documents, but they also produce confident mistakes. ClaimExtract answers the practical question: **how much of this work can be automated safely, and how do you catch the cases where the extractor is wrong?**

## How it works

```
 upload (PDF / PNG / JPEG / TIFF)
        │
        ▼
 load: sniff type from bytes ─┬─ digital PDF → embedded text layer
                              └─ scan / image → page images
        │
        ▼
 extract ─┬─ ocr: text (embedded or Tesseract) + label rules   (fast, free baseline)
          └─ llm: Claude vision + structured outputs → ClaimForm (handles messy scans)
        │
        ▼
 validate: required fields · NPI check digit · ICD-10 / CPT formats ·
           line items sum to total · dates · diagnosis pointers · overpayment
        │
        ▼
 route ─┬─ no issues, nothing uncertain → validated
        └─ any issue or uncertain field  → needs_review → human approves / corrects / rejects
        │
        ▼
 PostgreSQL: original extraction, current claim, issues, status, reviewer, timestamps
```

- **One contract for every extractor.** Both extractors return the same Pydantic `ClaimForm`, so they can be compared field by field. Money is `Decimal`, never float.
- **Structured outputs.** The LLM extractor uses `messages.parse` with the Pydantic model, so responses always match the schema. It also lists the fields it was unsure about, and those go to review.
- **Validation doesn't trust the extractor.** The model is told to copy values *as printed*; arithmetic and code checks happen afterwards, in code.
- **Humans can't approve past hard errors.** A reviewer can approve over warnings, but a claim with errors has to be corrected or rejected. The original extraction is kept unchanged next to the correction, for auditing and evaluation.

## Quick start

With Docker:

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY to use the llm extractor
docker compose up --build
```

Then open http://localhost:8000/docs for the interactive API.

Without Docker (needs Python 3.12 and [uv](https://docs.astral.sh/uv/); uses SQLite):

```bash
uv sync
CLAIMEXTRACT_DATABASE_URL=sqlite:///claimextract.db uv run uvicorn claimextract.api:app --reload
```

Scanned images need [Tesseract](https://github.com/tesseract-ocr/tesseract) for the `ocr` extractor. It's already included in the Docker image.

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/documents?extractor=ocr\|llm` | Upload a file; returns the extracted claim, issues and status |
| `GET` | `/documents?status=needs_review` | List documents (the review queue when filtered) |
| `GET` | `/documents/{id}` | One document |
| `POST` | `/documents/{id}/review` | `{"action": "approve" \| "correct" \| "reject", "reviewer": "...", "claim": {...}}` |
| `GET` | `/health` | Liveness and database check |

```bash
curl -F "file=@data/synthetic/claim_0000.pdf" "localhost:8000/documents?extractor=ocr"
```

## Synthetic data and evaluation

Generate CMS-1500-style claim forms with answer keys. Some are rendered as noisy scans (rotation, blur, speckle), and some carry injected defects (wrong total, invalid NPI, missing member ID, malformed ICD-10 code):

```bash
uv run claimextract-generate --count 200 --out data/synthetic --scan-rate 0.3 --defect-rate 0.2
uv run claimextract-evaluate --data data/synthetic --extractor ocr
uv run claimextract-evaluate --data data/synthetic --extractor llm   # needs ANTHROPIC_API_KEY; costs money
```

The evaluation reports field-level accuracy, exact-match rate by document kind (digital vs scan), how many defective documents were sent to review, how many clean documents passed straight through, and latency.

## Results so far

| Measure | OCR + rules | Claude vision |
|---|---|---|
| Exact match, digital PDFs (200 docs) | 100% | not run yet |
| Exact match, noisy scans | not run yet | not run yet |
| Exact match, unseen layouts | not run yet | not run yet |
| Defective documents sent to review | 40 / 40 | not run yet |
| Clean documents auto-validated | 160 / 160 | not run yet |
| Cost per document | $0 | not run yet |

The digital-PDF result is the easy case: the baseline parser reads the embedded text of a layout it was written for. The meaningful comparisons are noisy scans and layouts the rules have never seen. Those come next.

## Development

```bash
uv run pytest            # tests use SQLite; CI also runs them against PostgreSQL
uv run ruff check . && uv run ruff format --check .
```

CI (GitHub Actions) runs lint, the full test suite against PostgreSQL 17 with Tesseract installed, and a Docker build on every push.

## Project layout

```
src/claimextract/
├── api.py            FastAPI routes
├── service.py        pipeline: load → extract → validate → route; review actions
├── documents.py      file loading, type sniffing, PDF text layer vs scan
├── extractors/
│   ├── ocr.py        Tesseract / embedded text + label rules (baseline)
│   └── llm.py        Claude vision with structured outputs
├── validation.py     accounting, coding and date rules
├── schemas.py        Pydantic models (ClaimForm, API payloads)
├── db.py             SQLAlchemy models and sessions
├── generator.py      synthetic claim forms, scans and answer keys
├── evaluate.py       accuracy and routing evaluation
└── config.py         settings from environment variables
```

## Roadmap

- [ ] Run the scan and LLM evaluations; publish cost and latency per document
- [ ] Additional payer layouts, with some held back to test generalisation
- [ ] More document types: Explanation of Benefits, remittance advice
- [ ] Alembic migrations
- [ ] Post approved claims to ClaimLedger through its idempotent API
