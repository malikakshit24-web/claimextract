"""Synthetic CMS-1500-style claim forms with answer keys.

Every generated document comes with the exact ClaimForm that was printed on it,
so extraction accuracy can be measured. Optional defects (wrong totals, bad NPIs,
missing fields) are recorded in the answer key so validation can be scored too.

Usage:
    uv run claimextract-generate --count 50 --out data/synthetic --scan-rate 0.5
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from faker import Faker
from PIL import ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claimextract.schemas import ClaimForm, ServiceLine

PAYERS = [
    "Blue Ridge Health Plan",
    "Northstar Mutual Insurance",
    "Evergreen Medicare Advantage",
    "Summit Commercial PPO",
    "Harbor Medicaid Managed Care",
]
# (CPT, description, typical charge)
PROCEDURES = [
    ("99213", "Office visit, est. patient, low", Decimal("125.00")),
    ("99214", "Office visit, est. patient, moderate", Decimal("185.00")),
    ("99203", "Office visit, new patient, low", Decimal("160.00")),
    ("93000", "Electrocardiogram, complete", Decimal("65.00")),
    ("85025", "Complete blood count", Decimal("32.50")),
    ("80053", "Comprehensive metabolic panel", Decimal("48.75")),
    ("71046", "Chest X-ray, 2 views", Decimal("110.00")),
    ("90471", "Immunization administration", Decimal("28.00")),
    ("J1100", "Dexamethasone injection", Decimal("12.40")),
    ("97110", "Therapeutic exercise, 15 min", Decimal("55.00")),
]
DIAGNOSES = ["E11.9", "I10", "J06.9", "M54.50", "E78.5", "R07.9", "Z00.00", "J45.909", "K21.9", "F41.1"]
MODIFIERS = [None, None, None, "25", "59"]
DEFECTS = ("wrong_total", "invalid_npi", "missing_member_id", "bad_icd10")

FORM_TITLE = "HEALTH INSURANCE CLAIM FORM (CMS-1500 STYLE - SYNTHETIC)"


def make_npi(rng: random.Random) -> str:
    """Random 10-digit NPI with a valid Luhn check digit."""
    base = "1" + "".join(str(rng.randint(0, 9)) for _ in range(8))
    digits = [int(d) for d in "80840" + base]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return base + str((10 - total % 10) % 10)


@dataclass
class GeneratedClaim:
    claim: ClaimForm
    defects: list[str] = field(default_factory=list)


def random_claim(rng: random.Random, fake: Faker, defect_rate: float = 0.0) -> GeneratedClaim:
    dob = date(1940, 1, 1) + timedelta(days=rng.randint(0, 365 * 65))
    service_day = date(2026, 1, 1) + timedelta(days=rng.randint(0, 270))
    dx = rng.sample(DIAGNOSES, rng.randint(1, 4))
    lines = []
    for _ in range(rng.randint(1, 5)):
        cpt, _desc, price = rng.choice(PROCEDURES)
        units = rng.choice([1, 1, 1, 2, 3])
        pointer = "".join(sorted(rng.sample([chr(65 + i) for i in range(len(dx))], rng.randint(1, len(dx)))))
        lines.append(
            ServiceLine(
                date_of_service=service_day,
                cpt_code=cpt,
                modifier=rng.choice(MODIFIERS),
                diagnosis_pointer=pointer,
                units=units,
                charge=price * units,
            )
        )
    total = sum((line.charge for line in lines), Decimal("0"))
    paid = rng.choice([Decimal("0.00"), Decimal("0.00"), (total * Decimal("0.2")).quantize(Decimal("0.01"))])
    claim = ClaimForm(
        claim_id=f"CLM-{rng.randint(100000, 999999)}",
        patient_name=f"{fake.last_name()}, {fake.first_name()}",
        patient_dob=dob,
        member_id=f"{rng.choice('ABCDEFGHJK')}{rng.randint(10**8, 10**9 - 1)}",
        payer_name=rng.choice(PAYERS),
        provider_name=f"{fake.last_name()} Family Medicine",
        provider_npi=make_npi(rng),
        diagnosis_codes=dx,
        service_lines=lines,
        total_charge=total,
        amount_paid=paid,
    )

    defects: list[str] = []
    if rng.random() < defect_rate:
        defect = rng.choice(DEFECTS)
        defects.append(defect)
        if defect == "wrong_total":
            claim.total_charge = total + Decimal(rng.choice(["10.00", "0.50", "100.00"]))
        elif defect == "invalid_npi":
            npi = claim.provider_npi
            claim.provider_npi = npi[:9] + str((int(npi[9]) + 1) % 10)
        elif defect == "missing_member_id":
            claim.member_id = None
        elif defect == "bad_icd10":
            claim.diagnosis_codes[0] = "U" + claim.diagnosis_codes[0][1:]
    return GeneratedClaim(claim=claim, defects=defects)


def _fmt_date(d: date | None) -> str:
    return d.strftime("%m/%d/%Y") if d else ""


def _fmt_money(m: Decimal | None) -> str:
    return f"{m:.2f}" if m is not None else ""


def render_pdf(claim: ClaimForm) -> bytes:
    """Draw the claim as a one-page form. Each field is 'N. LABEL: value' in a box."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    left, right = 36, width - 36

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, height - 40, FORM_TITLE)
    c.setLineWidth(0.6)

    def box(x: float, y: float, w: float, label: str, value: str) -> None:
        c.rect(x, y - 6, w, 20)
        c.setFont("Helvetica", 9)
        c.drawString(x + 4, y, f"{label}: {value}")

    y = height - 75
    half = (right - left) / 2
    rows = [
        (("1. PAYER", claim.payer_name or ""), ("1a. INSURED ID", claim.member_id or "")),
        (("2. PATIENT NAME", claim.patient_name or ""), ("3. PATIENT DOB", _fmt_date(claim.patient_dob))),
        (("26. PATIENT ACCOUNT NO", claim.claim_id or ""), ("33. BILLING PROVIDER", claim.provider_name or "")),
        (("33a. NPI", claim.provider_npi or ""), ("29. AMOUNT PAID", _fmt_money(claim.amount_paid))),
    ]
    for (l1, v1), (l2, v2) in rows:
        box(left, y, half, l1, v1)
        box(left + half, y, half, l2, v2)
        y -= 24

    dx = "  ".join(f"{chr(65 + i)}. {code}" for i, code in enumerate(claim.diagnosis_codes))
    box(left, y, right - left, "21. DIAGNOSIS CODES", dx)
    y -= 34

    c.setFont("Helvetica-Bold", 9)
    c.drawString(left, y, "24. SERVICE LINES")
    y -= 16
    cols = [left, left + 90, left + 160, left + 210, left + 280, left + 330]
    headers = ["DATE OF SERVICE", "CPT/HCPCS", "MOD", "DX POINTER", "UNITS", "CHARGES"]
    c.setFont("Helvetica-Bold", 8)
    for x, h in zip(cols, headers, strict=True):
        c.drawString(x, y, h)
    c.setFont("Helvetica", 9)
    for line in claim.service_lines:
        y -= 16
        cells = [
            _fmt_date(line.date_of_service),
            line.cpt_code or "",
            line.modifier or "-",
            line.diagnosis_pointer or "",
            str(line.units) if line.units is not None else "",
            _fmt_money(line.charge),
        ]
        for x, cell in zip(cols, cells, strict=True):
            c.drawString(x, y, cell)
    y -= 30
    box(left, y, half, "28. TOTAL CHARGE", _fmt_money(claim.total_charge))
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(left, 30, "Synthetic document for testing. Contains no real patient information.")
    c.showPage()
    c.save()
    return buf.getvalue()


def pdf_to_scan(pdf_bytes: bytes, rng: random.Random) -> bytes:
    """Rasterise the PDF and degrade it like a fax/scan. Returns PNG bytes."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    image = pdf[0].render(scale=rng.choice([1.6, 2.0, 2.4])).to_pil().convert("L")
    pdf.close()
    image = image.rotate(rng.uniform(-2.0, 2.0), expand=True, fillcolor=255)
    if rng.random() < 0.6:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
    # salt-and-pepper speckle
    pixels = image.load()
    w, h = image.size
    for _ in range(int(w * h * 0.001)):
        pixels[rng.randrange(w), rng.randrange(h)] = rng.choice([0, 255])
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def generate(count: int, out: Path, seed: int = 7, scan_rate: float = 0.3, defect_rate: float = 0.2) -> list[dict]:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(count):
        gen = random_claim(rng, fake, defect_rate)
        pdf = render_pdf(gen.claim)
        stem = f"claim_{i:04d}"
        if rng.random() < scan_rate:
            filename, data, kind = f"{stem}.png", pdf_to_scan(pdf, rng), "scan"
        else:
            filename, data, kind = f"{stem}.pdf", pdf, "digital"
        (out / filename).write_bytes(data)
        entry = {
            "file": filename,
            "kind": kind,
            "defects": gen.defects,
            "claim": gen.claim.model_dump(mode="json"),
        }
        (out / f"{stem}.json").write_text(json.dumps(entry, indent=2))
        manifest.append(entry)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic claim forms with answer keys")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scan-rate", type=float, default=0.3, help="share rendered as noisy scans")
    parser.add_argument("--defect-rate", type=float, default=0.2, help="share with an injected defect")
    args = parser.parse_args()
    manifest = generate(args.count, args.out, args.seed, args.scan_rate, args.defect_rate)
    scans = sum(1 for m in manifest if m["kind"] == "scan")
    defects = sum(1 for m in manifest if m["defects"])
    print(f"Wrote {len(manifest)} documents to {args.out} ({scans} scans, {defects} with defects)")
