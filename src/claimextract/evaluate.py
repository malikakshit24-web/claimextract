"""Score an extractor against generated answer keys.

Reports field-level accuracy, document-level exact match, and how routing did:
documents with an injected defect should go to review, and clean documents
extracted correctly should pass straight through.

Usage:
    uv run claimextract-evaluate --data data/synthetic --extractor ocr
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from claimextract.documents import load_document
from claimextract.extractors import ExtractionError, get_extractor
from claimextract.schemas import ClaimForm, DocumentStatus
from claimextract.service import route
from claimextract.validation import validate_claim

SCALAR_FIELDS = [
    "claim_id",
    "patient_name",
    "patient_dob",
    "member_id",
    "payer_name",
    "provider_name",
    "provider_npi",
    "total_charge",
    "amount_paid",
]
LINE_FIELDS = ["date_of_service", "cpt_code", "modifier", "diagnosis_pointer", "units", "charge"]


def compare(expected: ClaimForm, actual: ClaimForm) -> dict[str, bool]:
    """Field name -> correct? Line items are compared position by position."""
    results = {name: getattr(expected, name) == getattr(actual, name) for name in SCALAR_FIELDS}
    results["diagnosis_codes"] = expected.diagnosis_codes == actual.diagnosis_codes
    for i, exp_line in enumerate(expected.service_lines):
        act_line = actual.service_lines[i] if i < len(actual.service_lines) else None
        for name in LINE_FIELDS:
            ok = act_line is not None and getattr(exp_line, name) == getattr(act_line, name)
            results[f"service_lines.{name}"] = results.get(f"service_lines.{name}", True) and ok
    results["service_lines.count"] = len(expected.service_lines) == len(actual.service_lines)
    return results


def evaluate(data_dir: Path, extractor_name: str) -> dict:
    extractor = get_extractor(extractor_name)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    per_field: dict[str, list[bool]] = defaultdict(list)
    by_kind: dict[str, list[bool]] = defaultdict(list)
    exact, failures, latencies = 0, [], []
    defect_total = defect_caught = clean_correct = clean_passed = 0

    for entry in manifest:
        expected = ClaimForm.model_validate(entry["claim"])
        document = load_document((data_dir / entry["file"]).read_bytes())
        start = time.perf_counter()
        try:
            result = extractor.extract(document)
        except ExtractionError as exc:
            failures.append({"file": entry["file"], "error": str(exc)})
            continue
        latencies.append(time.perf_counter() - start)

        fields = compare(expected, result.claim)
        for name, ok in fields.items():
            per_field[name].append(ok)
        all_ok = all(fields.values())
        exact += all_ok
        by_kind[entry["kind"]].append(all_ok)

        status = route(validate_claim(result.claim), result.uncertain_fields)
        if entry["defects"]:
            defect_total += 1
            defect_caught += status == DocumentStatus.NEEDS_REVIEW
        elif all_ok:
            clean_correct += 1
            clean_passed += status == DocumentStatus.VALIDATED

    scored = len(manifest) - len(failures)
    pct = lambda n, d: round(100 * n / d, 1) if d else None  # noqa: E731
    return {
        "extractor": extractor_name,
        "documents": len(manifest),
        "failed": len(failures),
        "exact_match_pct": pct(exact, scored),
        "exact_match_by_kind_pct": {k: pct(sum(v), len(v)) for k, v in by_kind.items()},
        "field_accuracy_pct": {k: pct(sum(v), len(v)) for k, v in sorted(per_field.items())},
        "defects_sent_to_review": f"{defect_caught}/{defect_total}",
        "clean_correct_auto_validated": f"{clean_passed}/{clean_correct}",
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "failures": failures[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an extractor against answer keys")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--extractor", default="ocr", choices=["ocr", "llm"])
    parser.add_argument("--out", type=Path, help="also write the report as JSON")
    args = parser.parse_args()
    report = evaluate(args.data, args.extractor)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)
