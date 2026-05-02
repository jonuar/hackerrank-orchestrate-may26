#!/usr/bin/env python3
"""
Support triage agent — entry point.
Reads support_issues/support_issues.csv, processes each ticket,
writes results to support_issues/output.csv.

Usage:
    python main.py
    python main.py --input ../support_issues/sample_support_issues.csv  # for dev
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

from classifier import classify
from retriever  import retrieve, format_context
from generator  import generate

# Paths
REPO_ROOT   = Path(__file__).parent.parent
INPUT_CSV   = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV  = REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_FIELDS = ["status", "product_area", "response", "justification", "request_type"]

REQUEST_DELAY = 4.5


def process_row(row: dict) -> dict:
    """Run the full pipeline for a single CSV row."""
    issue   = row.get("issue", "").strip()
    subject = row.get("subject", "").strip()
    company = row.get("company", "None").strip()

    # 1. Classify
    classification = classify(issue, subject, company)

    # 2. Retrieve
    query    = f"{issue} {subject}".strip()
    retrieval = retrieve(query, company=classification.company)
    context  = format_context(retrieval)

    # 3. Generate
    output = generate(
        issue=issue,
        subject=subject,
        company=classification.company,
        product_area=classification.product_area,
        request_type=classification.request_type,
        is_high_risk=classification.is_high_risk,
        risk_reasons=classification.risk_reasons,
        corpus_context=context,
    )

    return {
        "status":        output.status,
        "product_area":  output.product_area,
        "response":      output.response,
        "justification": output.justification,
        "request_type":  output.request_type,
    }


def run(input_path: Path, output_path: Path) -> None:
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"Processing {total} tickets from {input_path.name} ...")

    results = []
    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{total}] {row.get('subject', '(no subject)')[:60]}", end=" ... ", flush=True)
        try:
            result = process_row(row)
            print(f"{result['status'].upper()}")
        except Exception as e:
            print(f"ERROR: {e}")
            result = {
                "status":        "escalated",
                "product_area":  "general_inquiry",
                "response":      "Unable to process. A human agent will follow up.",
                "justification": f"Pipeline error: {e}",
                "request_type":  "product_issue",
            }
        results.append(result)

        # Rate limit guard
        if i < total:
            time.sleep(REQUEST_DELAY)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Output written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Support triage agent")
    parser.add_argument(
        "--input", type=Path, default=INPUT_CSV,
        help="Path to input CSV (default: support_issues/support_issues.csv)"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Path to output CSV (default: support_issues/output.csv)"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(args.input, args.output)


if __name__ == "__main__":
    main()