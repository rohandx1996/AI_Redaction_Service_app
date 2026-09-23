"""Evaluate match-span pipeline outputs against original-vs-published ground truth."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def load_truth(path: Path, manifest_path: Path) -> dict[str, list[dict]]:
    truth = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder_to_original = {
        Path(item["match_folder"]).name: Path(item["original"]).name.casefold()
        for item in manifest.get("matches", [])
    }
    by_original = {}
    for document in truth.get("documents", []):
        key = Path(document.get("original_pdf", "")).name.casefold()
        by_original[key] = [
            {"page": int(item.get("page_number", 0)), "value": item.get("original_value", ""), "norm": norm(item.get("original_value", "")), "category": item.get("category", "")}
            for item in document.get("redacted_information", []) or []
            if item.get("original_value")
        ]
    return {folder: by_original.get(original, []) for folder, original in folder_to_original.items()}


def load_predictions(batch_root: Path) -> list[dict]:
    manifest = json.loads((batch_root / "manifest.json").read_text(encoding="utf-8"))
    return [
        json.loads((batch_root / next_path / "pipeline_result.json").read_text(encoding="utf-8"))
        for next_path in [
            path.name for path in batch_root.iterdir()
            if path.is_dir() and (path / "pipeline_result.json").exists()
        ]
    ]


def evaluate_document(document: dict, truth: list[dict]) -> tuple[dict, list[dict]]:
    predictions = []
    for page in document.get("pages", []):
        for mapping in page.get("mapping", {}).get("mappings", []) or []:
            predictions.append({"page": int(page["page_number"]) + 1, "value": mapping.get("value", ""), "norm": norm(mapping.get("value", "")), "category": mapping.get("category", "")})
    remaining = list(predictions)
    matches = []
    missed = []
    for item in truth:
        index = next((i for i, candidate in enumerate(remaining) if candidate["page"] == item["page"] and candidate["norm"] == item["norm"]), None)
        if index is None:
            missed.append(item)
        else:
            matches.append((item, remaining.pop(index)))
    errors = [
        {"Document": document.get("document", ""), "Page": item["page"], "Error Type": "False Negative", "Category": item["category"], "Value": item["value"]}
        for item in missed
    ] + [
        {"Document": document.get("document", ""), "Page": item["page"], "Error Type": "False Positive", "Category": item["category"], "Value": item["value"]}
        for item in remaining
    ]
    truth_count = len(truth)
    prediction_count = len(predictions)
    true_positive = len(matches)
    return {
        "Document": document.get("document", ""), "Pages": document.get("page_count", 0), "Ground Truth Entities": truth_count,
        "Predicted Entities": prediction_count, "True Positives": true_positive,
        "Entity Recall": true_positive / truth_count if truth_count else 0,
        "Entity Precision": true_positive / prediction_count if prediction_count else 0,
        "Entity F1 Score": 2 * true_positive / (truth_count + prediction_count) if truth_count + prediction_count else 0,
        "Exact-value Match Rate": true_positive / truth_count if truth_count else 0,
        "False-negative Rate": len(missed) / truth_count if truth_count else 0,
        "False-positive Rate": len(remaining) / prediction_count if prediction_count else 0,
        "Mapped Redactions": document.get("mapped_redactions", 0),
        "Extraction/OCR/Mapping Seconds": document.get("timings_sec", {}).get("extraction_ocr_mapping", 0),
        "Redaction/Audit Seconds": document.get("timings_sec", {}).get("redaction_audit", 0),
        "Document Total Seconds": document.get("timings_sec", {}).get("document_total", 0),
        "PDF Audit Status": document.get("audit", {}).get("status", ""),
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=Path("output/matched_original_counterparts/matches_20260916_214743/litellm_all_original_vs_published_redactions.json"))
    parser.add_argument("--match-manifest", type=Path, default=Path("output/matched_original_counterparts/matches_20260916_214743/match_manifest.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.batch_root / "ground_truth_evaluation.xlsx"
    truth_by_folder = load_truth(args.ground_truth, args.match_manifest)
    rows, errors = [], []
    for document in load_predictions(args.batch_root):
        row, document_errors = evaluate_document(document, truth_by_folder.get(document.get("document", ""), []))
        rows.append(row)
        errors.extend(document_errors)
    summary = pd.DataFrame([{
        "Documents Evaluated": len(rows), "Ground Truth Entities": sum(row["Ground Truth Entities"] for row in rows), "Predicted Entities": sum(row["Predicted Entities"] for row in rows), "True Positives": sum(row["True Positives"] for row in rows), "Entity Recall": sum(row["True Positives"] for row in rows) / sum(row["Ground Truth Entities"] for row in rows) if rows else 0, "Entity Precision": sum(row["True Positives"] for row in rows) / sum(row["Predicted Entities"] for row in rows) if rows else 0, "Entity F1 Score": 2 * sum(row["True Positives"] for row in rows) / (sum(row["Ground Truth Entities"] for row in rows) + sum(row["Predicted Entities"] for row in rows)) if rows else 0, "Span-index Rate": "N/A: ground truth has no span_indices", "Total Seconds": sum(row["Document Total Seconds"] for row in rows), "Average Seconds/Document": sum(row["Document Total Seconds"] for row in rows) / len(rows) if rows else 0,
    }])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, "Summary", index=False, startrow=1)
        pd.DataFrame(rows).to_excel(writer, "Document Metrics", index=False, startrow=1)
        pd.DataFrame(errors).to_excel(writer, "Error Details", index=False, startrow=1)
    print(json.dumps({"output": str(output), "documents": len(rows), "errors": len(errors), "true_positives": sum(row["True Positives"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
