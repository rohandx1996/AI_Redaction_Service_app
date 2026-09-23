"""Infer Bosch redaction policy from matched original and published PDFs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

import fitz


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MATCH_ROOT = REPO_ROOT / "output" / "matched_original_counterparts" / "matches_20260916_214743"

LABEL_ALIASES = {
    "SUBMITTER_OR_CLIENT": [
        "sample submitted by", "submitted and identified by", "provided by the applicant",
        "client name", "customer name", "applicant", "submitted by", "customer",
    ],
    "SAMPLE_OR_PRODUCT_IDENTITY": [
        "sample name", "sample description", "style/item no", "style no", "item no",
        "part number", "product name", "material name", "trade name",
    ],
    "CLIENT_OR_SAMPLE_REFERENCE": [
        "client ref", "customer ref", "order no", "lot no", "sample id",
        "sample no", "reference no",
    ],
    "ADDRESS_OR_LOCATION": ["client address", "customer address", "address", "location", "site"],
    "PERSON_OR_SIGNATURE": ["signed for", "approved signatory", "authorized by", "contact person"],
    "CONTACT_DETAIL": ["telephone", "phone", "fax", "email", "e-mail"],
    "REPORT_ACCESS_IDENTIFIER": ["pin code", "document id", "certificate no"],
}

RETAINED_PATTERNS = {
    "CHEMICAL_OR_CAS": re.compile(
        r"(?i)\b(CAS|cadmium|lead|mercury|chromium|bromine|chlorine|fluorine|iodine|PFAS|PFOS|PFOA|phthalate|PBB|PBDE|HBCDD)\b"
    ),
    "STANDARD_OR_REGULATION": re.compile(r"(?i)\b(IEC|ISO|ASTM|EN\s?\d|EPA|RoHS|REACH|directive|regulation)\b"),
    "TEST_METHOD_OR_INSTRUMENT": re.compile(r"(?i)\b(ICP|GC[/ -]?MS|LC[/ -]?MS|XRF|UV[- /]?VIS|analysis|test method|extraction|digestion|filtration)\b"),
    "MEASUREMENT_OR_RESULT": re.compile(r"(?i)\b(mg/kg|ppm|wt%|MDL|limit|result|n\.d\.|not detected|comply|pass|fail)\b"),
    "PROCEDURE_OR_FLOWCHART": re.compile(r"(?i)\b(flow chart|preparation|pretreatment|procedure|dissolv|measure|separation|residue|solution)\b"),
}

IGNORE_LINES = re.compile(
    r"(?i)^(page\s*:?\s*\d+|test report|test results?|note\s*:?|conclusion\s*:?|={3,}|[-_*\s]+)$"
)
MARKER_RE = re.compile(r"(?i)redacted[ _-]*by[ _-]*bosch")


def clean_line(value: str) -> str:
    return " ".join(value.split()).strip()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def page_lines(document: fitz.Document, page_number: int) -> list[str]:
    return [clean_line(line) for line in document[page_number].get_text("text").splitlines() if clean_line(line)]


def best_similarity(line: str, candidates: list[str]) -> float:
    source = normalized(line)
    if not source:
        return 0.0
    return max(
        (SequenceMatcher(None, source, normalized(candidate)).ratio() for candidate in candidates),
        default=0.0,
    )


def semantic_label(line: str, previous_lines: list[str]) -> tuple[str | None, str | None]:
    context = " | ".join(previous_lines[-3:] + [line]).lower()
    for category, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if alias in context:
                return category, alias
    return None, None


def retained_category(line: str) -> str | None:
    for category, pattern in RETAINED_PATTERNS.items():
        if pattern.search(line):
            return category
    return None


def best_counterpart(group: dict) -> Path | None:
    candidates = [Path(item["path"]) for item in group["counterparts"] if item["role"] == "published"]
    if not candidates:
        candidates = [Path(item["path"]) for item in group["counterparts"] if item["role"] == "redacted"]
    if not candidates:
        return None
    scored = []
    for path in candidates:
        try:
            with fitz.open(str(path)) as document:
                characters = sum(len(page.get_text("text")) for page in document)
            scored.append((characters, path))
        except Exception:
            continue
    return max(scored, default=(0, None), key=lambda item: item[0])[1]


def append_example(target: dict[str, list[dict]], category: str, example: dict, limit: int = 80) -> None:
    if len(target[category]) < limit:
        target[category].append(example)


def analyze(match_root: Path) -> tuple[dict, dict]:
    manifest = json.loads((match_root / "match_manifest.json").read_text(encoding="utf-8"))
    redacted_examples: dict[str, list[dict]] = defaultdict(list)
    retained_examples: dict[str, list[dict]] = defaultdict(list)
    redacted_counts = Counter()
    retained_counts = Counter()
    label_counts: dict[str, Counter] = defaultdict(Counter)
    document_summaries = []
    flattened_pages = 0
    comparable_pages = 0
    marker_pages = 0

    for group in manifest["matches"]:
        original_path = Path(group["original"])
        counterpart_path = best_counterpart(group)
        if counterpart_path is None:
            continue
        document_redacted = Counter()
        document_retained = Counter()
        with fitz.open(str(original_path)) as original, fitz.open(str(counterpart_path)) as counterpart:
            for page_number in range(min(len(original), len(counterpart))):
                original_lines = page_lines(original, page_number)
                counterpart_lines = page_lines(counterpart, page_number)
                original_chars = sum(len(line) for line in original_lines)
                counterpart_chars = sum(len(line) for line in counterpart_lines)
                markers = [line for line in counterpart_lines if MARKER_RE.search(line)]
                marker_pages += bool(markers)
                comparable = original_chars > 0 and counterpart_chars >= original_chars * 0.20
                if not comparable:
                    flattened_pages += 1
                    continue
                comparable_pages += 1
                previous_lines: list[str] = []
                for line in original_lines:
                    score = best_similarity(line, counterpart_lines)
                    keep_category = retained_category(line)
                    if score >= 0.86 and keep_category:
                        retained_counts[keep_category] += 1
                        document_retained[keep_category] += 1
                        append_example(retained_examples, keep_category, {
                            "document": original_path.name,
                            "page": page_number + 1,
                            "original_text": line,
                            "counterpart": counterpart_path.name,
                            "similarity": round(score, 4),
                            "evidence": "text retained in matched counterpart",
                        })
                    category, alias = semantic_label(line, previous_lines)
                    if score < 0.48 and category and len(line) >= 3 and not IGNORE_LINES.match(line):
                        redacted_counts[category] += 1
                        document_redacted[category] += 1
                        label_counts[category][alias or "context"] += 1
                        append_example(redacted_examples, category, {
                            "document": original_path.name,
                            "page": page_number + 1,
                            "original_text": line,
                            "context_before": previous_lines[-3:],
                            "matched_label_alias": alias,
                            "counterpart": counterpart_path.name,
                            "similarity": round(score, 4),
                            "bosch_marker_on_page": markers,
                            "evidence": "context-linked original text absent from text-comparable counterpart",
                        })
                    previous_lines.append(line)
        document_summaries.append({
            "original": str(original_path),
            "counterpart_used": str(counterpart_path),
            "redacted_evidence_counts": dict(document_redacted),
            "retained_evidence_counts": dict(document_retained),
        })

    common_metadata = {
        "source_match_folder": str(match_root),
        "matched_originals_analyzed": len(document_summaries),
        "comparable_pages": comparable_pages,
        "flattened_or_noncomparable_pages": flattened_pages,
        "pages_with_bosch_marker": marker_pages,
        "method_limitations": [
            "Only text-comparable pages contribute value-level evidence.",
            "Flattened/image-only pages are counted but not treated as proof that every missing text line was redacted.",
            "Absence is contextual evidence, not semantic ground truth; examples remain auditable by document and page.",
        ],
    }
    redacted_output = {
        "format": "paired_redacted_information_evidence",
        **common_metadata,
        "decision": "protect identity-bearing supplier/customer/sample/submission information",
        "categories": [
            {
                "category": category,
                "evidence_count": redacted_counts[category],
                "observed_label_aliases": [
                    {"label": label, "count": count}
                    for label, count in label_counts[category].most_common()
                ],
                "examples": redacted_examples[category],
            }
            for category in LABEL_ALIASES
        ],
        "documents": document_summaries,
    }
    retained_output = {
        "format": "paired_retained_information_evidence",
        **common_metadata,
        "decision": "retain technical and scientific knowledge",
        "categories": [
            {
                "category": category,
                "evidence_count": retained_counts[category],
                "examples": retained_examples[category],
            }
            for category in RETAINED_PATTERNS
        ],
        "documents": document_summaries,
    }
    return redacted_output, retained_output


def write_markdown(path: Path, redacted: dict, retained: dict) -> None:
    lines = [
        "# Bosch Redaction Policy From Matched Files",
        "",
        "## Evidence Base",
        "",
        f"- Matched originals analyzed: {redacted['matched_originals_analyzed']}",
        f"- Text-comparable pages: {redacted['comparable_pages']}",
        f"- Flattened or non-comparable pages: {redacted['flattened_or_noncomparable_pages']}",
        f"- Pages containing a Bosch redaction marker: {redacted['pages_with_bosch_marker']}",
        "",
        "The comparison uses matched original and published/redacted PDFs. It does not assume that laboratory-specific wording is universal. Different labels are mapped to the same semantic role only when their nearby context supports that mapping.",
        "",
        "## Redacted Information",
        "",
        "Bosch redaction primarily protects identity-bearing information that links a report to a supplier, customer, applicant, submitted sample, proprietary product, or submission reference.",
        "",
        "| Semantic role | Evidence count | Observed label wording |",
        "|---|---:|---|",
    ]
    for category in redacted["categories"]:
        aliases = ", ".join(item["label"] for item in category["observed_label_aliases"][:8]) or "contextual"
        lines.append(f"| {category['category']} | {category['evidence_count']} | {aliases} |")
    lines.extend([
        "",
        "## Retained Information",
        "",
        "Technical knowledge remains available when it does not itself reveal protected commercial identity.",
        "",
        "| Technical category | Retained evidence count |",
        "|---|---:|",
    ])
    for category in retained["categories"]:
        lines.append(f"| {category['category']} | {category['evidence_count']} |")
    lines.extend([
        "",
        "## Contextual Rules",
        "",
        "1. Redact the value, not a generic field label such as `Sample Name` or `Client Name`.",
        "2. Treat `Sample Submitted By`, `Client Name`, `Applicant`, and equivalent wording as the same submitter/customer role when context supports it.",
        "3. Treat `Style/Item No.`, `Part Number`, `Sample Description`, and similar fields as protected only when they identify the submitted or proprietary test object.",
        "4. Preserve chemicals, CAS numbers, standards, methods, instruments, measurements, limits, results, procedures, flowcharts, and compliance conclusions.",
        "5. Laboratory branding is not automatically equivalent to supplier/customer identity. Its treatment must follow observed Bosch evidence and document context, not the mere presence of a laboratory name.",
        "6. Every repeated protected occurrence needs its own page-specific bbox; identifying a value once is not sufficient for PDF redaction.",
        "",
        "## Important Limitation",
        "",
        "Some published PDFs are flattened or expose only a Bosch marker in the text layer. Those pages are not used as value-level absence proof. The JSON files retain document/page references so every conclusion can be audited.",
        "",
        "## Outputs",
        "",
        "- `redacted_information_evidence.json`: contextual evidence for information Bosch removed.",
        "- `retained_information_evidence.json`: paired evidence for technical information Bosch retained.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-root", type=Path, default=DEFAULT_MATCH_ROOT)
    args = parser.parse_args()
    redacted, retained = analyze(args.match_root)
    redacted_path = args.match_root / "redacted_information_evidence.json"
    retained_path = args.match_root / "retained_information_evidence.json"
    markdown_path = args.match_root / "BOSCH_REDACTION_POLICY_FROM_MATCHED_FILES.md"
    redacted_path.write_text(json.dumps(redacted, ensure_ascii=False, indent=2), encoding="utf-8")
    retained_path.write_text(json.dumps(retained, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(markdown_path, redacted, retained)
    print(json.dumps({
        "redacted_evidence": str(redacted_path),
        "retained_evidence": str(retained_path),
        "markdown": str(markdown_path),
        "matched_originals_analyzed": redacted["matched_originals_analyzed"],
        "comparable_pages": redacted["comparable_pages"],
        "flattened_or_noncomparable_pages": redacted["flattened_or_noncomparable_pages"],
    }, indent=2))


if __name__ == "__main__":
    main()