"""Group original PDFs with high-confidence published and redacted counterparts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil


DEFAULT_CORPUS_ROOT = Path(r"C:\Users\DHR1RT\Downloads\OneDrive_1_9-7-2026")
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "output" / "matched_original_counterparts"
STATUS_SUFFIX = re.compile(r"(?i)(?:[\s_-]*(?:published|redacted|original))(?:\s*\(\d+\))?$")
IDENTIFIER_PATTERNS = (
    re.compile(r"(?i)(?:EKR|ETR|JER|SHAEC|CANEC|TSNEC)[A-Z0-9-]{7,}"),
    re.compile(r"(?i)CPSA[-_A-Z0-9]{7,}"),
    re.compile(r"(?i)A\d{15,}"),
    re.compile(r"(?<!\d)\d{7,}[-_]\d{1,3}(?!\d)"),
)


def normalized_stem(path: Path) -> str:
    stem = STATUS_SUFFIX.sub("", path.stem)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def identifiers(path: Path) -> set[str]:
    values = set()
    for pattern in IDENTIFIER_PATTERNS:
        for match in pattern.findall(path.stem):
            value = re.sub(r"[^A-Z0-9]+", "", match.upper())
            if len(value) >= 8:
                values.add(value)
    return values


def similarity(left: Path, right: Path) -> float:
    return SequenceMatcher(None, normalized_stem(left), normalized_stem(right)).ratio()


def find_match(
    counterpart: Path,
    originals: list[Path],
    exact_index: dict[str, list[Path]],
    id_index: dict[str, list[Path]],
) -> tuple[Path | None, str | None, float, list[Path]]:
    exact = exact_index.get(normalized_stem(counterpart), [])
    if len(exact) == 1:
        return exact[0], "normalized_stem", 1.0, []
    if len(exact) > 1:
        return None, None, 0.0, exact

    candidates = set()
    for identifier in identifiers(counterpart):
        candidates.update(id_index.get(identifier, []))
    if not candidates:
        return None, None, 0.0, []

    ranked = sorted(
        ((similarity(counterpart, original), original) for original in candidates),
        key=lambda item: (-item[0], str(item[1]).lower()),
    )
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score >= 0.72 and best_score - second_score >= 0.08:
        return best, "report_id_and_name_similarity", round(best_score, 4), []
    return None, None, round(best_score, 4), [item[1] for item in ranked]


def safe_folder_name(index: int, original: Path) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", STATUS_SUFFIX.sub("", original.stem))
    return f"{index:04d}_{readable[:170]}"


def copy_with_role(source: Path, destination: Path, role: str, ordinal: int) -> Path:
    suffix = source.suffix.lower() or ".bin"
    target = destination / f"{role}_{ordinal:02d}{suffix}"
    shutil.copy2(source, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    original_root = args.corpus_root / "02_Original"
    counterpart_roots = {
        "published": args.corpus_root / "00_Published",
        "redacted": args.corpus_root / "01_Redacted",
    }
    originals = sorted(original_root.rglob("*.pdf"))
    exact_index: dict[str, list[Path]] = defaultdict(list)
    id_index: dict[str, list[Path]] = defaultdict(list)
    for original in originals:
        exact_index[normalized_stem(original)].append(original)
        for identifier in identifiers(original):
            id_index[identifier].append(original)

    grouped: dict[Path, list[dict]] = defaultdict(list)
    ambiguous = []
    unmatched = []
    for role, root in counterpart_roots.items():
        for counterpart in sorted(root.rglob("*.pdf")):
            original, method, score, candidates = find_match(counterpart, originals, exact_index, id_index)
            if original:
                grouped[original].append({
                    "role": role,
                    "path": counterpart,
                    "method": method,
                    "score": score,
                    "shared_identifiers": sorted(identifiers(original) & identifiers(counterpart)),
                })
            elif candidates:
                ambiguous.append({
                    "role": role,
                    "counterpart": str(counterpart),
                    "best_score": score,
                    "candidates": [str(path) for path in candidates[:10]],
                })
            else:
                unmatched.append({"role": role, "counterpart": str(counterpart)})

    run_id = datetime.now().strftime("matches_%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    match_records = []
    for index, (original, matches) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]).lower()), 1):
        match_dir = output_dir / safe_folder_name(index, original)
        copied_files = []
        if not args.dry_run:
            match_dir.mkdir(parents=True, exist_ok=True)
            copied_original = copy_with_role(original, match_dir, "original", 1)
            copied_files.append(str(copied_original.relative_to(output_dir)))
            role_counts = defaultdict(int)
            for match in sorted(matches, key=lambda item: (item["role"], str(item["path"]).lower())):
                role_counts[match["role"]] += 1
                copied = copy_with_role(match["path"], match_dir, match["role"], role_counts[match["role"]])
                copied_files.append(str(copied.relative_to(output_dir)))
        match_records.append({
            "original": str(original),
            "match_folder": str(match_dir),
            "counterparts": [
                {
                    "role": match["role"],
                    "path": str(match["path"]),
                    "method": match["method"],
                    "score": match["score"],
                    "shared_identifiers": match["shared_identifiers"],
                }
                for match in matches
            ],
            "copied_files": copied_files,
        })

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_root": str(args.corpus_root),
        "output_folder": str(output_dir),
        "dry_run": args.dry_run,
        "original_pdf_count": len(originals),
        "matched_original_count": len(grouped),
        "matched_counterpart_count": sum(len(matches) for matches in grouped.values()),
        "published_match_count": sum(match["role"] == "published" for matches in grouped.values() for match in matches),
        "redacted_match_count": sum(match["role"] == "redacted" for matches in grouped.values() for match in matches),
        "ambiguous_counterpart_count": len(ambiguous),
        "unmatched_counterpart_count": len(unmatched),
        "matches": match_records,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "match_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "output_folder",
        "dry_run",
        "original_pdf_count",
        "matched_original_count",
        "matched_counterpart_count",
        "published_match_count",
        "redacted_match_count",
        "ambiguous_counterpart_count",
        "unmatched_counterpart_count",
    )}, indent=2))


if __name__ == "__main__":
    main()