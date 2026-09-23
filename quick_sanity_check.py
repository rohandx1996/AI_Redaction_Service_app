"""Rough, no-ground-truth sanity pass over every page JSON in paddle_dbnet_svtr_pages.

Uses PaddleOCR's own per-span confidence plus a lightweight "does this look like
a real word" heuristic (vowel presence, no absurd repeats) as a proxy for
whether the extracted text is plausible. No external dictionary required.
"""

from pathlib import Path
import json
import re

ROOT = Path("output/batch/paddle_dbnet_svtr_pages")

WORD_RE = re.compile(r"[A-Za-z]{2,}")
VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
REPEAT_RE = re.compile(r"(.)\1{3,}")  # 4+ repeated same char = garbage signal


def plausible_word(tok: str) -> bool:
    if REPEAT_RE.search(tok):
        return False
    if not VOWEL_RE.search(tok) and len(tok) > 3:
        return False
    return True


def analyze_doc(doc_dir: Path) -> dict:
    page_dirs = sorted(p for p in doc_dir.iterdir() if p.is_dir() and p.name.startswith("page_"))
    total_conf = 0.0
    n_spans = 0
    low_conf_spans = 0
    plausible_words = 0
    total_words = 0
    n_pages = 0
    empty_pages = 0
    sample_texts = []

    for page_dir in page_dirs:
        json_path = page_dir / f"{page_dir.name}.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n_pages += 1
        spans = data.get("spans", [])
        if not spans:
            empty_pages += 1
            continue
        page_words = []
        for span in spans:
            conf = span.get("confidence", 0.0)
            total_conf += conf
            n_spans += 1
            if conf < 0.5:
                low_conf_spans += 1
            text = span.get("text", "")
            page_words.append(text)
            for tok in WORD_RE.findall(text):
                total_words += 1
                if plausible_word(tok):
                    plausible_words += 1
        if len(sample_texts) < 2:
            sample_texts.append(" ".join(page_words)[:180])

    avg_conf = total_conf / n_spans if n_spans else 0.0
    low_conf_ratio = low_conf_spans / n_spans if n_spans else 0.0
    word_plausibility = plausible_words / total_words if total_words else 0.0

    if avg_conf >= 0.90 and word_plausibility >= 0.85:
        verdict = "GOOD"
    elif avg_conf >= 0.80 and word_plausibility >= 0.70:
        verdict = "FAIR"
    else:
        verdict = "POOR"

    return {
        "document": doc_dir.name,
        "pages": n_pages,
        "empty_pages": empty_pages,
        "avg_confidence": round(avg_conf, 4),
        "low_conf_ratio": round(low_conf_ratio, 4),
        "word_plausibility": round(word_plausibility, 4),
        "verdict": verdict,
        "samples": sample_texts,
    }


def main():
    results = []
    for doc_dir in sorted(ROOT.iterdir()):
        if doc_dir.is_dir():
            results.append(analyze_doc(doc_dir))

    results.sort(key=lambda r: (r["verdict"] != "POOR", r["verdict"] != "FAIR", r["avg_confidence"]))

    print(f"Total documents: {len(results)}")
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("Verdict counts:", counts)
    print()
    print(f"{'VERDICT':6} {'AVGCONF':8} {'LOWCONF%':9} {'WORDPLAUS%':11} {'PAGES':6} {'EMPTY':6} DOCUMENT")
    for r in results:
        print(
            f"{r['verdict']:6} {r['avg_confidence']:.4f}   {r['low_conf_ratio']*100:6.2f}%   "
            f"{r['word_plausibility']*100:8.2f}%  {r['pages']:5} {r['empty_pages']:5}  {r['document']}"
        )

    out_path = Path("output/reports/rough_sanity_check.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote details to {out_path}")


if __name__ == "__main__":
    main()
