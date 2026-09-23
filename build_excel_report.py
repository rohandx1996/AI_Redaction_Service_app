"""Turn output/reports/*.csv into a manager-facing Excel workbook.

Sheets: Executive Summary, Poor Performers (worst documents highlighted),
All Documents, Top Missed Words, Per-Page Detail, Chinese/Partial exclusions.
Each sheet has a title banner, styled header, banded rows, and color-scale
highlighting on accuracy/error columns (green = good, red = bad).
"""

from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

REPORTS_DIR = Path("output/reports")
POOR_RECALL_THRESHOLD = 0.85  # documents below this token-recall are flagged as poor performers (redaction risk)

TITLE_FILL = PatternFill("solid", fgColor="0B2C4A")
TITLE_FONT = Font(color="FFFFFF", bold=True, size=14)
HEADER_FILL = PatternFill("solid", fgColor="1F6FB2")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
BAND_FILL = PatternFill("solid", fgColor="EAF1FA")
THIN_SIDE = Side(style="thin", color="C9D6E3")
THIN_BORDER = Border(THIN_SIDE, THIN_SIDE, THIN_SIDE, THIN_SIDE)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center")

# Green (good) -> Yellow -> Red (bad), used for accuracy columns
GREEN_TO_RED = ColorScaleRule(
    start_type="min", start_color="F8696B",
    mid_type="percentile", mid_value=50, mid_color="FFEB84",
    end_type="max", end_color="63BE7B",
)
# Red (bad/high error) -> Yellow -> Green (good/low error), used for CER/WER
RED_TO_GREEN = ColorScaleRule(
    start_type="min", start_color="63BE7B",
    mid_type="percentile", mid_value=50, mid_color="FFEB84",
    end_type="max", end_color="F8696B",
)

DOC_RENAME = {
    "document": "Document", "pages": "Pages", "empty_pages": "Empty Pages (0 detections)",
    "avg_recall": "Avg Token Recall", "avg_precision": "Avg Token Precision",
    "avg_token_f1": "Avg Token Accuracy (F1)", "avg_cer": "Avg CER", "avg_wer": "Avg WER",
    "avg_confidence": "Avg Model Confidence", "worst_page": "Worst Page",
}
PAGE_RENAME = {
    "document": "Document", "page": "Page", "has_baseline": "Has Ground Truth",
    "empty_page": "Empty Page (0 detections)", "token_recall": "Token Recall",
    "token_precision": "Token Precision", "token_f1": "Token Accuracy (F1)",
    "cer": "CER", "wer": "WER", "avg_confidence": "Avg Model Confidence",
}


def autofit(ws: Worksheet, df: pd.DataFrame, header_row: int, max_width: int = 55) -> None:
    for i, col in enumerate(df.columns, start=1):
        max_len = df[col].apply(lambda x: len(str(x))).max() if len(df) else 0
        width = min(max(len(str(col)), max_len) + 3, max_width)
        ws.column_dimensions[get_column_letter(i)].width = max(width, 10)
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1).coordinate


def write_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    title: str,
    subtitle: str = "",
    pct_cols: list[str] | None = None,
    good_high_cols: list[str] | None = None,
    good_low_cols: list[str] | None = None,
) -> None:
    """Write df to a formatted sheet with a title banner, styled header, and highlighting."""
    header_row = 3 if subtitle else 2
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=header_row - 1)
    ws = writer.sheets[sheet_name]
    ncols = max(df.shape[1], 1)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    title_cell = ws.cell(row=1, column=1, value=title)
    title_cell.fill = TITLE_FILL
    title_cell.font = TITLE_FONT
    title_cell.alignment = LEFT
    ws.row_dimensions[1].height = 24

    if subtitle:
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
        sub_cell = ws.cell(row=2, column=1, value=subtitle)
        sub_cell.font = Font(italic=True, color="4C5B6B")
        sub_cell.alignment = LEFT

    for col in range(1, ncols + 1):
        cell = ws.cell(row=header_row, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = THIN_BORDER
    ws.row_dimensions[header_row].height = 20

    last_row = header_row + len(df)
    for row in range(header_row + 1, last_row + 1):
        band = (row - header_row) % 2 == 0
        for col in range(1, ncols + 1):
            cell = ws.cell(row=row, column=col)
            cell.border = THIN_BORDER
            cell.alignment = CENTER if col > 1 else LEFT
            if band:
                cell.fill = BAND_FILL

    autofit(ws, df, header_row)

    col_names = list(df.columns)
    for col_name in pct_cols or []:
        if col_name not in col_names:
            continue
        col_idx = col_names.index(col_name) + 1
        for row in range(header_row + 1, last_row + 1):
            ws.cell(row=row, column=col_idx).number_format = "0.0%"

    def col_range(col_name: str) -> str | None:
        if col_name not in col_names or len(df) == 0:
            return None
        col_idx = col_names.index(col_name) + 1
        letter = get_column_letter(col_idx)
        return f"{letter}{header_row + 1}:{letter}{last_row}"

    for col_name in good_high_cols or []:
        rng = col_range(col_name)
        if rng:
            ws.conditional_formatting.add(rng, GREEN_TO_RED)
    for col_name in good_low_cols or []:
        rng = col_range(col_name)
        if rng:
            ws.conditional_formatting.add(rng, RED_TO_GREEN)


def build_per_document(per_page: pd.DataFrame) -> pd.DataFrame:
    scored = per_page[per_page["has_baseline"]]
    grouped = scored.groupby("document").agg(
        pages=("page", "count"),
        avg_recall=("token_recall", "mean"),
        avg_precision=("token_precision", "mean"),
        avg_token_f1=("token_f1", "mean"),
        avg_cer=("cer", "mean"),
        avg_wer=("wer", "mean"),
        avg_confidence=("avg_confidence", "mean"),
    ).reset_index()
    empty_pages = per_page.groupby("document")["empty_page"].sum().reset_index().rename(columns={"empty_page": "empty_pages"})
    grouped = grouped.merge(empty_pages, on="document", how="left")
    worst_page = scored.loc[scored.groupby("document")["token_recall"].idxmin(), ["document", "page"]]
    worst_page = worst_page.rename(columns={"page": "worst_page"})
    grouped = grouped.merge(worst_page, on="document")
    grouped = grouped.sort_values("avg_recall")
    return grouped


def parse_top_missed(report_md: str) -> pd.DataFrame:
    in_table = False
    rows = []
    for line in report_md.splitlines():
        if line.startswith("| Word | Times Missed |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 2 and parts[1].isdigit():
                rows.append((parts[0], int(parts[1])))
    return pd.DataFrame(rows, columns=["Word (ground truth)", "Times Missed"])


def parse_llm_narrative(report_md: str) -> pd.DataFrame:
    lines = report_md.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("## Qualitative Insights")), None)
    if start is None:
        return pd.DataFrame(columns=["LLM Executive Narrative"])
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    paragraphs = [line for line in lines[start + 1:end] if line.strip()]
    return pd.DataFrame(paragraphs, columns=["LLM Executive Narrative"])


def main() -> None:
    per_page = pd.read_csv(REPORTS_DIR / "ocr_accuracy_per_page.csv")
    chinese = pd.read_csv(REPORTS_DIR / "chinese_documents_excluded.csv")
    partial = pd.read_csv(REPORTS_DIR / "partial_documents.csv")

    per_doc = build_per_document(per_page)
    poor_docs = per_doc[per_doc["avg_recall"] < POOR_RECALL_THRESHOLD]
    scored_pages = per_page[per_page["has_baseline"]]

    summary_df = pd.DataFrame([
        ("Documents analyzed (non-Chinese, fully processed)", len(per_doc)),
        ("Documents excluded - Chinese language", len(chinese)),
        ("Documents excluded - partial/in-progress", len(partial)),
        ("Pages scored (usable ground truth)", int(scored_pages.shape[0])),
        ("Pages with no ground truth (image-only)", int((~per_page["has_baseline"]).sum())),
        ("Pages with zero OCR detections (redaction risk)", int(per_page["empty_page"].sum())),
        ("Overall Token Recall - primary redaction-risk metric, higher is better", f"{scored_pages['token_recall'].mean():.2%}"),
        ("Overall Token Precision - higher is better", f"{scored_pages['token_precision'].mean():.2%}"),
        ("Overall Token-level Accuracy (F1) - higher is better", f"{scored_pages['token_f1'].mean():.2%}"),
        ("Overall Character Error Rate (CER) - secondary diagnostic, lower is better", f"{scored_pages['cer'].mean():.2%}"),
        ("Overall Word Error Rate (WER) - secondary diagnostic, lower is better", f"{scored_pages['wer'].mean():.2%}"),
        ("Poor-performing documents (Token Recall < 85%, redaction risk)", len(poor_docs)),
        ("Best document (Token Recall)", f"{per_doc.iloc[-1]['document']} ({per_doc.iloc[-1]['avg_recall']:.2%})"),
        ("Worst document (Token Recall)", f"{per_doc.iloc[0]['document']} ({per_doc.iloc[0]['avg_recall']:.2%})"),
    ], columns=["Metric", "Value"])

    top_missed_df = parse_top_missed((REPORTS_DIR / "ocr_accuracy_report.md").read_text(encoding="utf-8"))
    narrative_df = parse_llm_narrative((REPORTS_DIR / "ocr_accuracy_report.md").read_text(encoding="utf-8"))
    llm_notes_path = REPORTS_DIR / "llm_validations.csv"
    llm_notes_out = pd.read_csv(llm_notes_path).rename(columns={
        "document": "Document", "page": "Page", "has_baseline": "Has Ground Truth", "note": "LLM Note",
    }) if llm_notes_path.exists() else pd.DataFrame(columns=["Document", "Page", "Has Ground Truth", "LLM Note"])

    poor_docs_out = poor_docs.rename(columns=DOC_RENAME)
    per_doc_out = per_doc.rename(columns=DOC_RENAME)
    per_page_out = per_page.rename(columns=PAGE_RENAME)
    chinese_out = chinese.rename(columns={"folder_name": "Document Folder", "source_pdf": "Source PDF"})
    partial_out = partial.rename(columns={
        "folder_name": "Document Folder", "actual_pages": "Pages Done",
        "expected_pages": "Total Pages", "source_pdf": "Source PDF",
    })

    doc_pct_cols = ["Avg Token Recall", "Avg Token Precision", "Avg Token Accuracy (F1)", "Avg CER", "Avg WER", "Avg Model Confidence"]
    page_pct_cols = ["Token Recall", "Token Precision", "Token Accuracy (F1)", "CER", "WER", "Avg Model Confidence"]

    with pd.ExcelWriter(REPORTS_DIR / "ocr_accuracy_report.xlsx", engine="openpyxl") as writer:
        write_sheet(
            writer, summary_df, "Executive Summary",
            title="PaddleOCR Text Recognition Accuracy - Executive Summary",
            subtitle="Token Recall is the primary redaction-risk metric (did OCR recover the word at all). CER/WER are secondary diagnostics only.",
        )
        write_sheet(
            writer, poor_docs_out, "Poor Performers",
            title="Poor-Performing Documents (Token Recall below 85%)",
            subtitle="Review these first - highest redaction risk (words OCR failed to recover), sorted worst to best.",
            pct_cols=doc_pct_cols, good_high_cols=["Avg Token Recall", "Avg Token Precision", "Avg Token Accuracy (F1)"], good_low_cols=["Avg CER", "Avg WER"],
        )
        write_sheet(
            writer, per_doc_out, "All Documents",
            title="All Documents - Accuracy Ranking (Worst to Best by Token Recall)",
            pct_cols=doc_pct_cols, good_high_cols=["Avg Token Recall", "Avg Token Precision", "Avg Token Accuracy (F1)"], good_low_cols=["Avg CER", "Avg WER"],
        )
        write_sheet(
            writer, top_missed_df, "Top Missed Words",
            title="Most Frequently Missed / Misrecognized Words",
            subtitle="Ground-truth words PaddleOCR failed to recognize, ranked by frequency across all pages.",
        )
        write_sheet(
            writer, narrative_df, "Qualitative Insights",
            title="Qualitative Insights",
            subtitle="AI narrative synthesis - currently skipped (pure PyMuPDF-based evaluation run); enable SKIP_LLM=False to regenerate via gateway.",
        )
        write_sheet(
            writer, llm_notes_out, "LLM Validation Notes",
            title="LLM Spot-Check Validation (Worst Page per Document + No-Baseline Pages)",
            subtitle="Currently skipped (pure PyMuPDF-based evaluation run); enable SKIP_LLM=False to regenerate via gateway.",
        )
        write_sheet(
            writer, per_page_out, "Per-Page Detail",
            title="Per-Page Accuracy Detail",
            subtitle="Granular page-level metrics for drill-down investigation. Token Recall is the primary redaction-risk column.",
            pct_cols=page_pct_cols, good_high_cols=["Token Recall", "Token Precision", "Token Accuracy (F1)"], good_low_cols=["CER", "WER"],
        )
        write_sheet(
            writer, chinese_out, "Chinese Docs (Excluded)",
            title="Chinese-Language Documents (Excluded from Scoring)",
            subtitle="Counted and logged only, per request - not evaluated for OCR accuracy.",
        )
        write_sheet(
            writer, partial_out, "Partial (In-Progress)",
            title="Partially Processed Documents",
            subtitle="Document still mid-processing when the batch job was stopped - excluded from metrics.",
        )

    print(f"Excel report written to {REPORTS_DIR / 'ocr_accuracy_report.xlsx'}", flush=True)


if __name__ == "__main__":
    main()
