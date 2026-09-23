"""Write one consolidated OCR JSON export per selected document."""

from __future__ import annotations

import json
from pathlib import Path

from aggregate_ocr_json import build_export

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output" / "reports" / "consolidated_ocr"
DOCUMENTS = [
    "01_OSAT_Files__ASE__BMM350__Updated__ACC_SAC405_95_5SN_4_0AG_0_5CU_20260829__Security_C__pdf_049d60f47f",
    "01_OSAT_Files__ASE__BMM350__Updated__FUJ_LTC9320_LTC9310_E07_20260411__Security_C__pdf_367ebfca71",
    "01_OSAT_Files__ASE__BMM350__Updated__DOW_Purified_CuSO4_20240616__Security_C__pdf_94936babda",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for document_name in DOCUMENTS:
        export = build_export([document_name])
        output_path = OUTPUT_DIR / f"{document_name}_ocr_text.json"
        output_path.write_text(
            json.dumps(export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {export['total_page_count']} pages: {output_path}")


if __name__ == "__main__":
    main()
