from pathlib import Path

from config import load_settings
from utils import ensure_dir, log_event


def main() -> None:
    settings = load_settings()
    ensure_dir(settings.INPUT_DIR)
    ensure_dir(settings.OUTPUT_DIR)
    ensure_dir(settings.LOGS_DIR)
    log_event("Pipeline started", settings.LOGS_DIR)

    pdf_files = sorted(Path(settings.INPUT_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {settings.INPUT_DIR}. Add a PDF and run again.")
        return

    print(f"Found {len(pdf_files)} PDF files.")
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        log_event(f"Processing PDF: {pdf_path.name}", settings.LOGS_DIR)

    print("Pipeline complete.")
    log_event("Pipeline completed", settings.LOGS_DIR)


if __name__ == "__main__":
    main()
