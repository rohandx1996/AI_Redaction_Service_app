"""Source-checkout launcher for the installed ``ocr-redact`` command."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redaction_flow.cli import main


if __name__ == "__main__":
    main()
