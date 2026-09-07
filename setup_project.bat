@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0ocr_redaction_poc"

if not exist "%ROOT%" mkdir "%ROOT%"
if not exist "%ROOT%\input" mkdir "%ROOT%\input"
if not exist "%ROOT%\output" mkdir "%ROOT%\output"
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
if not exist "%ROOT%\models" mkdir "%ROOT%\models"

rem --- check_env.bat ---
> "%ROOT%\check_env.bat" (
  echo @echo off
  echo echo Checking Python...
  echo python --version
  echo if errorlevel 1 (
  echo   echo ERROR: Python is not installed or not in PATH.
  echo   exit /b 1
  echo )
  echo.
  echo echo Checking pip...
  echo pip --version
  echo if errorlevel 1 (
  echo   echo ERROR: pip is not available.
  echo   exit /b 1
  echo )
  echo.
  echo python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
  echo if errorlevel 1 (
  echo   echo ERROR: Python 3.10 or above is required.
  echo   exit /b 1
  echo )
  echo.
  echo echo Environment OK
)

rem --- requirements.txt ---
> "%ROOT%\requirements.txt" (
  echo pymupdf==1.24.10
  echo python-dotenv==1.0.1
  echo rich==13.9.2
  echo pillow==10.4.0
  echo requests==2.32.3
  echo transformers==4.46.0
  echo torch==2.4.1
  echo torchvision==0.19.1
  echo sentencepiece==0.2.0
  echo tiktoken
  echo verovio
  echo accelerate==1.0.1
  echo pdfplumber==0.11.4
  echo langdetect==1.0.9
  echo openai
)

rem --- install.bat ---
> "%ROOT%\install.bat" (
  echo @echo off
  echo setlocal
  echo.
  echo where python >nul 2>nul
  echo if errorlevel 1 (
  echo    echo ERROR: Python is not installed or not in PATH.
  echo    exit /b 1
  echo )
  echo.
  echo if not exist "venv" (
  echo     echo Creating virtual environment...
  echo     python -m venv venv
  echo )
  echo.
  echo call venv\Scripts\activate.bat
  echo python -m pip install --upgrade pip
  echo pip install -r requirements.txt
  echo.
  echo echo.
  echo echo Step 1: Activate the Conda environment: conda activate ocr-redaction-poc
  echo echo Step 2: Set BOSCH_GATEWAY_URL and BOSCH_GATEWAY_TOKEN in .env
  echo echo Step 3: Put your PDF in the input\ folder
  echo echo Step 4: Run run.bat
)

rem --- run.bat ---
> "%ROOT%\run.bat" (
  echo @echo off
  echo setlocal
  echo call venv\Scripts\activate.bat
  echo python main.py
  echo if errorlevel 1 (
  echo    echo.
  echo    echo ERROR: Pipeline failed. Review the stack trace above.
  echo    pause
  echo    exit /b 1
  echo )
  echo echo.
  echo echo Pipeline completed.
  echo pause
)

rem --- .env ---
> "%ROOT%\.env" (
  echo # Model settings
  echo GOT_OCR_MODEL_PATH=models/got_ocr
  echo BOSCH_GATEWAY_URL=
  echo BOSCH_GATEWAY_TOKEN=
  echo LLM_MODEL=gpt-4o
  echo.
  echo # Pipeline settings
  echo INPUT_DIR=input
  echo OUTPUT_DIR=output
  echo LOGS_DIR=logs
  echo.
  echo # Confidence threshold for LLM redaction decisions
  echo REDACTION_CONFIDENCE_THRESHOLD=0.7
)

rem --- config.py ---
> "%ROOT%\config.py" (
  echo from dataclasses import dataclass
  echo from pathlib import Path
  echo import os
  echo import requests
  echo from dotenv import load_dotenv
  echo.
  echo load_dotenv()
  echo.
  echo @dataclass
  echo class Settings:
  echo     GOT_OCR_MODEL_PATH: str = "models/got_ocr"
  echo     bosch_gateway_url: str = ""
  echo     bosch_gateway_token: str = ""
  echo     llm_model: str = "gpt-4o"
  echo     INPUT_DIR: str = "input"
  echo     OUTPUT_DIR: str = "output"
  echo     LOGS_DIR: str = "logs"
  echo     REDACTION_CONFIDENCE_THRESHOLD: float = 0.7
  echo.
  echo def load_settings() -> Settings:
  echo     settings = Settings(
  echo         GOT_OCR_MODEL_PATH=os.getenv("GOT_OCR_MODEL_PATH", "models/got_ocr"),
  echo         bosch_gateway_url=os.getenv("BOSCH_GATEWAY_URL", ""),
  echo         bosch_gateway_token=os.getenv("BOSCH_GATEWAY_TOKEN", ""),
  echo         llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
  echo         INPUT_DIR=os.getenv("INPUT_DIR", "input"),
  echo         OUTPUT_DIR=os.getenv("OUTPUT_DIR", "output"),
  echo         LOGS_DIR=os.getenv("LOGS_DIR", "logs"),
  echo         REDACTION_CONFIDENCE_THRESHOLD=float(os.getenv("REDACTION_CONFIDENCE_THRESHOLD", "0.7")),
  echo     )
  echo.
  echo     project_root = Path(__file__).resolve().parent
  echo     model_path = (project_root / settings.GOT_OCR_MODEL_PATH).resolve()
  echo     if not model_path.exists():
  echo         print(f"WARNING: GOT_OCR_MODEL_PATH does not exist on disk: {model_path}")
  echo.
  echo     return settings
)

rem --- utils.py ---
> "%ROOT%\utils.py" (
  echo import hashlib
  echo from pathlib import Path
  echo from datetime import datetime
  echo.
  echo def ensure_dir(path: str | Path) -> Path:
  echo     directory = Path(path)
  echo     directory.mkdir(parents=True, exist_ok=True)
  echo     return directory
  echo.
  echo def hash_file(file_path: str | Path) -> str:
  echo     digest = hashlib.sha256()
  echo     with open(file_path, "rb") as handle:
  echo         for chunk in iter(lambda: handle.read(65536), b""):
  echo             digest.update(chunk)
  echo     return digest.hexdigest()
  echo.
  echo def log_event(message: str, log_dir: str | Path, file_name: str = "pipeline.log") -> Path:
  echo     directory = ensure_dir(log_dir)
  echo     log_path = directory / file_name
  echo     timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
  echo     with open(log_path, "a", encoding="utf-8") as handle:
  echo         handle.write(f"[{timestamp}] {message}\n")
  echo     return log_path
)

rem --- main.py ---
> "%ROOT%\main.py" (
  echo from pathlib import Path
  echo from config import load_settings
  echo from utils import ensure_dir, log_event
  echo.
  echo def main() -> None:
  echo     settings = load_settings()
  echo     ensure_dir(settings.INPUT_DIR)
  echo     ensure_dir(settings.OUTPUT_DIR)
  echo     ensure_dir(settings.LOGS_DIR)
  echo     log_event("Pipeline started", settings.LOGS_DIR)
  echo.
  echo     pdf_files = sorted(Path(settings.INPUT_DIR).glob("*.pdf"))
  echo     if not pdf_files:
  echo         print(f"No PDF files found in {settings.INPUT_DIR}. Add a PDF and run again.")
  echo         return
  echo.
  echo     print(f"Found {len(pdf_files)} PDF files.")
  echo     for pdf_path in pdf_files:
  echo         print(f"Processing: {pdf_path.name}")
  echo         log_event(f"Processing PDF: {pdf_path.name}", settings.LOGS_DIR)
  echo.
  echo     print("Pipeline complete.")
  echo     log_event("Pipeline completed", settings.LOGS_DIR)
  echo.
  echo if __name__ == "__main__":
  echo     main()
)

rem --- ocr.py ---
> "%ROOT%\ocr.py" (
  echo from pathlib import Path
  echo. 
  echo def extract_text_from_pdf(pdf_path: str | Path):
  echo     pdf_path = Path(pdf_path)
  echo     print(f"OCR placeholder: would process {pdf_path.name} with GOT-OCR2 or PaddleOCR.")
  echo     return {
  echo         "text": "Placeholder OCR output",
  echo         "pages": 1,
  echo         "confidence": 0.85,
  echo     }
)

rem --- llm.py ---
> "%ROOT%\llm.py" (
  echo import requests
  echo.
  echo def ask_llm_for_redaction_decision(prompt: str, base_url: str, model: str):
  echo     payload = {
  echo         "model": model,
  echo         "prompt": prompt,
  echo         "stream": False,
  echo     }
  echo     try:
  echo         response = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
  echo         response.raise_for_status()
  echo         return response.json()
  echo     except Exception as exc:  # pragma: no cover - placeholder
  echo         return {"error": str(exc)}
)

rem --- redact.py ---
> "%ROOT%\redact.py" (
  echo from pathlib import Path
  echo import fitz
  echo.
  echo def redact_pdf(pdf_path: str | Path, output_path: str | Path, regions: list[dict]):
  echo     input_file = Path(pdf_path)
  echo     output_file = Path(output_path)
  echo     output_file.parent.mkdir(parents=True, exist_ok=True)
  echo.
  echo     document = fitz.open(str(input_file))
  echo     for page_number, region in enumerate(regions, start=1):
  echo         page = document[page_number - 1]
  echo         rect = fitz.Rect(region["x0"], region["y0"], region["x1"], region["y1"])
  echo         page.add_redact_annot(rect, fill=(0, 0, 0))
  echo         page.apply_redactions()
  echo.
  echo     document.save(str(output_file))
  echo     document.close()
  echo     print(f"Redacted PDF saved to: {output_file}")
)

echo Project scaffold created at %ROOT%
