from dataclasses import dataclass
from pathlib import Path
import os

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    GOT_OCR_MODEL_PATH: str = "models/got_ocr"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    INPUT_DIR: str = "input"
    OUTPUT_DIR: str = "output"
    LOGS_DIR: str = "logs"
    REDACTION_CONFIDENCE_THRESHOLD: float = 0.7


def load_settings() -> Settings:
    settings = Settings(
        GOT_OCR_MODEL_PATH=os.getenv("GOT_OCR_MODEL_PATH", "models/got_ocr"),
        OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", "llama3.2"),
        OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        INPUT_DIR=os.getenv("INPUT_DIR", "input"),
        OUTPUT_DIR=os.getenv("OUTPUT_DIR", "output"),
        LOGS_DIR=os.getenv("LOGS_DIR", "logs"),
        REDACTION_CONFIDENCE_THRESHOLD=float(os.getenv("REDACTION_CONFIDENCE_THRESHOLD", "0.7")),
    )

    project_root = Path(__file__).resolve().parent
    model_path = (project_root / settings.GOT_OCR_MODEL_PATH).resolve()
    if not model_path.exists():
        print(f"WARNING: GOT_OCR_MODEL_PATH does not exist on disk: {model_path}")

    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code != 200:
            print(f"WARNING: Ollama is not reachable at {settings.OLLAMA_BASE_URL}")
    except requests.RequestException:
        print(f"WARNING: Ollama is not reachable at {settings.OLLAMA_BASE_URL}")

    return settings
