from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    got_ocr_model_path: str = "models/got_ocr"
    bosch_gateway_url: str = ""
    bosch_gateway_token: str = ""
    llm_model: str = "gpt-4o"
    INPUT_DIR: str = "input"
    OUTPUT_DIR: str = "output"
    LOGS_DIR: str = "logs"
    REDACTION_CONFIDENCE_THRESHOLD: float = 0.7


def load_settings() -> Settings:
    settings = Settings(
        got_ocr_model_path=os.getenv("GOT_OCR_MODEL_PATH", "models/got_ocr"),
        bosch_gateway_url=os.getenv("BOSCH_GATEWAY_URL", ""),
        bosch_gateway_token=os.getenv("BOSCH_GATEWAY_TOKEN", ""),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
        INPUT_DIR=os.getenv("INPUT_DIR", "input"),
        OUTPUT_DIR=os.getenv("OUTPUT_DIR", "output"),
        LOGS_DIR=os.getenv("LOGS_DIR", "logs"),
        REDACTION_CONFIDENCE_THRESHOLD=float(os.getenv("REDACTION_CONFIDENCE_THRESHOLD", "0.7")),
    )

    project_root = Path(__file__).resolve().parent
    model_path = (project_root / settings.got_ocr_model_path).resolve()
    if not model_path.exists():
        print(f"WARNING: GOT_OCR_MODEL_PATH does not exist on disk: {model_path}")

    return settings


settings = load_settings()
