import hashlib
from datetime import datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def hash_file(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_event(message: str, log_dir: str | Path, file_name: str = "pipeline.log") -> Path:
    directory = ensure_dir(log_dir)
    log_path = directory / file_name
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    return log_path
