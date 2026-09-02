import requests


def ask_llm_for_redaction_decision(prompt: str, base_url: str, model: str):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - placeholder
        return {"error": str(exc)}
