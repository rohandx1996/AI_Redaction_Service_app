"""Small Anthropic-compatible Bosch Gateway client with model-specific request settings."""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import requests

from dotenv import load_dotenv

load_dotenv()


class GatewayError(RuntimeError):
    pass


class VisionGateway:
    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: int = 240, retries: int = 4):
        self.base_url = (base_url or os.getenv("BOSCH_GATEWAY_URL", "")).rstrip("/")
        self.token = token or os.getenv("BOSCH_GATEWAY_TOKEN", "")
        self.timeout = timeout
        self.retries = retries
        if not self.base_url or not self.token:
            raise ValueError("Set BOSCH_GATEWAY_URL and BOSCH_GATEWAY_TOKEN in the environment")

    def complete(self, model: str, system: str, prompt: str, image: bytes | None = None,
                 max_tokens: int = 4096) -> tuple[dict[str, Any], dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(image).decode("ascii"),
            }})
        content.append({"type": "text", "text": prompt})
        body: dict[str, Any] = {
            "model": model, "system": system, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if "haiku" in model.casefold():
            body["temperature"] = 0
        headers = {"x-api-key": self.token, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(f"{self.base_url}/v1/messages", headers=headers,
                                         json=body, timeout=self.timeout)
                if response.status_code in (429, 500, 502, 503, 529):
                    raise GatewayError(f"Gateway HTTP {response.status_code}: {response.text[:400]}")
                response.raise_for_status()
                payload = response.json()
                text = "".join(block.get("text", "") for block in payload.get("content", [])
                               if block.get("type") == "text")
                return self._parse_json(text), payload.get("usage", {})
            except (requests.RequestException, GatewayError, ValueError) as exc:
                last_error = exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code < 500:
                    break
                if attempt < self.retries:
                    time.sleep(min(45, 2 ** attempt * 2))
        raise GatewayError(f"Model call failed: {last_error}") from last_error

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        value = text.strip()
        if value.startswith("```"):
            value = value.strip("`").strip()
            if value[:4].casefold() == "json":
                value = value[4:].strip()
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Model response did not contain a JSON object")
        parsed = json.loads(value[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON response must be an object")
        return parsed
