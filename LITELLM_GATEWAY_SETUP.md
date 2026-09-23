# Using the Bosch AI Gateway via LiteLLM — Correct Setup

## 1. Required config in `.env`

```env
BOSCH_GATEWAY_URL=https://aigateway.datadriven-sensors.bosch.tech
BOSCH_GATEWAY_TOKEN=<your token>
LLM_MODEL=gpt-4o
```

Never paste the real token into a chat conversation — set it only in local `.env`. If it's ever pasted somewhere, treat it as compromised and rotate it.

## 2. Required network proxy — the #1 thing that breaks it

```powershell
$env:HTTP_PROXY='http://rb-proxy-de.bosch.com:8080'
$env:HTTPS_PROXY='http://rb-proxy-de.bosch.com:8080'
```

Set these in the **same shell/process** before running Python. Bosch machines block direct outbound HTTPS; everything must go through this corporate proxy. Using a different proxy (e.g. `127.0.0.1:8888`, a local dev proxy) or no proxy at all will fail with connection errors.

## 3. Correct LiteLLM call pattern

```python
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('BOSCH_GATEWAY_URL')
token = os.getenv('BOSCH_GATEWAY_TOKEN')
model = os.getenv('LLM_MODEL', 'gpt-4o')

import litellm
resp = litellm.completion(
    model=f'openai/{model}',           # must prefix with "openai/" — it's an OpenAI-compatible endpoint
    api_base=url.rstrip('/') + '/v1',  # must append /v1
    api_key=token,
    messages=[{'role': 'user', 'content': 'Say hello in one word.'}],
    temperature=0,
    max_tokens=10,
)
print(resp['choices'][0]['message']['content'])
```

## 4. Two-stage verification (do this if it's not working)

- **Stage 1 (no token, just reachability):** `GET {url}/v1/models` — expect `401 Missing API key`. If you get a connection/DNS error instead, the proxy isn't set correctly.
- **Stage 2 (with token):** the full `litellm.completion(...)` call above — expect a real response with `choices[0].message.content`.

## Where things go wrong (all observed directly)

1. **Wrong or missing proxy.** Using `127.0.0.1:8888` (a local dev proxy) or no proxy at all fails with connection errors — it must be `rb-proxy-de.bosch.com:8080`.
2. **Missing `openai/` model prefix.** LiteLLM needs `openai/<model>`, not just `<model>`, when calling a custom OpenAI-compatible `api_base`.
3. **Missing `/v1` suffix on `api_base`.** The gateway URL from email doesn't include `/v1` — you must append it yourself.
4. **Token pasted into chat.** Never paste the real token into a conversation — set it only in local `.env`. If it's ever pasted, treat it as compromised and rotate it.
5. **`litellm.__version__` doesn't exist.** Don't use that attribute to check if litellm is installed — it raises `AttributeError`. Just try importing and calling `litellm.completion(...)` directly.
6. **Sandboxed execution environment (E_NOTIMPL error).** On some machines, sandboxed process creation is broken; commands must run unsandboxed. Unrelated to the gateway itself, but causes confusing failures if not recognized.
7. **Remote model-cost-map fetch warning.** LiteLLM tries to fetch `model_prices_and_context_window.json` from GitHub at import time and fails without proxy — this is a harmless warning, not a real error, and doesn't block completions.
