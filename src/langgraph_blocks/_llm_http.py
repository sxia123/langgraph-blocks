"""Shared stdlib HTTP transport helpers for API and Custom Endpoint LLM calls."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# --- EMBED START ---
_DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _resolve_endpoint_url(endpoint_url: str) -> str:
    """Normalize an OpenAI-compatible custom endpoint base or full completions URL."""
    clean = endpoint_url.strip().rstrip("/")
    suffixes = ("/chat/completions", "/completions", "/responses", "/messages")
    if clean.endswith(suffixes):
        return clean
    return f"{clean}/chat/completions"


def _extract_llm_text(provider_kind: str, data: Any) -> str:
    """Extract assistant text cleanly from OpenAI, Anthropic, Gemini, or custom JSON payloads."""
    if not isinstance(data, dict):
        return str(data)
    if provider_kind == "anthropic":
        parts = [p.get("text", "") for p in data.get("content", []) if isinstance(p, dict)]
        return "".join(parts) or str(data.get("completion", ""))
    if provider_kind == "gemini":
        cands = data.get("candidates") or []
        if cands and isinstance(cands[0], dict):
            parts = cands[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "")
        if "text" in choices[0]:
            return str(choices[0]["text"] or "")
    msg = data.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return str(msg["content"] or "")
    for key in ("response", "output", "content", "text"):
        if key in data and isinstance(data[key], (str, int, float)):
            return str(data[key])
    return json.dumps(data)


def _resolve_api_key(
    llm_mode: str,
    api_provider: str,
    api_key_env: str,
    runtime_api_key: str = "",
) -> str:
    """Resolve an API key from transient runtime input or an environment variable."""
    if runtime_api_key:
        return runtime_api_key
    default_env = _DEFAULT_API_KEY_ENVS.get(api_provider, "") if llm_mode == "api" else ""
    env_name = api_key_env.strip() or default_env
    return os.environ.get(env_name, "").strip() if env_name else ""


def _http_llm_call(
    prompt: str,
    model: str,
    *,
    llm_mode: str = "api",
    api_provider: str = "openai",
    api_key_env: str = "",
    endpoint_url: str = "",
    api_key: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str = "",
    timeout: float = 60.0,
) -> str | None:
    """Call an API provider or custom HTTP endpoint using Python stdlib only.

    Returns None when offline simulation should be used (no custom endpoint_url
    configured and no API key resolved in runtime input or environment).
    """
    mode = (llm_mode or "api").strip()
    provider = (api_provider or "openai").strip()
    token = _resolve_api_key(mode, provider, api_key_env, api_key)

    if mode == "custom_endpoint":
        if not endpoint_url.strip():
            return None
        url = _resolve_endpoint_url(endpoint_url)
        kind = "openai"
    else:
        if not token:
            return None
        kind = provider
        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
        elif provider == "gemini":
            if not model.strip():
                raise RuntimeError("Gemini API mode requires a model ID.")
            clean_model = urllib.parse.quote(model.strip(), safe="")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
        else:
            url = "https://api.openai.com/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if kind == "anthropic":
        headers["x-api-key"] = token
        headers["anthropic-version"] = "2023-06-01"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens) if max_tokens is not None else 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = float(temperature)
    elif kind == "gemini":
        headers["x-goog-api-key"] = token
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        gen_cfg: dict[str, Any] = {}
        if temperature is not None:
            gen_cfg["temperature"] = float(temperature)
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = int(max_tokens)
        if gen_cfg:
            payload["generationConfig"] = gen_cfg
    else:
        if token:
            headers["Authorization"] = f"Bearer {token}"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        err_detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"LLM endpoint HTTP {exc.code} from {url}: {err_detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM endpoint request to {url} failed: {exc.reason}") from exc

    parsed = json.loads(raw_body)
    return _extract_llm_text(kind, parsed)
