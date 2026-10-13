"""Minimal LLM client (P2-T04): text generation via Google AI Studio."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _post_json(url: str, body: bytes, headers: dict, timeout: int,
               retries: int, label: str) -> dict:
    """POST JSON with retries on transient errors (429/5xx)."""
    req = urllib.request.Request(url, data=body, headers=headers)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries:
                wait = 5 * (2**attempt)
                print(
                    f"{label} HTTP {e.code}, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{retries})...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise


def _generate_openrouter(prompt: str, key: str, retries: int) -> str:
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini-2024-07-18")
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    data = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        body,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        timeout=90,
        retries=retries,
        label="OpenRouter",
    )
    return data["choices"][0]["message"]["content"]


def _generate_gemini(prompt: str, model: str, key: str, retries: int) -> str:
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    data = _post_json(
        url,
        body,
        {"Content-Type": "application/json"},
        timeout=90,
        retries=retries,
        label="LLM",
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def generate(prompt: str, model: str = "gemini-2.5-flash", retries: int = 3) -> str:
    """Generate text. Uses OpenRouter if OPENROUTER_API_KEY is set,
    otherwise Google AI Studio (GEMINI_API_KEY).

    Retries transient errors (429/5xx) with exponential backoff so agents
    survive rate limits instead of crashing.
    """
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        return _generate_openrouter(prompt, or_key, retries)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "set OPENROUTER_API_KEY or GEMINI_API_KEY in .env"
        )
    return _generate_gemini(prompt, model, key, retries)
