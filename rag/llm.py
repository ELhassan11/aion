"""Minimal LLM client (P2-T04): text generation via Google AI Studio."""

from __future__ import annotations

import json
import os
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


def generate(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Generate text with Gemini. Needs GEMINI_API_KEY in .env."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY in .env (from Google AI Studio)")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)
    return data["candidates"][0]["content"]["parts"][0]["text"]
