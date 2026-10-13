"""Embedding backends (P2-T02).

Two backends behind one interface:
- ApiEmbedder: OpenAI-compatible /embeddings (OpenAI, OpenRouter, ...).
  Light, no local models. API key from .env.
- LocalEmbedder: sentence-transformers (e.g. BAAI/bge-m3, the ADR pick).
  Heavy (torch); imported lazily so the module loads without it.
"""

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


class Embedder:
    """Common interface. dim is set after the first embed() call."""

    dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class ApiEmbedder(Embedder):
    """Embeddings via an OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")
        self._key = key

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        items = sorted(data["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in items]
        self.dim = len(vectors[0])
        return vectors


class LocalEmbedder(Embedder):
    """Local embeddings via sentence-transformers (e.g. BAAI/bge-m3)."""

    def __init__(self, model: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed: pip install sentence-transformers torch"
            )
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class GeminiEmbedder(Embedder):
    """Embeddings via Google AI Studio (Gemini API). Free tier, generous quota."""

    def __init__(self, model: str = "gemini-embedding-001"):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY in .env (from Google AI Studio)")
        self.model = model
        self._key = key

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 100):  # batch endpoint limit
            batch = texts[i : i + 100]
            body = json.dumps(
                {
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": t}]},
                        }
                        for t in batch
                    ]
                }
            ).encode()
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{self.model}:batchEmbedContents?key={self._key}"
            )
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            vectors.extend(e["values"] for e in data["embeddings"])
        self.dim = len(vectors[0])
        return vectors
