"""Vector index (P2-T02): cosine-similarity search over chunk embeddings.

Build an index from a DocumentChunk JSONL, save it, and query it:

    python -m rag.index --input out/chunks_split.jsonl --output out/index.npz \\
        --base-url https://openrouter.ai/api/v1 --model openai/text-embedding-3-small
    python -m rag.index --search out/index.npz --query "سياسة الاستبدال" --top-k 3 \\
        --base-url https://openrouter.ai/api/v1 --model openai/text-embedding-3-small

Embeddings are L2-normalized, so cosine similarity is a dot product.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import DocumentChunk  # noqa: E402
from rag.embeddings import ApiEmbedder, Embedder, GeminiEmbedder, LocalEmbedder  # noqa: E402


class VectorIndex:
    def __init__(self, ids: list[str], vectors: np.ndarray, texts: list[str]):
        self.ids = ids
        self.texts = texts
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)

    @classmethod
    def build(
        cls, chunks_path: Path, embedder: Embedder, batch_size: int = 32
    ) -> "VectorIndex":
        ids, texts = [], []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = DocumentChunk.model_validate_json(line)
                    ids.append(chunk.id)
                    texts.append(chunk.text)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(embedder.embed(texts[i : i + batch_size]))
        return cls(ids, np.array(vectors, dtype=np.float32), texts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ids=np.array(self.ids),
            vectors=self.vectors,
            texts=np.array(self.texts, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        data = np.load(path, allow_pickle=True)
        return cls(
            [str(i) for i in data["ids"]],
            np.array(data["vectors"], dtype=np.float32),
            [str(t) for t in data["texts"]],
        )

    def search(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[str, float, str]]:
        q = np.array(query_vector, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = self.vectors @ q
        top = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i]), self.texts[i]) for i in top]


def _make_embedder(args) -> Embedder:
    if args.embedder == "local":
        return LocalEmbedder(args.model or "BAAI/bge-m3")
    if args.embedder == "gemini":
        return GeminiEmbedder(args.model or "gemini-embedding-001")
    return ApiEmbedder(args.model or "openai/text-embedding-3-small",
                       base_url=args.base_url)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vector index (P2-T02)")
    ap.add_argument("--input", help="input DocumentChunk JSONL (build mode)")
    ap.add_argument("--output", help="index .npz path (build mode)")
    ap.add_argument("--search", help="index .npz path (search mode)")
    ap.add_argument("--query", help="query text (search mode)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embedder", choices=["api", "local", "gemini"], default="api")
    ap.add_argument("--model", default=None,
                    help="embedding model (default depends on backend)")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    args = ap.parse_args()

    embedder = _make_embedder(args)

    if args.search:
        if not args.query:
            ap.error("--query is required in search mode")
        index = VectorIndex.load(Path(args.search))
        (qvec,) = embedder.embed([args.query])
        for rank, (cid, score, text) in enumerate(index.search(qvec, k=args.top_k), 1):
            print(f"#{rank} [{score:.3f}] {cid}")
            print(text[:200].replace("\n", " "))
            print()
    else:
        if not args.input or not args.output:
            ap.error("--input and --output are required in build mode")
        index = VectorIndex.build(Path(args.input), embedder)
        index.save(Path(args.output))
        print(f"indexed {len(index.ids)} chunks (dim={embedder.dim}) -> {args.output}")


if __name__ == "__main__":
    main()
