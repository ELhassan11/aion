"""Hybrid search (P2-T03): BM25 keyword search + vector search, fused by RRF.

Vector search understands meaning; BM25 catches exact terms (names,
numbers, product codes). Reciprocal Rank Fusion combines both rankings
without needing score normalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.normalize import normalize_arabic  # noqa: E402


def tokenize(text: str) -> list[str]:
    """Tokenize (Arabic-normalized) text into terms."""
    return re.findall(r"\w+", normalize_arabic(text), re.UNICODE)


class BM25:
    """Okapi BM25 over tokenized documents (stdlib only)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(self.n, 1)
        df: Counter[str] = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_terms: list[str]) -> list[float]:
        out: list[float] = []
        for doc, dl in zip(self.docs, self.doc_lens):
            tf: Counter[str] = Counter(doc)
            score = 0.0
            for term in query_terms:
                if term not in self.idf:
                    continue
                f = tf.get(term, 0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[term] * f * (self.k1 + 1) / max(denom, 1e-12)
            out.append(score)
        return out

    def rank(self, query: str, ids: list[str], k: int) -> list[str]:
        scored = sorted(
            zip(ids, self.scores(tokenize(query)), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [cid for cid, s in scored[:k] if s > 0]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(cid) = sum(1 / (k + rank))."""
    fused: Counter[str] = Counter()
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            fused[cid] += 1.0 / (k + rank)
    return fused.most_common()


def hybrid_search(
    vector_ranking: list[str],
    bm25: BM25,
    ids: list[str],
    query: str,
    k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse a vector ranking with a BM25 ranking via RRF."""
    keyword_ranking = bm25.rank(query, ids, candidate_k)
    return rrf_fuse(
        [vector_ranking[:candidate_k], keyword_ranking], k=rrf_k
    )[:k]


def main() -> None:
    import argparse

    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder
    from rag.index import VectorIndex

    ap = argparse.ArgumentParser(description="Hybrid search: BM25 + vector, RRF fused (P2-T03)")
    ap.add_argument("--index", required=True, help="index .npz path")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embedder", choices=["api", "local", "gemini"], default="api")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    args = ap.parse_args()

    if args.embedder == "local":
        embedder = LocalEmbedder(args.model or "BAAI/bge-m3")
    elif args.embedder == "gemini":
        embedder = GeminiEmbedder(args.model or "gemini-embedding-001")
    else:
        embedder = ApiEmbedder(args.model or "openai/text-embedding-3-small",
                               base_url=args.base_url)

    index = VectorIndex.load(Path(args.index))
    bm25 = BM25([tokenize(t) for t in index.texts])

    (qvec,) = embedder.embed([args.query])
    vector_ranking = [cid for cid, _, _ in index.search(qvec, k=20)]
    results = hybrid_search(vector_ranking, bm25, index.ids, args.query, k=args.top_k)
    by_id = dict(zip(index.ids, index.texts, strict=True))

    print(f"query: {args.query}")
    for rank, (cid, score) in enumerate(results, 1):
        print(f"#{rank} [rrf={score:.4f}] {cid}")
        print(by_id[cid][:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
