"""Query expansion + reranking (P2-T04).

Full retrieval pipeline:
1. Expand the query into LLM-generated variations.
2. Hybrid-search (BM25 + vector, RRF) each variation -> candidate pool.
3. LLM-rerank the candidates -> final top-k.

Usage:
    python -m rag.rerank --index out/index.npz --query "سياسة الاستبدال" \\
        --top-k 3 --embedder gemini
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.hybrid import BM25, rrf_fuse, tokenize  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.llm import generate  # noqa: E402


def expand_query(query: str, n: int = 2) -> list[str]:
    """Generate n alternative phrasings of the query via the LLM."""
    prompt = (
        "You are helping an Arabic search engine. Generate "
        f"{n} alternative phrasings of this search query, one per line, "
        "no numbering, no extra text. Keep them in Arabic.\n"
        f"Query: {query}"
    )
    out = generate(prompt)
    variations = [
        re.sub(r"^[\d\-.)\s]+", "", line).strip()
        for line in out.splitlines()
        if line.strip()
    ]
    return [query] + variations[:n]


def rerank(
    query: str, passages: list[tuple[str, str]], top_k: int = 5
) -> list[tuple[str, float]]:
    """Score passages 0-10 for relevance via the LLM; return top_k sorted."""
    numbered = "\n\n".join(
        f"[{i + 1}] {text[:600]}" for i, (_, text) in enumerate(passages)
    )
    prompt = (
        "Score each passage for relevance to the query on a 0-10 scale "
        "(10 = directly answers it). Reply with one line per passage only, "
        'in the exact format "<number>: <score>", no extra text.\n'
        f"Query: {query}\n\nPassages:\n{numbered}"
    )
    out = generate(prompt)
    scores: dict[int, float] = {}
    for line in out.splitlines():
        m = re.match(r"\[?(\d+)\]?\s*[:\-]\s*(\d+(?:\.\d+)?)", line.strip())
        if m:
            idx, score = int(m.group(1)) - 1, float(m.group(2))
            if 0 <= idx < len(passages):
                scores[idx] = score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(passages[i][0], s) for i, s in ranked]


def main() -> None:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder

    ap = argparse.ArgumentParser(description="Query expansion + reranking (P2-T04)")
    ap.add_argument("--index", required=True, help="index .npz path")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--variations", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--embedder", choices=["api", "local", "gemini"], default="api")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    args = ap.parse_args()

    if args.embedder == "local":
        embedder = LocalEmbedder(args.model or "BAAI/bge-m3")
    elif args.embedder == "gemini":
        embedder = GeminiEmbedder(args.model or "gemini-embedding-001")
    else:
        embedder = ApiEmbedder(
            args.model or "openai/text-embedding-3-small", base_url=args.base_url
        )

    index = VectorIndex.load(Path(args.index))
    bm25 = BM25([tokenize(t) for t in index.texts])
    by_id = dict(zip(index.ids, index.texts, strict=True))

    queries = [args.query] if args.no_expand else expand_query(args.query, n=args.variations)
    print(f"searching {len(queries)} query variation(s)")

    rankings: list[list[str]] = []
    for q in queries:
        (qvec,) = embedder.embed([q])
        vector_ranking = [cid for cid, _, _ in index.search(qvec, k=args.candidates)]
        keyword_ranking = bm25.rank(q, index.ids, args.candidates)
        rankings.append(vector_ranking)
        rankings.append(keyword_ranking)
    fused = rrf_fuse(rankings)[: args.candidates]
    candidates = [(cid, by_id[cid]) for cid, _ in fused]

    if args.no_rerank:
        results = [(cid, score) for cid, score in fused[: args.top_k]]
    else:
        results = rerank(args.query, candidates, top_k=args.top_k)

    print(f"query: {args.query}\n")
    for rank, (cid, score) in enumerate(results, 1):
        print(f"#{rank} [score={score:.2f}] {cid}")
        print(by_id[cid][:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
