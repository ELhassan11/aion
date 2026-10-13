"""Retrieval evaluation (P2-T08): golden Q&A set with keyword-based metrics.

Each golden item has a question and must_contain keywords. For each
question we run retrieval and check (on normalized text) whether the
keywords appear in the top-k chunks:

- hit@k: 1 if ALL keywords found, else 0
- keyword_recall@k: fraction of keywords found

Usage:
    python -m eval.retrieval.evaluate --index out/index.npz \\
        --golden eval/retrieval/golden.jsonl --top-k 3 --embedder gemini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.normalize import normalize_arabic  # noqa: E402
from rag.hybrid import BM25, rrf_fuse, tokenize  # noqa: E402
from rag.index import VectorIndex  # noqa: E402


def evaluate(
    questions: list[dict],
    retrieve,
    top_k: int,
) -> dict:
    per_question: list[dict] = []
    for item in questions:
        q = item["question"]
        keywords = [normalize_arabic(k) for k in item["must_contain"]]
        top_texts = [normalize_arabic(t) for t in retrieve(q, top_k)]
        joined = "\n".join(top_texts)
        found = [k for k in keywords if k in joined]
        hit = 1.0 if len(found) == len(keywords) else 0.0
        per_question.append(
            {
                "question": q,
                "hit": hit,
                "keyword_recall": len(found) / max(len(keywords), 1),
                "missing": [k for k in keywords if k not in found],
            }
        )
    n = max(len(per_question), 1)
    return {
        "per_question": per_question,
        "mean_hit": sum(p["hit"] for p in per_question) / n,
        "mean_keyword_recall": sum(p["keyword_recall"] for p in per_question) / n,
        "n": len(per_question),
    }


def main() -> None:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder

    ap = argparse.ArgumentParser(description="Retrieval eval on golden set (P2-T08)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
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
    questions = [
        json.loads(line)
        for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def retrieve(query: str, k: int) -> list[str]:
        (qvec,) = embedder.embed([query])
        vector_ranking = [cid for cid, _, _ in index.search(qvec, k=10)]
        if args.mode == "vector":
            ranked = vector_ranking
        else:
            keyword_ranking = bm25.rank(query, index.ids, 10)
            ranked = [cid for cid, _ in rrf_fuse([vector_ranking, keyword_ranking])]
        return [by_id[cid] for cid in ranked[:k] if cid in by_id]

    report = evaluate(questions, retrieve, args.top_k)

    print(f"mode={args.mode} top_k={args.top_k} n={report['n']}")
    for p in report["per_question"]:
        mark = "✅" if p["hit"] else "❌"
        print(f"{mark} hit={p['hit']:.0f} recall={p['keyword_recall']:.2f} | {p['question']}")
        if p["missing"]:
            print(f"   missing keywords: {p['missing']}")
    print(f"\nmean hit@{args.top_k}: {report['mean_hit']:.2f}")
    print(f"mean keyword recall@{args.top_k}: {report['mean_keyword_recall']:.2f}")


if __name__ == "__main__":
    main()
