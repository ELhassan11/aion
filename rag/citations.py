"""Citation system (P2-T07): answers with traceable [chunk-id] citations.

Pipeline: hybrid retrieval -> rerank -> LLM answer with inline citations
-> verification that every citation points to a retrieved chunk.

Usage:
    python -m rag.citations --index out/index.npz --query "..." \\
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
from rag.rerank import rerank  # noqa: E402

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9_-]*)\]")


def generate_answer(query: str, passages: list[tuple[str, str]]) -> str:
    """Ask the LLM to answer from the passages, citing [chunk-id] per claim."""
    context = "\n\n".join(f"[{cid}]\n{text[:800]}" for cid, text in passages)
    prompt = (
        "Answer the question in Arabic using ONLY the sources below. "
        "After every factual claim, cite its source as [chunk-id] exactly as shown. "
        "If the sources do not contain the answer, say so plainly.\n\n"
        f"Question: {query}\n\nSources:\n{context}"
    )
    return generate(prompt)


def extract_citations(answer: str) -> list[str]:
    """Return chunk ids cited as [id] in the answer, in order of appearance."""
    seen: list[str] = []
    for cid in CITATION_RE.findall(answer):
        if cid not in seen:
            seen.append(cid)
    return seen


def verify_citations(
    cited: list[str], retrieved_ids: list[str]
) -> dict:
    """Check cited ids against retrieved ids; report valid/invalid/coverage."""
    retrieved = set(retrieved_ids)
    valid = [c for c in cited if c in retrieved]
    invalid = [c for c in cited if c not in retrieved]
    return {
        "cited": cited,
        "valid": valid,
        "invalid": invalid,
        "coverage": len(valid) / max(len(retrieved_ids), 1),
    }


def main() -> None:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder

    ap = argparse.ArgumentParser(description="Cited RAG answers (P2-T07)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=3)
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

    (qvec,) = embedder.embed([args.query])
    vector_ranking = [cid for cid, _, _ in index.search(qvec, k=10)]
    keyword_ranking = bm25.rank(args.query, index.ids, 10)
    fused = rrf_fuse([vector_ranking, keyword_ranking])[:10]
    candidates = [(cid, by_id[cid]) for cid, _ in fused if cid in by_id]
    ranked = rerank(args.query, candidates, top_k=args.top_k)
    retrieved_ids = [cid for cid, _ in ranked]
    passages = [(cid, by_id[cid]) for cid in retrieved_ids]

    answer = generate_answer(args.query, passages)
    cited = extract_citations(answer)
    report = verify_citations(cited, retrieved_ids)

    print(f"query: {args.query}\n")
    print("answer:")
    print(answer)
    print("\ncitations:")
    print(f"  cited ids: {report['cited'] or '(none)'}")
    print(f"  valid: {len(report['valid'])}/{len(report['cited'])}")
    if report["invalid"]:
        print(f"  INVALID (not retrieved): {report['invalid']}")
    print(f"  coverage: {report['coverage']:.0%} of retrieved chunks cited")


if __name__ == "__main__":
    main()
