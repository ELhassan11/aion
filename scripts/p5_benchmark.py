"""Final benchmark runs (P5-T01): the numbers for the defense.

Compares retrieval configurations on the golden set (vector-only vs
hybrid vs hybrid+expansion+rerank) and runs the agent unit tests.
Writes a JSON report and prints a comparison table.

Usage:
    python scripts/p5_benchmark.py --index out/index.npz \\
        --golden eval/retrieval/golden.jsonl --embedder gemini \\
        --output out/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.retrieval.evaluate import evaluate  # noqa: E402
from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder  # noqa: E402
from rag.hybrid import BM25, rrf_fuse, tokenize  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.rerank import expand_query, rerank  # noqa: E402


def make_embedder(name: str, model: str | None):
    if name == "local":
        return LocalEmbedder(model or "BAAI/bge-m3")
    if name == "gemini":
        return GeminiEmbedder(model or "gemini-embedding-001")
    return ApiEmbedder(model or "openai/text-embedding-3-small")


def main() -> None:
    ap = argparse.ArgumentParser(description="Final benchmarks (P5-T01)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--embedder", choices=["api", "local", "gemini"],
                    default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="out/benchmark.json")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip the slow expansion+rerank config")
    args = ap.parse_args()

    embedder = make_embedder(args.embedder, args.model)
    index = VectorIndex.load(Path(args.index))
    bm25 = BM25([tokenize(t) for t in index.texts])
    by_id = dict(zip(index.ids, index.texts, strict=True))
    questions = [
        json.loads(line)
        for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def hybrid_rank(query: str) -> list[str]:
        (qvec,) = embedder.embed([query])
        vr = [c for c, _, _ in index.search(qvec, k=10)]
        kr = bm25.rank(query, index.ids, 10)
        return [c for c, _ in rrf_fuse([vr, kr])]

    def retrieve_vector(query: str, k: int) -> list[str]:
        (qvec,) = embedder.embed([query])
        cids = [c for c, _, _ in index.search(qvec, k=10)][:k]
        return [by_id[c] for c in cids if c in by_id]

    def retrieve_hybrid(query: str, k: int) -> list[str]:
        return [by_id[c] for c in hybrid_rank(query)[:k] if c in by_id]

    def retrieve_full(query: str, k: int) -> list[str]:
        pool: dict[str, str] = {}
        for var in expand_query(query):
            for cid in hybrid_rank(var)[:10]:
                if cid in by_id:
                    pool[cid] = by_id[cid]
        ranked = rerank(query, list(pool.items()), top_k=k)
        return [by_id[c] for c, _ in ranked if c in by_id]

    configs = {
        "vector-only": retrieve_vector,
        "hybrid": retrieve_hybrid,
    }
    if not args.skip_full:
        configs["hybrid+expand+rerank"] = retrieve_full

    results: dict = {
        "date": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "n_questions": len(questions),
        "configs": {},
    }
    for name, fn in configs.items():
        t0 = time.time()
        report = evaluate(questions, fn, args.top_k)
        by_diff: dict[str, list[float]] = {}
        for item, p in zip(questions, report["per_question"]):
            by_diff.setdefault(item.get("difficulty", "easy"), []).append(p["hit"])
        results["configs"][name] = {
            "mean_hit": round(report["mean_hit"], 3),
            "mean_keyword_recall": round(report["mean_keyword_recall"], 3),
            "seconds": round(time.time() - t0, 1),
            "by_difficulty": {d: round(sum(h) / len(h), 3)
                              for d, h in by_diff.items()},
            "per_question": [
                {"q": p["question"][:60], "hit": p["hit"]} for p in report["per_question"]
            ],
        }

    # agent unit tests
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = proc.stderr.strip().splitlines()[-3:]
    results["agent_tests"] = {
        "passed": proc.returncode == 0,
        "seconds": round(time.time() - t0, 1),
        "summary": " | ".join(tail),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"\n{'config':<24} {'hit@3':<8} {'recall@3':<9} {'time'}")
    print("-" * 55)
    for name, r in results["configs"].items():
        print(f"{name:<24} {r['mean_hit']:<8.3f} "
              f"{r['mean_keyword_recall']:<9.3f} {r['seconds']}s")
        for d, h in sorted(r["by_difficulty"].items()):
            print(f"  └─ {d:<20} hit={h:.3f}")
    at = results["agent_tests"]
    print(f"\nagent tests: {'✅ PASS' if at['passed'] else '❌ FAIL'} "
          f"({at['seconds']}s)")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
