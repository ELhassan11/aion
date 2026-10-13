"""P2 checkpoint demo: live retrieval showcase on Arabic documents.

Runs the full P2 pipeline end to end:
  1. Golden-set evaluation (score)
  2. Hybrid search (BM25 + vector, RRF)
  3. Multi-hop retrieval (entity linking + graph expansion)
  4. Cited answer generation + verification

Usage:
    python scripts/p2_checkpoint_demo.py --index out/index.npz \\
        --graph out/graph.json --golden eval/retrieval/golden.jsonl \\
        --embedder gemini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.retrieval.evaluate import evaluate  # noqa: E402
from ingestion.normalize import normalize_arabic  # noqa: E402
from knowledge_graph.graph import KnowledgeGraph  # noqa: E402
from rag.citations import (  # noqa: E402
    extract_citations,
    generate_answer,
    verify_citations,
)
from rag.hybrid import BM25, rrf_fuse, tokenize  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.multihop import multihop_search  # noqa: E402
from rag.rerank import rerank  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder

    ap = argparse.ArgumentParser(description="P2 checkpoint demo")
    ap.add_argument("--index", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--golden", required=True)
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

    print("loading index + graph ...")
    index = VectorIndex.load(Path(args.index))
    graph = KnowledgeGraph.load(Path(args.graph))
    bm25 = BM25([tokenize(t) for t in index.texts])
    by_id = dict(zip(index.ids, index.texts, strict=True))
    print(f"index: {len(index.ids)} chunks | graph: {len(graph.nodes)} nodes, "
          f"{len(graph.edges)} edges")

    def retrieve(query: str, k: int) -> list[str]:
        (qvec,) = embedder.embed([query])
        vector_ranking = [cid for cid, _, _ in index.search(qvec, k=10)]
        keyword_ranking = bm25.rank(query, index.ids, 10)
        ranked = [cid for cid, _ in rrf_fuse([vector_ranking, keyword_ranking])]
        return [by_id[c] for c in ranked[:k] if c in by_id]

    # 1. Golden evaluation
    section("1) التقييم الذهبي — 10 أسئلة")
    questions = [
        json.loads(line)
        for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate(questions, retrieve, top_k=3)
    for p in report["per_question"]:
        mark = "✅" if p["hit"] else "❌"
        print(f"{mark} {p['question']}")
    print(f"\nالنتيجة: mean hit@3 = {report['mean_hit']:.2f} | "
          f"mean keyword recall@3 = {report['mean_keyword_recall']:.2f}")

    # 2. Hybrid search showcase
    section("2) بحث هجين — سؤال مباشر")
    q1 = "بكام الشاحن السريع 65 واط؟"
    print(f"السؤال: {q1}")
    for i, t in enumerate(retrieve(q1, 3), 1):
        print(f"#{i} {t[:120].replace(chr(10), ' ')}...")

    # 3. Multi-hop showcase
    section("3) استرجاع متعدد القفزات — سؤال مركب")
    q2 = "كم سعر سماعة بلوتوث X200 وما مدة ضمانها؟"
    out = multihop_search(q2, index, bm25, graph, embedder, top_k=3)
    print(f"السؤال: {q2}")
    print(f"الكيانات المربوطة: {out['entities']}")
    for e in out["hop_edges"][:4]:
        print(f"  {e['head']} --[{e['relation']}]--> {e['tail']}")
    for rank, (cid, score) in enumerate(out["results"], 1):
        print(f"#{rank} [score={score:.2f}] {cid}")

    # 4. Cited answer showcase
    section("4) إجابة موثقة — مع التحقق")
    q3 = "كم سعر سماعة بلوتوث X200؟"
    print(f"السؤال: {q3}")
    (qvec,) = embedder.embed([q3])
    vr = [c for c, _, _ in index.search(qvec, k=10)]
    kr = bm25.rank(q3, index.ids, 10)
    fused = rrf_fuse([vr, kr])[:10]
    cands = [(c, by_id[c]) for c, _ in fused if c in by_id]
    ranked = rerank(q3, cands, top_k=3)
    rids = [c for c, _ in ranked]
    answer = generate_answer(q3, [(c, by_id[c]) for c in rids])
    cited = extract_citations(answer)
    rep = verify_citations(cited, rids)
    print(f"الإجابة: {answer}")
    print(f"الاستشهادات: {rep['valid']}/{len(rep['cited'])} سليمة"
          + (f" | غير سليمة: {rep['invalid']}" if rep["invalid"] else ""))

    section("الحكم")
    ok = report["mean_hit"] >= 0.8 and not rep["invalid"]
    print("✅ الاسترجاع سليم — نكمل P3" if ok else "❌ الجودة ضعيفة — نوقف ونصلح")


if __name__ == "__main__":
    main()
