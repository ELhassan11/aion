"""Multi-hop retrieval (P2-T06): query -> entities -> graph neighbors -> chunks.

Some questions need facts from more than one chunk. This module:
1. Links query mentions to knowledge-graph entities (normalized substring match).
2. Expands one hop over the graph: neighbors of linked entities.
3. Maps expanded entities back to their source chunks.
4. Merges with hybrid-search candidates and LLM-reranks everything.

Usage:
    python -m rag.multihop --index out/index.npz --graph out/graph.json \\
        --query "كم سعر سماعة بلوتوث X200 وما مدة ضمانها؟" \\
        --top-k 3 --embedder gemini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.normalize import normalize_arabic  # noqa: E402
from knowledge_graph.graph import KnowledgeGraph  # noqa: E402
from rag.hybrid import BM25, rrf_fuse, tokenize  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.rerank import rerank  # noqa: E402


def link_entities(query: str, graph: KnowledgeGraph) -> list[str]:
    """Find graph entities mentioned in the query (normalized substring)."""
    q = normalize_arabic(query)
    return [name for name in graph.nodes if name and name in q]


def expand_one_hop(
    entities: list[str], graph: KnowledgeGraph
) -> tuple[list[dict], list[str]]:
    """Return (edges, chunk_ids) reachable in one hop from the entities."""
    edges: list[dict] = []
    chunk_ids: list[str] = []
    for ent in entities:
        for e in graph.neighbors(ent):
            edges.append(e)
            cid = e.get("source_chunk_id")
            if cid:
                chunk_ids.append(cid)
    return edges, chunk_ids


def multihop_search(
    query: str,
    index: VectorIndex,
    bm25: BM25,
    graph: KnowledgeGraph,
    embedder,
    top_k: int = 5,
    candidate_k: int = 10,
) -> dict:
    by_id = dict(zip(index.ids, index.texts, strict=True))

    # Hop 1: hybrid retrieval
    (qvec,) = embedder.embed([query])
    vector_ranking = [cid for cid, _, _ in index.search(qvec, k=candidate_k)]
    keyword_ranking = bm25.rank(query, index.ids, candidate_k)
    fused = rrf_fuse([vector_ranking, keyword_ranking])

    # Hop 2: entity linking + graph expansion
    entities = link_entities(query, graph)
    hop_edges, hop_chunk_ids = expand_one_hop(entities, graph)

    # Merge: fused ids first, then graph-reached chunk ids
    seen = {cid for cid, _ in fused}
    merged_ids = [cid for cid, _ in fused]
    for cid in hop_chunk_ids:
        if cid not in seen and cid in by_id:
            seen.add(cid)
            merged_ids.append(cid)

    candidates = [(cid, by_id[cid]) for cid in merged_ids[: candidate_k * 2]]
    results = rerank(query, candidates, top_k=top_k)

    return {
        "entities": entities,
        "hop_edges": hop_edges,
        "results": results,
        "texts": by_id,
    }


def main() -> None:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder

    ap = argparse.ArgumentParser(description="Multi-hop retrieval (P2-T06)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--graph", required=True)
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
    graph = KnowledgeGraph.load(Path(args.graph))
    bm25 = BM25([tokenize(t) for t in index.texts])

    out = multihop_search(
        args.query, index, bm25, graph, embedder, top_k=args.top_k
    )

    print(f"query: {args.query}")
    print(f"linked entities: {out['entities'] or '(none)'}")
    if out["hop_edges"]:
        print("graph hop:")
        for e in out["hop_edges"][:6]:
            print(f"  {e['head']} --[{e['relation']}]--> {e['tail']}")
    print()
    for rank, (cid, score) in enumerate(out["results"], 1):
        print(f"#{rank} [score={score:.2f}] {cid}")
        print(out["texts"][cid][:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
