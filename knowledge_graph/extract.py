"""Knowledge graph extraction (P2-T05): LLM -> (head, relation, tail) triples.

Usage:
    python -m knowledge_graph.extract --input out/chunks_split.jsonl \\
        --output out/graph.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import DocumentChunk  # noqa: E402
from knowledge_graph.graph import KnowledgeGraph  # noqa: E402
from rag.llm import generate  # noqa: E402


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract factual triples from text via the LLM."""
    prompt = (
        "Extract factual triples from the Arabic text below. "
        "Reply with one triple per line in the exact format:\n"
        "<head entity> | <relation> | <tail entity>\n"
        "No numbering, no extra text. Keep entities and relations in Arabic.\n\n"
        f"Text:\n{text[:2000]}"
    )
    out = generate(prompt)
    triples: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def main() -> None:
    ap = argparse.ArgumentParser(description="Build knowledge graph from chunks (P2-T05)")
    ap.add_argument("--input", required=True, help="input DocumentChunk JSONL")
    ap.add_argument("--output", required=True, help="output graph JSON path")
    args = ap.parse_args()

    graph = KnowledgeGraph()
    n_chunks, n_triples = 0, 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = DocumentChunk.model_validate_json(line)
            n_chunks += 1
            for head, relation, tail in extract_triples(chunk.text):
                graph.add_triple(head, relation, tail, source_chunk_id=chunk.id)
                n_triples += 1

    graph.save(Path(args.output))
    print(
        f"extracted {n_triples} triples from {n_chunks} chunks | "
        f"nodes: {len(graph.nodes)} edges: {len(graph.edges)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
