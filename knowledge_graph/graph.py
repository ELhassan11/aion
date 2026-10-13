"""In-memory knowledge graph (P2-T05): triples, JSON persistence, queries.

Nodes are normalized entity names (see ingestion.normalize); edges are
(head, relation, tail) triples. Neo4j persistence is a later step —
this module owns the graph model and extraction flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.normalize import normalize_arabic  # noqa: E402


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    def add_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        source_chunk_id: str | None = None,
    ) -> None:
        head = normalize_arabic(head.strip())
        tail = normalize_arabic(tail.strip())
        relation = relation.strip()
        if not head or not tail or not relation:
            return
        self.nodes.setdefault(head, {"name": head})
        self.nodes.setdefault(tail, {"name": tail})
        edge = {"head": head, "relation": relation, "tail": tail}
        if source_chunk_id:
            edge["source_chunk_id"] = source_chunk_id
        if edge not in self.edges:
            self.edges.append(edge)

    def neighbors(
        self, entity: str, relation: str | None = None
    ) -> list[dict]:
        entity = normalize_arabic(entity.strip())
        out = []
        for e in self.edges:
            if e["head"] == entity and (relation is None or e["relation"] == relation):
                out.append(e)
            elif e["tail"] == entity and (relation is None or e["relation"] == relation):
                out.append(e)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"nodes": list(self.nodes.values()), "edges": self.edges},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        g = cls()
        for n in data.get("nodes", []):
            g.nodes[n["name"]] = n
        g.edges = data.get("edges", [])
        return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the knowledge graph (P2-T05)")
    ap.add_argument("--graph", required=True, help="graph JSON path")
    ap.add_argument("--entity", required=True, help="entity to look up")
    ap.add_argument("--relation", default=None, help="filter by relation")
    args = ap.parse_args()

    graph = KnowledgeGraph.load(Path(args.graph))
    hits = graph.neighbors(args.entity, relation=args.relation)
    print(f"entity: {args.entity} | nodes: {len(graph.nodes)} | edges: {len(graph.edges)}")
    for e in hits:
        print(f"  {e['head']} --[{e['relation']}]--> {e['tail']}")
    if not hits:
        print("  (no relations found)")


if __name__ == "__main__":
    main()
