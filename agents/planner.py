"""Planning & research agent (P3-T01).

Takes a goal, asks the LLM to break it into search/answer steps,
executes searches with the P2 retrieval engine, then synthesizes
a cited final answer.

Usage:
    python -m agents.planner --index out/index.npz \\
        --goal "قارن بين سماعة X200 والشاحن السريع من حيث السعر والضمان" \\
        --embedder gemini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.citations import extract_citations, verify_citations  # noqa: E402
from rag.hybrid import BM25, rrf_fuse, tokenize  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.llm import generate  # noqa: E402

try:
    from agents.data_tools import TOOLS, call_tool

    _TOOL_DOCS = "\n".join(
        f"- {t.name}{json.dumps(t.parameters, ensure_ascii=False)}: {t.description}"
        for t in TOOLS.values()
    )
except ImportError:  # data tools optional
    TOOLS, call_tool, _TOOL_DOCS = {}, None, "(no data tools available)"


@dataclass
class PlanStep:
    kind: str  # "search" | "tool" | "answer"
    description: str
    query: str = ""
    tool: str = ""
    params: dict = field(default_factory=dict)
    status: str = "pending"
    result: str = ""


@dataclass
class PlannerAgent:
    index: VectorIndex
    bm25: BM25
    embedder: object = None
    top_k: int = 5

    def make_plan(self, goal: str) -> list[PlanStep]:
        prompt = (
            "Break this user goal into a short research plan. Reply with a JSON "
            'array only, no markdown fences. Each step is one of:\n'
            '- {"kind": "search", "description": "...", "query": "..."} — search documents\n'
            '- {"kind": "tool", "description": "...", "tool": "<name>", '
            '"params": {...}} — call a data tool\n'
            '- {"kind": "answer", "description": "..."} — write the final answer\n'
            "Available data tools:\n"
            f"{_TOOL_DOCS}\n"
            "Use 2-5 steps; the last step must be kind=answer. "
            "Write descriptions and queries in Arabic.\n"
            f"Goal: {goal}"
        )
        out = generate(prompt)
        out = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
        steps = []
        for s in json.loads(out):
            steps.append(
                PlanStep(
                    kind=s["kind"],
                    description=s.get("description", ""),
                    query=s.get("query", ""),
                    tool=s.get("tool", ""),
                    params=s.get("params", {}) or {},
                )
            )
        if not steps or steps[-1].kind != "answer":
            steps.append(PlanStep(kind="answer", description="تجميع الإجابة النهائية"))
        return steps

    def search(self, query: str) -> list[tuple[str, str]]:
        by_id = dict(zip(self.index.ids, self.index.texts, strict=True))
        (qvec,) = self.embedder.embed([query])
        vector_ranking = [c for c, _, _ in self.index.search(qvec, k=10)]
        keyword_ranking = self.bm25.rank(query, self.index.ids, 10)
        fused = rrf_fuse([vector_ranking, keyword_ranking])[: self.top_k]
        return [(c, by_id[c]) for c, _ in fused if c in by_id]

    def run(self, goal: str, verbose: bool = True) -> dict:
        plan = self.make_plan(goal)
        evidence: list[tuple[str, str]] = []
        trace: list[dict] = []

        for i, step in enumerate(plan, 1):
            if verbose:
                print(f"\n[خطوة {i}/{len(plan)}] {step.description} ({step.kind})")
            if step.kind == "search":
                hits = self.search(step.query)
                evidence.extend(hits)
                step.result = f"{len(hits)} مقاطع"
                if verbose:
                    for cid, text in hits[:3]:
                        print(f"  - {cid}: {text[:80].replace(chr(10), ' ')}...")
            elif step.kind == "tool":
                if call_tool is None:
                    step.result = "أدوات البيانات غير متاحة"
                else:
                    out = call_tool(step.tool, **step.params)
                    evidence.append((f"tool:{step.tool}", out[:1500]))
                    step.result = out[:300]
                if verbose:
                    print(f"  أداة {step.tool} {step.params}")
                    print(f"  النتيجة: {step.result[:200].replace(chr(10), ' ')}...")
            else:
                context = "\n\n".join(
                    f"[{cid}]\n{text[:800]}" for cid, text in evidence
                )
                prompt = (
                    "أجب عن الهدف التالي بالعربية باستخدام الأدلة فقط. "
                    "بعد كل معلومة ضع معرف مصدرها بين قوسين مربعين بنفس الشكل "
                    "الظاهر تمامًا، مثال: [doc-9e0dea2b69e4-c0000]. "
                    "لا تكتب كلمة chunk-id أبدًا.\n\n"
                    f"الهدف: {goal}\n\nالأدلة:\n{context}"
                )
                step.result = generate(prompt)
                if verbose:
                    print(f"  الإجابة:\n  {step.result}")
            step.status = "done"
            trace.append(
                {"kind": step.kind, "description": step.description,
                 "query": step.query, "tool": step.tool,
                 "params": step.params, "result": step.result}
            )

        answer = next(
            (s.result for s in reversed(plan) if s.kind == "answer"), ""
        )
        cited = extract_citations(answer)
        retrieved_ids = [c for c, _ in evidence]
        report = verify_citations(cited, retrieved_ids)
        return {"plan": trace, "answer": answer, "citations": report}


def main() -> None:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder

    ap = argparse.ArgumentParser(description="Planning & research agent (P3-T01)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--goal", required=True)
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
        embedder = ApiEmbedder(
            args.model or "openai/text-embedding-3-small", base_url=args.base_url
        )

    index = VectorIndex.load(Path(args.index))
    bm25 = BM25([tokenize(t) for t in index.texts])
    agent = PlannerAgent(index=index, bm25=bm25, embedder=embedder, top_k=args.top_k)

    print(f"الهدف: {args.goal}")
    out = agent.run(args.goal)
    print("\n" + "=" * 60)
    print("الإجابة النهائية:")
    print(out["answer"])
    rep = out["citations"]
    print(f"\nالاستشهادات: {len(rep['valid'])}/{len(rep['cited'])} سليمة")


if __name__ == "__main__":
    main()
