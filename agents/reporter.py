"""Reporting agent (P3-T06): structured Arabic reports from data + retrieval.

Report types:
- customer: per-customer report (profile, tickets, policy context)
- digest: operations digest (stats, open tickets, churn risks)

Usage:
    python -m agents.reporter --type customer --customer-id cust-001 \\
        --index out/index.npz --embedder gemini --output out/report.md
    python -m agents.reporter --type digest --output out/digest.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.data_tools import call_tool  # noqa: E402
from rag.llm import generate  # noqa: E402


def _retrieve_context(query: str, index_path: str, embedder_name: str,
                      k: int = 3) -> str:
    """Optional retrieval context for the report (P2 engine)."""
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder
    from rag.hybrid import BM25, rrf_fuse, tokenize
    from rag.index import VectorIndex

    if embedder_name == "local":
        embedder = LocalEmbedder("BAAI/bge-m3")
    elif embedder_name == "gemini":
        embedder = GeminiEmbedder("gemini-embedding-001")
    else:
        embedder = ApiEmbedder("openai/text-embedding-3-small")

    index = VectorIndex.load(Path(index_path))
    bm25 = BM25([tokenize(t) for t in index.texts])
    by_id = dict(zip(index.ids, index.texts, strict=True))
    (qvec,) = embedder.embed([query])
    vr = [c for c, _, _ in index.search(qvec, k=10)]
    kr = bm25.rank(query, index.ids, 10)
    fused = rrf_fuse([vr, kr])[:k]
    return "\n\n".join(
        f"[{c}]\n{by_id[c][:600]}" for c, _ in fused if c in by_id
    )


def customer_report(customer_id: str, index_path: str | None,
                    embedder_name: str) -> str:
    profile = json.loads(call_tool("customer_lookup", customer_id=customer_id))
    if "error" in profile:
        return f"# تقرير عميل\n\nخطأ: {profile['error']}"
    context = ""
    if index_path:
        context = _retrieve_context(
            "سياسة الاستبدال والاسترجاع", index_path, embedder_name)
    prompt = (
        "اكتب تقريرًا عربيًا منظمًا (markdown) عن العميل التالي للإدارة. "
        "أقسام: نبذة، التذاكر، المخاطر والتوصيات. كن موجزًا ومهنيًا.\n\n"
        f"بيانات العميل:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n"
        + (f"\nسياق السياسات:\n{context}\n" if context else "")
    )
    body = generate(prompt)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"# تقرير العميل — {profile.get('name', customer_id)}\n\n_{date}_\n\n{body}"


def digest_report() -> str:
    stats = json.loads(call_tool("customer_stats"))
    high_risk = json.loads(call_tool("customer_search", query="خامل"))
    prompt = (
        "اكتب موجز عمليات عربيًا (markdown) من البيانات التالية: "
        "أقسام: نظرة عامة، عملاء يحتاجون متابعة، توصيات. موجز ومهني.\n\n"
        f"الإحصائيات:\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n\n"
        f"عملاء خاملون:\n{json.dumps(high_risk, ensure_ascii=False, indent=2)}\n"
    )
    body = generate(prompt)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"# الموجز اليومي — {date}\n\n{body}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporting agent (P3-T06)")
    ap.add_argument("--type", choices=["customer", "digest"], required=True)
    ap.add_argument("--customer-id", default=None)
    ap.add_argument("--index", default=None, help="index .npz for policy context")
    ap.add_argument("--embedder", choices=["api", "local", "gemini"],
                    default="gemini")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.type == "customer":
        if not args.customer_id:
            ap.error("--customer-id required for customer reports")
        report = customer_report(args.customer_id, args.index, args.embedder)
    else:
        report = digest_report()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"report written -> {out} ({len(report)} chars)")
    print("\n--- preview ---\n")
    print(report[:800])


if __name__ == "__main__":
    main()
