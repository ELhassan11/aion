"""Follow-up agent + customer knowledge graph (P4-T05).

1. build_customer_graph: customers linked to products (tags), segments,
   tickets, and to each other through shared products.
2. draft_followup: personalized Arabic follow-up message per customer,
   using their profile + next best action + graph neighbours.
   Messages containing offers need human approval (P3-T05 gate).

Usage:
    python -m customer_intelligence.followup \\
        --profiles out/customer_360_enriched.json \\
        --actions out/churn_actions.json \\
        --output out/followups.json
    python -m customer_intelligence.followup --customer-id cust-005 \\
        --profiles out/customer_360_enriched.json \\
        --actions out/churn_actions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.llm import generate  # noqa: E402

_OFFER_WORDS = ["خصم", "عرض", "تعويض", "مجان"]


def build_customer_graph(profiles: list[dict]) -> dict:
    """Nodes: customer/product/segment. Edges: buys, in_segment, similar."""
    graph = {"nodes": [], "edges": []}
    seen = set()

    def add_node(nid: str, ntype: str, label: str):
        if nid not in seen:
            seen.add(nid)
            graph["nodes"].append(
                {"id": nid, "type": ntype, "label": label})

    for p in profiles:
        cid = p["id"]
        add_node(cid, "customer", p.get("name", cid))
        seg = p.get("segment", "")
        seg_id = f"seg:{seg}"
        add_node(seg_id, "segment", seg)
        graph["edges"].append(
            {"from": cid, "to": seg_id, "rel": "in_segment"})
        for tag in p.get("tags", []):
            pid = f"prod:{tag}"
            add_node(pid, "product", tag)
            graph["edges"].append(
                {"from": cid, "to": pid, "rel": "interested_in"})

    # similar_to: customers sharing a product
    prod_to_cust: dict[str, list[str]] = {}
    for e in graph["edges"]:
        if e["rel"] == "interested_in":
            prod_to_cust.setdefault(e["to"], []).append(e["from"])
    for custs in prod_to_cust.values():
        for i in range(len(custs)):
            for j in range(i + 1, len(custs)):
                graph["edges"].append(
                    {"from": custs[i], "to": custs[j], "rel": "similar_to"})
    return graph


def neighbours(graph: dict, customer_id: str) -> list[str]:
    """Names of similar customers (for social-proof in messages)."""
    ids = {e["to"] for e in graph["edges"]
           if e["from"] == customer_id and e["rel"] == "similar_to"}
    ids |= {e["from"] for e in graph["edges"]
            if e["to"] == customer_id and e["rel"] == "similar_to"}
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    return [labels[i] for i in ids if i in labels]


def draft_followup(profile: dict, action: dict,
                   similar: list[str]) -> str:
    prompt = (
        "اكتب رسالة متابعة قصيرة (سطرين كحد أقصى) بالعربية الفصحى المبسطة "
        "والودودة، موجهة للعميل بالاسم، لترسل عبر "
        f"{action.get('channel', 'واتساب')}. الهدف: {action.get('action')} "
        f"({action.get('reason', '')}). "
        + (f"يمكنك ذكر أن عملاء آخرين مهتمين بنفس المنتجات "
            f"({', '.join(similar[:2])}) استفادوا من العرض. " if similar else "")
        + "لا تذكر بيانات شخصية. لا تعد بمبلغ محدد.\n\n"
        f"العميل: {profile.get('name')} "
        f"(الشريحة: {profile.get('segment')})"
    )
    return generate(prompt).strip()


def needs_approval(message: str) -> bool:
    return any(w in message for w in _OFFER_WORDS)


def run(profiles: list[dict], actions: list[dict]) -> list[dict]:
    graph = build_customer_graph(profiles)
    by_id = {p["id"]: p for p in profiles}
    act_by_id = {a["customer_id"]: a["next_best_action"] for a in actions}
    followups = []
    for p in profiles:
        action = act_by_id.get(p["id"], {"action": "متابعة دورية",
                                         "channel": "واتساب",
                                         "reason": ""})
        msg = draft_followup(p, action, neighbours(graph, p["id"]))
        followups.append({
            "customer_id": p["id"],
            "name": p.get("name"),
            "action": action["action"],
            "channel": action.get("channel"),
            "message": msg,
            "needs_approval": needs_approval(msg),
        })
    return followups


def main() -> None:
    ap = argparse.ArgumentParser(description="Follow-up agent + KG (P4-T05)")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--customer-id", default=None)
    ap.add_argument("--graph-only", action="store_true",
                    help="only print graph stats, no LLM calls")
    args = ap.parse_args()

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    actions = json.loads(Path(args.actions).read_text(encoding="utf-8"))

    if args.customer_id:
        profiles = [p for p in profiles if p["id"] == args.customer_id]
        if not profiles:
            print(f"unknown customer: {args.customer_id}")
            return

    graph = build_customer_graph(profiles)
    print(f"graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    if args.graph_only:
        return

    followups = run(profiles, actions)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(followups, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"drafted {len(followups)} follow-up(s) -> {out}")

    for f in followups:
        print(f"\n— {f['name']} [{f['channel']}]"
              + (" ⚠️ يحتاج موافقة" if f["needs_approval"] else ""))
        print(f"  {f['message'][:200]}")


if __name__ == "__main__":
    main()
