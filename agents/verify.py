"""Verification layer (P3-T04): check agent answers before delivery.

Checks:
1. Citations — every [id] points to a retrieved chunk (no hallucinations).
2. Claims — LLM judge: does each cited chunk actually support the claim?
3. PII — flag emails / phone numbers leaking into answers.

Usage:
    python -m agents.verify --demo
    python -m agents.verify --answer-file ans.txt --evidence-file ev.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.citations import extract_citations, verify_citations  # noqa: E402
from rag.llm import generate  # noqa: E402

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+20|0020)?\s?01[0-9]{9}")

CITATION_RE = re.compile(r"\[(?:chunk-id:\s*)?([a-z0-9][a-z0-9_\-:.]*)\]")


def check_pii(text: str) -> list[dict]:
    """Find emails and phone numbers in text."""
    findings = []
    for m in EMAIL_RE.finditer(text):
        findings.append({"type": "email", "value": m.group(0)})
    for m in PHONE_RE.finditer(text):
        findings.append({"type": "phone", "value": m.group(0)})
    return findings


def redact_pii(text: str) -> str:
    """Replace emails/phones with placeholders."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def split_claims(answer: str) -> list[tuple[str, str]]:
    """Split answer into (sentence, cited_id) pairs for cited sentences."""
    claims = []
    for sent in re.split(r"(?<=[.!?؟])\s+", answer):
        m = CITATION_RE.search(sent)
        if m:
            claims.append((sent.strip(), m.group(1)))
    return claims


def check_claims(
    claims: list[tuple[str, str]], evidence: dict[str, str]
) -> list[dict]:
    """LLM judge: does the cited chunk support each claim?"""
    if not claims:
        return []
    numbered = "\n\n".join(
        f"[{i + 1}] Claim: {sent}\n    Cited chunk [{cid}]: "
        f"{evidence.get(cid, '(chunk not in evidence)')[:600]}"
        for i, (sent, cid) in enumerate(claims)
    )
    prompt = (
        "For each numbered claim, judge whether the cited chunk supports it. "
        'Reply with one line per claim only, format "<number>: SUPPORTED" or '
        '"<number>: UNSUPPORTED", no extra text.\n\n'
        f"{numbered}"
    )
    out = generate(prompt)
    results: dict[int, str] = {}
    for line in out.splitlines():
        m = re.match(r"\[?(\d+)\]?\s*[:\-]\s*(SUPPORTED|UNSUPPORTED)",
                     line.strip(), re.IGNORECASE)
        if m:
            results[int(m.group(1)) - 1] = m.group(2).upper()
    return [
        {"claim": sent, "chunk_id": cid,
         "verdict": results.get(i, "UNKNOWN")}
        for i, (sent, cid) in enumerate(claims)
    ]


def verify(
    answer: str, evidence: dict[str, str], retrieved_ids: list[str] | None = None
) -> dict:
    """Run all checks; return report with passed flag."""
    retrieved_ids = retrieved_ids or list(evidence.keys())
    cited = extract_citations(answer)
    cit = verify_citations(cited, retrieved_ids)
    claims = split_claims(answer)
    claim_checks = check_claims(claims, evidence)
    pii = check_pii(answer)

    issues = []
    if cit["invalid"]:
        issues.append(f"invalid citations: {cit['invalid']}")
    unsupported = [c for c in claim_checks if c["verdict"] == "UNSUPPORTED"]
    if unsupported:
        issues.append(f"{len(unsupported)} unsupported claim(s)")
    if pii:
        issues.append(f"PII leaked: {[p['type'] for p in pii]}")

    return {
        "passed": not issues,
        "issues": issues,
        "citations": cit,
        "claims": claim_checks,
        "pii": pii,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verification layer (P3-T04)")
    ap.add_argument("--demo", action="store_true",
                    help="run on a sample answer with planted issues")
    ap.add_argument("--answer-file", default=None)
    ap.add_argument("--evidence-file", default=None,
                    help="JSON {chunk_id: text}")
    args = ap.parse_args()

    if args.demo:
        evidence = {
            "doc-1": "سعر سماعة بلوتوث X200 هو 1,499 جنيه.",
        }
        answer = (
            "سعر السماعة 1,499 جنيه [doc-1]. "
            "وضمانها عشر سنوات [doc-1]. "
            "للتواصل: ahmed.m@example.com أو 01012345678 [doc-999]."
        )
        report = verify(answer, evidence)
    else:
        if not args.answer_file or not args.evidence_file:
            ap.error("--answer-file and --evidence-file required (or --demo)")
        answer = Path(args.answer_file).read_text(encoding="utf-8")
        evidence = json.loads(Path(args.evidence_file).read_text(encoding="utf-8"))
        report = verify(answer, evidence)

    print(f"verdict: {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    for issue in report["issues"]:
        print(f"  - {issue}")
    cit = report["citations"]
    print(f"citations: {len(cit['valid'])}/{len(cit['cited'])} valid")
    for c in report["claims"]:
        mark = "✅" if c["verdict"] == "SUPPORTED" else "❌"
        print(f"{mark} [{c['verdict']}] {c['claim'][:80]}")
    if report["pii"]:
        print("PII findings:")
        for p in report["pii"]:
            print(f"  - {p['type']}: {p['value']}")


if __name__ == "__main__":
    main()
