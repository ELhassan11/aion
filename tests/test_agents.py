"""Agent tests (P3-T07): deterministic unit tests, no network calls.

LLM and embedding calls are stubbed; data tools run against the sample
data files. Run with:

    python -m unittest discover -s tests -v
"""

import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stub_rag_modules():
    """Replace heavy rag deps (pydantic chain) with light stubs."""
    for name in ["rag.citations", "rag.hybrid", "rag.index", "rag.llm"]:
        sys.modules[name] = types.ModuleType(name)
    cit = sys.modules["rag.citations"]
    cit_re = re.compile(r"\[(?:chunk-id:\s*)?([a-z0-9][a-z0-9_\-:.]*)\]")
    cit.extract_citations = lambda a: list(dict.fromkeys(cit_re.findall(a)))

    def _vc(cited, rids):
        s = set(rids)
        return {
            "cited": cited,
            "valid": [c for c in cited if c in s],
            "invalid": [c for c in cited if c not in s],
        }

    cit.verify_citations = _vc
    hyb = sys.modules["rag.hybrid"]

    class FakeBM25:
        def __init__(self, docs):
            self.docs = docs

        def rank(self, query, ids, k):
            return ids[:k]

    hyb.BM25 = FakeBM25
    hyb.rrf_fuse = lambda rankings, **k: [(c, 1.0) for c in rankings[0]]
    hyb.tokenize = lambda t: t.split()
    sys.modules["rag.index"].VectorIndex = object
    sys.modules["rag.llm"].generate = lambda prompt, **kw: ""


_stub_rag_modules()

from agents import approvals, data_tools, verify  # noqa: E402


class TestDataTools(unittest.TestCase):
    def test_lookup_known_customer(self):
        out = json.loads(data_tools.call_tool(
            "customer_lookup", customer_id="cust-001"))
        self.assertEqual(out["id"], "cust-001")
        self.assertIn("tickets", out)

    def test_lookup_unknown_customer(self):
        out = json.loads(data_tools.call_tool(
            "customer_lookup", customer_id="nope"))
        self.assertIn("error", out)

    def test_search_by_segment(self):
        out = json.loads(data_tools.call_tool("customer_search", query="نشط"))
        self.assertTrue(all(c["segment"] == "نشط" for c in out))
        self.assertGreater(len(out), 0)

    def test_stats_shape(self):
        out = json.loads(data_tools.call_tool("customer_stats"))
        for key in ("customers", "avg_lifetime_value", "tickets",
                    "open_tickets"):
            self.assertIn(key, out)

    def test_unknown_tool(self):
        out = json.loads(data_tools.call_tool("nope"))
        self.assertIn("error", out)

    def test_bad_params(self):
        out = json.loads(data_tools.call_tool(
            "customer_lookup", wrong="x"))
        self.assertIn("error", out)


class TestApprovals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gate = approvals.ApprovalGate(
            Path(self.tmp.name) / "a.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_cycle(self):
        req = self.gate.request("do X", "details", risk="high")
        self.assertEqual(req.status, "pending")
        self.assertEqual(len(self.gate.pending()), 1)
        decided = self.gate.decide(req.id, True, by="tester")
        self.assertEqual(decided.status, "approved")
        self.assertEqual(self.gate.pending(), [])

    def test_reject_with_reason(self):
        req = self.gate.request("do Y", risk="high")
        decided = self.gate.decide(req.id, False, reason="too risky")
        self.assertEqual(decided.status, "rejected")
        self.assertEqual(decided.reason, "too risky")

    def test_double_decide_blocked(self):
        req = self.gate.request("do Z", risk="high")
        self.gate.decide(req.id, True)
        self.assertIsNone(self.gate.decide(req.id, False))

    def test_persistence(self):
        req = self.gate.request("do W", risk="high")
        gate2 = approvals.ApprovalGate(Path(self.tmp.name) / "a.json")
        self.assertEqual(len(gate2.pending()), 1)
        self.assertEqual(gate2.pending()[0].id, req.id)

    def test_policy(self):
        self.assertTrue(approvals.needs_approval("high"))
        self.assertFalse(approvals.needs_approval("low"))
        self.assertTrue(approvals.needs_approval("medium"))
        self.assertFalse(approvals.needs_approval("medium", auto_medium=True))


class TestVerify(unittest.TestCase):
    def test_pii_detection(self):
        findings = verify.check_pii("كلم a@b.com أو 01012345678")
        types = sorted(f["type"] for f in findings)
        self.assertEqual(types, ["email", "phone"])

    def test_pii_clean(self):
        self.assertEqual(verify.check_pii("نص عادي بلا بيانات"), [])

    def test_redact(self):
        out = verify.redact_pii("a@b.com و +201012345678")
        self.assertNotIn("a@b.com", out)
        self.assertIn("[EMAIL]", out)
        self.assertIn("[PHONE]", out)

    def test_split_claims(self):
        claims = verify.split_claims(
            "السعر 100 [doc-1]. بلا استشهاد. الضمان سنة [doc-2].")
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0][1], "doc-1")

    def test_verify_catches_issues(self):
        with mock.patch.object(
                verify, "check_claims",
                return_value=[{"claim": "x", "chunk_id": "doc-1",
                               "verdict": "UNSUPPORTED"}]):
            rep = verify.verify("نص [doc-999] و a@b.com",
                                {"doc-1": "نص"})
        self.assertFalse(rep["passed"])
        self.assertTrue(any("invalid citations" in i
                            for i in rep["issues"]))
        self.assertTrue(any("PII" in i for i in rep["issues"]))

    def test_verify_passes_clean(self):
        with mock.patch.object(
                verify, "check_claims",
                return_value=[{"claim": "x", "chunk_id": "doc-1",
                               "verdict": "SUPPORTED"}]):
            rep = verify.verify("السعر 100 [doc-1]", {"doc-1": "السعر 100"})
        self.assertTrue(rep["passed"])


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeIndex:
    ids = ["doc-1", "doc-2"]
    texts = ["نص أول عن السعر", "نص ثان عن الضمان"]

    def search(self, qvec, k=10):
        return [("doc-1", 0.9, self.texts[0]),
                ("doc-2", 0.5, self.texts[1])][:k]


PLAN_JSON = json.dumps([
    {"kind": "search", "description": "بحث", "query": "سؤال"},
    {"kind": "tool", "description": "أداة",
     "tool": "customer_stats", "params": {}},
    {"kind": "answer", "description": "إجابة"},
], ensure_ascii=False)


class TestPlanner(unittest.TestCase):
    def test_plan_parsing(self):
        from agents import planner
        with mock.patch.object(planner, "generate",
                               return_value=PLAN_JSON):
            agent = planner.PlannerAgent(index=FakeIndex(),
                                         bm25=None,
                                         embedder=FakeEmbedder())
            plan = agent.make_plan("هدف")
        self.assertEqual([s.kind for s in plan],
                         ["search", "tool", "answer"])
        self.assertEqual(plan[1].tool, "customer_stats")

    def test_plan_appends_answer(self):
        from agents import planner
        only_search = json.dumps(
            [{"kind": "search", "description": "ب", "query": "س"}])
        with mock.patch.object(planner, "generate",
                               return_value=only_search):
            agent = planner.PlannerAgent(index=FakeIndex(),
                                         bm25=None,
                                         embedder=FakeEmbedder())
            plan = agent.make_plan("هدف")
        self.assertEqual(plan[-1].kind, "answer")

    def test_run_executes_tool_step(self):
        from agents import planner
        from agents.planner import BM25

        calls = {"n": 0}

        def fake_generate(prompt, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return PLAN_JSON
            return "الإجابة [doc-1]"

        with mock.patch.object(planner, "generate", fake_generate):
            agent = planner.PlannerAgent(
                index=FakeIndex(), bm25=BM25([]),
                embedder=FakeEmbedder(), top_k=2)
            out = agent.run("هدف", verbose=False)
        kinds = [s["kind"] for s in out["plan"]]
        self.assertEqual(kinds, ["search", "tool", "answer"])
        self.assertIn("الإجابة", out["answer"])


if __name__ == "__main__":
    unittest.main()
