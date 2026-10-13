#!/usr/bin/env python3
"""Model comparison harness (P0-T06).

Compares candidate LLMs on the same Arabic test cases and scores them
by keyword match (simple, transparent, reproducible).

Usage:
    # OpenAI-compatible API:
    OPENAI_API_KEY=sk-... python eval/model_comparison/compare.py --models gpt-4o-mini
    # Local Ollama:
    python eval/model_comparison/compare.py --models ollama:qwen2.5:7b --ollama-url http://localhost:11434

Results are printed as a markdown table and saved to eval/model_comparison/results/.
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent / "data" / "sample" / "documents"
RESULTS = HERE / "results"


def _load_dotenv() -> None:
    """Read .env from the project root (no external dependency)."""
    env_path = HERE.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def load_cases() -> list:
    with open(HERE / "test_cases.json", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(case: dict) -> str:
    parts = []
    if case.get("context"):
        doc = DOCS / case["context"]
        parts.append("السياق:\n" + doc.read_text(encoding="utf-8"))
    parts.append("السؤال: " + case["prompt"])
    parts.append("أجب بالعربية باختصار.")
    return "\n\n".join(parts)


def openai_generate(model: str, prompt: str, base_url: str) -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY or OPENROUTER_API_KEY")
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def ollama_generate(model: str, prompt: str, url: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{url}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["response"]


def gemini_generate(model: str, prompt: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY (from Google AI Studio)")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["candidates"][0]["content"]["parts"][0]["text"]


def gemini_list_models() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY (from Google AI Studio)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    for m in data.get("models", []):
        print(m["name"].replace("models/", ""))


def score(answer: str, expected_keywords: list) -> float:
    if not expected_keywords:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw in answer)
    return hits / len(expected_keywords)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated, e.g. gpt-4o-mini or ollama:qwen2.5:7b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--base-url", default="https://api.openai.com/v1",
                    help="API base URL (use https://openrouter.ai/api/v1 for OpenRouter)")
    ap.add_argument("--list-gemini-models", action="store_true",
                    help="list models available to your GEMINI_API_KEY and exit")
    args = ap.parse_args()

    if args.list_gemini_models:
        gemini_list_models()
        return
    if not args.models:
        ap.error("--models is required (unless --list-gemini-models)")

    cases = load_cases()
    rows = []
    for model in [m.strip() for m in args.models.split(",")]:
        scores, details, err_count = [], [], 0
        for case in cases:
            prompt = build_prompt(case)
            try:
                if model.startswith("ollama:"):
                    answer = ollama_generate(model[len("ollama:"):], prompt, args.ollama_url)
                elif model.startswith("gemini:"):
                    answer = gemini_generate(model[len("gemini:"):], prompt)
                else:
                    answer = openai_generate(model, prompt, args.base_url)
                s = score(answer, case["expected_keywords"])
                details.append({"case_id": case["id"], "score": s, "error": None})
            except Exception as e:  # noqa: BLE001 - report, don't crash the run
                s = 0.0
                err_count += 1
                details.append({"case_id": case["id"], "score": s, "error": str(e)})
            scores.append(s)
        avg = sum(scores) / len(scores) if scores else 0.0
        rows.append((model, avg, details))
        if err_count:
            first_err = next(d["error"] for d in details if d["error"])
            print(f"!! {model}: {err_count}/{len(cases)} cases errored. First error: {first_err}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RESULTS / f"run-{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"model": m, "avg_score": a, "per_case": d} for m, a, d in rows],
            f, ensure_ascii=False, indent=2,
        )

    print("| النموذج | متوسط الدرجة |")
    print("|---|---|")
    for model, avg, _ in sorted(rows, key=lambda r: r[1], reverse=True):
        print(f"| {model} | {avg:.2f} |")
    print(f"\nsaved to eval/model_comparison/results/run-{stamp}.json")


if __name__ == "__main__":
    sys.exit(main())
