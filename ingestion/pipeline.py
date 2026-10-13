"""ETL pipeline (P1-T01): extract -> transform -> load.

Reads raw sources, applies basic cleaning, and emits DocumentChunk records
(backend.models contracts) as JSONL.

Scope note: each document becomes exactly one chunk here. Semantic
chunking (splitting documents) is P2-T01's job; deep Arabic normalization
is P1-T04's job.

Usage:
    python -m ingestion.pipeline --input data/sample/documents --output out/chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import DocumentChunk, SourceType  # noqa: E402


def _doc_id(source_uri: str) -> str:
    return "doc-" + hashlib.sha256(source_uri.encode()).hexdigest()[:12]


def _chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}-c{index:04d}"


def clean_text(text: str) -> str:
    """Basic cleaning: unify newlines, collapse spaces, drop empty lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out, blanks = [], 0
    for line in lines:
        if not line:
            blanks += 1
            if blanks == 1:
                out.append(line)
        else:
            blanks = 0
            out.append(line)
    return "\n".join(out).strip()


class SourceReader:
    """Base class: knows which file extensions it handles."""

    extensions: tuple[str, ...] = ()
    source_type: SourceType = SourceType.TEXT

    def read(self, path: Path) -> str:
        raise NotImplementedError


class MarkdownReader(SourceReader):
    extensions = (".md", ".markdown")
    source_type = SourceType.MARKDOWN

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


class TextReader(SourceReader):
    extensions = (".txt",)
    source_type = SourceType.TEXT

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


READERS: list[SourceReader] = [MarkdownReader(), TextReader()]


def reader_for(path: Path) -> SourceReader | None:
    ext = path.suffix.lower()
    for reader in READERS:
        if ext in reader.extensions:
            return reader
    return None


def run_pipeline(input_dir: Path, output: Path) -> list[DocumentChunk]:
    """Run extract -> transform -> load over every readable file in input_dir."""
    chunks: list[DocumentChunk] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        reader = reader_for(path)
        if reader is None:
            print(f"skip (no reader): {path.name}", file=sys.stderr)
            continue
        text = clean_text(reader.read(path))
        if not text:
            print(f"skip (empty after cleaning): {path.name}", file=sys.stderr)
            continue
        document_id = _doc_id(str(path))
        chunks.append(
            DocumentChunk(
                id=_chunk_id(document_id, 0),
                document_id=document_id,
                source_type=reader.source_type,
                source_uri=str(path),
                text=text,
                chunk_index=0,
                metadata={"filename": path.name},
                created_at=datetime.now(timezone.utc),
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    # Validate the round-trip: everything written must satisfy the contract.
    with open(output, encoding="utf-8") as f:
        for line in f:
            DocumentChunk.model_validate_json(line)

    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="AION ETL pipeline (P1-T01)")
    ap.add_argument("--input", required=True, help="input directory with raw documents")
    ap.add_argument("--output", required=True, help="output JSONL path")
    args = ap.parse_args()
    chunks = run_pipeline(Path(args.input), Path(args.output))
    print(f"wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
