"""Semantic chunker (P2-T01): structure-aware text splitting.

Splits documents into chunks that respect meaning boundaries:
paragraphs are never torn apart unless they exceed max_chars
(then they split on sentence boundaries), and consecutive chunks
share a small overlap so context is not lost at the edges.

True embedding-based topic segmentation can refine this later;
structure-aware splitting is the correct foundation (P2-T02 builds
embeddings on top of these chunks).

Usage:
    python -m rag.chunker --input out/chunks.jsonl --output out/chunks_split.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import DocumentChunk  # noqa: E402

# Sentence ends: Latin + Arabic punctuation.
_SENTENCE_END = re.compile(r"(?<=[.!?؟…])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences (Arabic-aware)."""
    parts = _SENTENCE_END.split(paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines."""
    parts = _PARAGRAPH_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    """Split text into overlapping, structure-aware chunks."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(current))

    for para in split_paragraphs(text):
        # Oversized paragraph: split it into sentences first.
        units = [para] if len(para) <= max_chars else split_sentences(para)
        for unit in units:
            if current_len + len(unit) + 2 > max_chars and current:
                flush()
                current, current_len = [], 0
            current.append(unit)
            current_len += len(unit) + 2
    flush()

    if overlap <= 0 or len(chunks) < 2:
        return chunks

    # Add overlap: prepend the tail of the previous chunk.
    overlapped = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        # Avoid duplicating text when the tail is already a prefix.
        overlapped.append(tail + cur if not cur.startswith(tail) else cur)
    return overlapped


def chunk_document(chunk: DocumentChunk, max_chars: int = 1000, overlap: int = 100) -> list[DocumentChunk]:
    """Split one document-level chunk into several indexed chunks."""
    texts = chunk_text(chunk.text, max_chars=max_chars, overlap=overlap)
    out: list[DocumentChunk] = []
    for i, text in enumerate(texts):
        out.append(
            DocumentChunk(
                id=f"{chunk.document_id}-c{i:04d}",
                document_id=chunk.document_id,
                source_type=chunk.source_type,
                source_uri=chunk.source_uri,
                text=text,
                chunk_index=i,
                metadata={**(chunk.metadata or {}), "parent_chunk_id": chunk.id},
                created_at=datetime.now(timezone.utc),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic chunker (P2-T01)")
    ap.add_argument("--input", required=True, help="input DocumentChunk JSONL")
    ap.add_argument("--output", required=True, help="output DocumentChunk JSONL")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    total_in, total_out = 0, 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as fin, open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total_in += 1
            doc = DocumentChunk.model_validate_json(line)
            for chunk in chunk_document(doc, max_chars=args.max_chars, overlap=args.overlap):
                fout.write(chunk.model_dump_json() + "\n")
                total_out += 1

    print(f"split {total_in} documents into {total_out} chunks -> {args.output}")


if __name__ == "__main__":
    main()
