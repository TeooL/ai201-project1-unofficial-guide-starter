"""Record-level chunking.

Strategy (see planning.md → Chunking Strategy): one review / comment / catalog
entry == one chunk. Metadata is prepended to the chunk text as an attribution
header so retrieval and generation always know which professor/course/source a
chunk belongs to. No overlap between records (they are independent and
single-professor); a small overlap is applied ONLY when a single over-long post
must be split across the ~256-token cap.
"""

from __future__ import annotations

import re

MAX_TOKENS = 256
OVERLAP_TOKENS = 40

# Approx tokens from words: English averages ~0.75 words per token.
_WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / _WORDS_PER_TOKEN))


def build_header(rec: dict) -> str:
    """Compact attribution line, e.g.
    'Professor Daniel Mercer | CSE 214 | rating 2.1/5, difficulty 4.6/5 | RateMyProfessors | 2023-09-14'
    """
    parts: list[str] = []
    if rec.get("professor"):
        parts.append(f"Professor {rec['professor']}")
    if rec.get("course"):
        parts.append(rec["course"])
    rd = []
    if rec.get("rating") is not None:
        rd.append(f"rating {rec['rating']}/5")
    if rec.get("difficulty") is not None:
        rd.append(f"difficulty {rec['difficulty']}/5")
    if rd:
        parts.append(", ".join(rd))
    if rec.get("source"):
        parts.append(rec["source"])
    if rec.get("date"):
        parts.append(rec["date"])
    return " | ".join(parts)


def _split_long(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split an over-long body on sentence boundaries, carrying a word overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    overlap_words = max(0, round(overlap_tokens * _WORDS_PER_TOKEN))
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        st = estimate_tokens(sent)
        if cur and cur_tokens + st > max_tokens:
            pieces.append(" ".join(cur))
            tail = " ".join(cur).split()[-overlap_words:] if overlap_words else []
            cur = [" ".join(tail)] if tail else []
            cur_tokens = estimate_tokens(cur[0]) if cur else 0
        cur.append(sent)
        cur_tokens += st
    if cur:
        pieces.append(" ".join(cur))
    return [p.strip() for p in pieces if p.strip()]


def _make_chunk(rec: dict, body: str, part: int, n_parts: int) -> dict:
    header = build_header(rec)
    suffix = f" (part {part + 1}/{n_parts})" if n_parts > 1 else ""
    text = f"{header}{suffix}\n{body}"
    return {
        "id": rec["id"] if n_parts == 1 else f"{rec['id']}-{part}",
        "text": text,
        "metadata": {
            "professor": rec.get("professor"),
            "course": rec.get("course"),
            "rating": rec.get("rating"),
            "difficulty": rec.get("difficulty"),
            "source": rec.get("source"),
            "source_file": rec.get("source_file"),
            "date": rec.get("date"),
            "url": rec.get("url"),
            "doc_id": rec["id"],
        },
    }


def chunk_records(
    records: list[dict],
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Turn normalized records into chunks. Empty bodies are dropped."""
    chunks: list[dict] = []
    for rec in records:
        body = (rec.get("text") or "").strip()
        if not body:  # empty-chunk guard
            continue
        if estimate_tokens(body) <= max_tokens:
            chunks.append(_make_chunk(rec, body, 0, 1))
        else:
            parts = _split_long(body, max_tokens, overlap_tokens)
            for i, piece in enumerate(parts):
                chunks.append(_make_chunk(rec, piece, i, len(parts)))
    # Final safety filter: never emit a zero-length chunk.
    return [c for c in chunks if c["text"].strip()]
