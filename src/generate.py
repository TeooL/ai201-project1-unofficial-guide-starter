"""Grounded generation (Milestone 5).

Pipeline stage 5: retrieved review chunks + question -> grounded answer with
source attribution, via the Groq API (llama-3.3-70b-versatile).

Grounding is enforced two ways (see planning.md -> Anticipated Challenges):

1. Structural: a distance threshold drops loosely-related chunks before the LLM
   sees them. If nothing survives the threshold, we refuse WITHOUT calling the
   LLM at all — an out-of-domain question can't be answered from context that
   isn't there, so there's nothing to hallucinate from.
2. Prompt: the system prompt restricts the model to the numbered context, makes
   it cite sources with [n] tags, and tells it to decline when the reviews don't
   cover the question.
"""

from __future__ import annotations

import re

from dotenv import load_dotenv
from groq import Groq

from .embed import retrieve

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
TOP_K = 8           # planning.md: gather enough reviews for consensus to emerge
MAX_DISTANCE = 0.55  # cosine distance above this = too weak to count as coverage

REFUSAL = (
    "I don't have enough student feedback in my sources to answer that. "
    "My knowledge is limited to the SBU CSE professor reviews I've collected."
)

SYSTEM_PROMPT = """You are The Unofficial Guide, a tool that answers questions about \
Computer Science professors at Stony Brook University using ONLY student reviews \
provided to you as numbered context.

Rules:
- Answer strictly from the numbered SOURCES below. Do not use any outside knowledge.
- Cite the sources you used with bracketed numbers, e.g. [1], [3], inline in your answer.
- When reviews disagree, say so and represent both sides rather than flattening them.
- If the SOURCES do not contain enough information to answer the question, reply with \
EXACTLY this and nothing else: "INSUFFICIENT_CONTEXT"
- Be concise. Synthesize across reviews; don't just quote one."""

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


def _format_sources(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h['text']}")
    return "\n\n".join(blocks)


def _source_label(h: dict) -> str:
    md = h["metadata"]
    bits = [md.get("source", "?")]
    if md.get("professor"):
        bits.append(md["professor"])
    if md.get("course"):
        bits.append(md["course"])
    if md.get("date"):
        bits.append(md["date"])
    return " — ".join(str(b) for b in bits) + f"  ({md.get('source_file', '?')})"


def answer(query: str, k: int = TOP_K, max_distance: float = MAX_DISTANCE) -> dict:
    """Return {'answer', 'sources', 'refused', 'hits'} for a query."""
    hits = retrieve(query, k=k)
    kept = [h for h in hits if h["distance"] <= max_distance]

    # Structural refusal: no sufficiently-relevant context -> don't even call the LLM.
    if not kept:
        return {"answer": REFUSAL, "sources": [], "refused": True, "hits": hits}

    context = _format_sources(kept)
    user_msg = f"SOURCES:\n{context}\n\nQUESTION: {query}"

    completion = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = completion.choices[0].message.content.strip()

    # Prompt-level refusal sentinel -> normalize to the standard refusal message.
    if "INSUFFICIENT_CONTEXT" in text:
        return {"answer": REFUSAL, "sources": [], "refused": True, "hits": hits}

    # Show only the sources the answer actually cited (parse [n] / [1, 2] tags),
    # preserving their original numbers so inline tags match the list. If the
    # model cited nothing, fall back to listing all kept context.
    cited: set[int] = set()
    for group in re.findall(r"\[([\d,\s]+)\]", text):
        cited.update(int(num) for num in re.findall(r"\d+", group))

    sources = [
        {"n": i, "label": _source_label(h), "distance": round(h["distance"], 3)}
        for i, h in enumerate(kept, 1)
        if not cited or i in cited
    ]
    return {"answer": text, "sources": sources, "refused": False, "hits": hits}
