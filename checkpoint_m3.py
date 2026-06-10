"""Milestone 3 checkpoint: ingest -> chunk -> inspect.

Prints corpus/chunk stats, runs automatic sanity checks (empty chunks, leftover
HTML, uniform length, metadata presence), then prints 5 random chunks in full.

Usage:
    python checkpoint_m3.py [--seed N]
"""

from __future__ import annotations

import argparse
import random
import re

from src.chunk import chunk_records, estimate_tokens
from src.ingest import load_all

HTML_TAG = re.compile(r"<[^>]+>")
HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    args = ap.parse_args()

    records = load_all()
    chunks = chunk_records(records)

    # ---- corpus summary -------------------------------------------------
    by_source: dict[str, int] = {}
    for c in chunks:
        s = c["metadata"]["source"]
        by_source[s] = by_source.get(s, 0) + 1
    token_counts = [estimate_tokens(c["text"]) for c in chunks]

    print("=" * 70)
    print("MILESTONE 3 CHECKPOINT — ingestion & chunking")
    print("=" * 70)
    print(f"Records loaded : {len(records)}")
    print(f"Chunks produced: {len(chunks)}")
    print("Chunks by source:")
    for s, n in sorted(by_source.items()):
        print(f"  - {s}: {n}")
    if token_counts:
        print(
            f"Chunk size (est. tokens): min={min(token_counts)} "
            f"max={max(token_counts)} avg={sum(token_counts) / len(token_counts):.0f}"
        )

    # ---- automatic sanity checks ---------------------------------------
    print("\nSanity checks:")
    empties = [c for c in chunks if not c["text"].strip()]
    htmlish = [c for c in chunks if HTML_TAG.search(c["text"]) or HTML_ENTITY.search(c["text"])]
    no_source = [c for c in chunks if not c["metadata"].get("source")]
    uniform = len(set(token_counts)) <= 1 and len(token_counts) > 1

    def line(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'WARN'}] {label}{(' — ' + detail) if detail else ''}")

    line("no empty chunks", not empties, f"{len(empties)} empty" if empties else "")
    line("no HTML tags / entities left", not htmlish,
         f"{len(htmlish)} suspect" if htmlish else "")
    line("every chunk has a source", not no_source,
         f"{len(no_source)} missing" if no_source else "")
    line("chunk lengths vary (not mechanical)", not uniform,
         "all identical length" if uniform else "")

    # ---- 5 random chunks -----------------------------------------------
    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(chunks, min(5, len(chunks)))
    print("\n" + "=" * 70)
    print(f"5 RANDOM CHUNKS  (seed={args.seed})")
    print("=" * 70)
    for i, c in enumerate(sample, 1):
        md = c["metadata"]
        print(f"\n--- chunk {i}/5 | id={c['id']} | source={md['source']} "
              f"| professor={md['professor']} | course={md['course']} "
              f"| ~{estimate_tokens(c['text'])} tokens ---")
        print(c["text"])
    print()


if __name__ == "__main__":
    main()
