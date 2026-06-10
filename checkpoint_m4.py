"""Milestone 4 checkpoint: embed -> store -> test retrieval.

Builds the vector store, then runs eval-plan queries and prints the returned
chunks with cosine distances. Flags any top result with distance >= 0.5 (the
checkpoint threshold).

Usage:
    python checkpoint_m4.py
"""

from __future__ import annotations

from src.embed import build_index, retrieve

# Eval-plan queries (planning.md), with placeholder names resolved to the
# fictional professors in the synthetic corpus.
QUERIES = [
    ("Q1", "What do students say about the grading difficulty of CSE 214 with Professor Mercer?", None),
    ("Q2", "Is Professor Rahman a good choice for CSE 114 if I'm new to programming?", None),
    ("Q3", "How do students describe Professor Osei's lecture style and exams in CSE 220?", None),
    ("Q5", "Do students think attendance and participation matter in Professor Whitfield's CSE 101 class?", None),
]

K = 5
THRESHOLD = 0.5


def main() -> None:
    n = build_index()
    print(f"Indexed {n} chunks. Retrieving top-{K} per query (cosine distance).\n")

    for tag, query, where in QUERIES:
        hits = retrieve(query, k=K, where=where)
        best = hits[0]["distance"] if hits else None
        flag = "" if best is not None and best < THRESHOLD else "  <-- top distance >= 0.5"
        print("=" * 72)
        print(f"{tag}: {query}{flag}")
        print("=" * 72)
        for rank, h in enumerate(hits, 1):
            md = h["metadata"]
            print(
                f"\n[{rank}] distance={h['distance']:.3f} | source={md.get('source')} "
                f"| file={md.get('source_file')} | pos={md.get('chunk_index')} "
                f"| professor={md.get('professor')} | course={md.get('course')}"
            )
            print(f"    {h['text'].replace(chr(10), ' ')}")
        print()

    # Demonstrate the planned metadata pre-filter: same Q1, restricted to CSE 214.
    print("=" * 72)
    print("METADATA PRE-FILTER DEMO — Q1 restricted to where={'course': 'CSE 214'}")
    print("=" * 72)
    for rank, h in enumerate(retrieve(QUERIES[0][1], k=K, where={"course": "CSE 214"}), 1):
        md = h["metadata"]
        print(f"[{rank}] distance={h['distance']:.3f} | {md.get('professor')} | {md.get('course')} "
              f"| {md.get('source')}")
    print()


if __name__ == "__main__":
    main()
