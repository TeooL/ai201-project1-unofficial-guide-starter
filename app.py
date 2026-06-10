"""The Unofficial Guide — interactive CLI (Milestone 5).

Ask questions about SBU CSE professors; answers are grounded in collected
student reviews and cite their sources. Run:

    python app.py
"""

from __future__ import annotations

from src.embed import build_index, get_client, COLLECTION_NAME
from src.generate import answer

BANNER = r"""
============================================================
  The Unofficial Guide — SBU CSE professor reviews
  Ask things like:
    - What's grading like in CSE 214 with Professor Mercer?
    - Is Professor Rahman good for CSE 114 beginners?
  Type 'quit' or 'exit' to leave.
============================================================
"""


def _ensure_index() -> None:
    """Build the vector store on first run if it isn't there yet."""
    try:
        get_client().get_collection(COLLECTION_NAME)
    except Exception:
        print("Building the review index (first run)...")
        n = build_index()
        print(f"Indexed {n} chunks.\n")


def main() -> None:
    _ensure_index()
    print(BANNER)
    while True:
        try:
            query = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        result = answer(query)
        print(f"\n{result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"  [{s['n']}] {s['label']}  (distance {s['distance']})")
        print()


if __name__ == "__main__":
    main()
