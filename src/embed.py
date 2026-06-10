"""Embedding + vector store + retrieval (Milestone 4).

Pipeline stage 3-4 from planning.md:
    chunks --(all-MiniLM-L6-v2)--> embeddings --> ChromaDB (cosine)
    query  --(same model)--------> embedding   --> top-k similarity search

Design choices (see planning.md → Retrieval Approach):
  * Embedding model: all-MiniLM-L6-v2 via sentence-transformers. Local, free, no
    rate limits, 384-dim, ~256-token window that matches our small chunks.
  * Distance: cosine. Embeddings are L2-normalized and the collection is created
    with hnsw:space="cosine", so a distance is 1 - cosine_similarity in [0, 2];
    relevant matches land well under 0.5.
  * Each chunk is stored with metadata for attribution AND for the planned
    metadata pre-filter (professor / course), passed to retrieve() as `where`.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .chunk import chunk_records
from .ingest import load_all

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sbu_cse_reviews"
PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

_model: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_client() -> chromadb.ClientAPI:
    """Persistent on-disk Chroma client (survives across runs)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def _embed(texts: list[str]):
    # normalize_embeddings=True -> unit vectors, so cosine distance is meaningful.
    return get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def _clean_meta(md: dict) -> dict:
    """Chroma rejects None and non-scalar metadata values — drop/coerce them."""
    out: dict = {}
    for key, value in md.items():
        if value is None:
            continue
        out[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return out


def build_index(rebuild: bool = True) -> int:
    """Ingest -> chunk -> embed -> store. Returns the number of chunks indexed."""
    chunks = chunk_records(load_all())
    client = get_client()

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # didn't exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # "chunk's position in its source document" = order among chunks from the file.
    pos_counter: dict[str, int] = {}
    ids, documents, metadatas = [], [], []
    for c in chunks:
        src_file = c["metadata"].get("source_file", "unknown")
        idx = pos_counter.get(src_file, 0)
        pos_counter[src_file] = idx + 1

        meta = _clean_meta({**c["metadata"], "chunk_index": idx})
        ids.append(c["id"])
        documents.append(c["text"])
        metadatas.append(meta)

    embeddings = _embed(documents)
    collection.add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return len(ids)


def get_collection():
    return get_client().get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    """Return the top-k chunks for a query.

    `where` is an optional ChromaDB metadata filter, e.g. {"course": "CSE 214"}
    or {"professor": "Daniel Mercer"} — the planned metadata pre-filter.
    """
    collection = get_collection()
    result = collection.query(
        query_embeddings=_embed([query]),
        n_results=k,
        where=where,
    )
    hits: list[dict] = []
    # Chroma returns parallel lists nested one level per query; we sent one query.
    for cid, doc, dist, meta in zip(
        result["ids"][0],
        result["documents"][0],
        result["distances"][0],
        result["metadatas"][0],
    ):
        hits.append({"id": cid, "text": doc, "distance": dist, "metadata": meta})
    return hits


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into '{COLLECTION_NAME}' at {PERSIST_DIR}")
