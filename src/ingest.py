"""Ingestion: load raw source files from documents/ into normalized records.

Each loader returns a list of dicts with a shared schema so the chunker doesn't
need to know which source a record came from:

    {
        "id":         str,           # stable, source-prefixed
        "text":       str,           # cleaned review/comment/description body
        "professor":  str | None,
        "course":     str | None,    # e.g. "CSE 214"
        "rating":     float | None,  # /5
        "difficulty": float | None,  # /5
        "source":     str,           # human-readable provenance
        "date":       str | None,
        "url":        str | None,
    }

To swap in real scraped data later, point these loaders at the real files (same
JSON/HTML shapes) — nothing downstream changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from .clean import clean_text

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


def _norm_course(code: str | None) -> str | None:
    """'CSE214' / 'cse 214' -> 'CSE 214'."""
    if not code:
        return None
    m = re.match(r"\s*([A-Za-z]{2,4})\s*0*(\d{2,3})", code)
    return f"{m.group(1).upper()} {m.group(2)}" if m else code.strip()


def load_ratemyprofessors(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for prof in data.get("professors", []):
        name = prof.get("professor")
        for r in prof.get("reviews", []):
            records.append(
                {
                    "id": r["id"],
                    "text": clean_text(r.get("comment")),
                    "professor": name,
                    "course": _norm_course(r.get("class")),
                    "rating": r.get("quality"),
                    "difficulty": r.get("difficulty"),
                    "source": "RateMyProfessors",
                    "date": r.get("date"),
                    "url": None,
                }
            )
    return records


def load_reddit(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for thread in data.get("threads", []):
        url = thread.get("thread_url")
        for c in thread.get("comments", []):
            records.append(
                {
                    "id": c["id"],
                    "text": clean_text(c.get("body_html")),
                    "professor": None,  # not reliably tagged in free-text comments
                    "course": _norm_course(c.get("course")),
                    "rating": None,
                    "difficulty": None,
                    "source": f"Reddit r/{data.get('subreddit', 'SBU')}",
                    "date": c.get("created"),
                    "url": url,
                }
            )
    return records


def load_catalog(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    records: list[dict] = []
    for i, div in enumerate(soup.select("div.course")):
        heading = div.find("h3")
        desc = div.find(class_="desc")
        if not heading or not desc:
            continue
        title = clean_text(heading.get_text())
        course = _norm_course(title.split(":")[0])
        records.append(
            {
                "id": f"catalog-{i}",
                "text": f"{title}. {clean_text(desc.get_text())}",
                "professor": None,
                "course": course,
                "rating": None,
                "difficulty": None,
                "source": "SBU CSE Catalog",
                "date": None,
                "url": None,
            }
        )
    return records


def load_all(documents_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """Load every known source present in documents/."""
    records: list[dict] = []
    loaders = {
        "ratemyprofessors_cse.json": load_ratemyprofessors,
        "reddit_rSBU.json": load_reddit,
        "sbu_cse_catalog.html": load_catalog,
    }
    for filename, loader in loaders.items():
        path = documents_dir / filename
        if path.exists():
            loaded = loader(path)
            for rec in loaded:  # stamp the originating file for attribution
                rec["source_file"] = filename
            records.extend(loaded)
    return records


if __name__ == "__main__":
    recs = load_all()
    print(f"Loaded {len(recs)} records from {DOCUMENTS_DIR}")
    by_source: dict[str, int] = {}
    for r in recs:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")
