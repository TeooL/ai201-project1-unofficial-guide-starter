"""Text cleaning: strip HTML and normalize whitespace.

Reviews and Reddit comments arrive with HTML tags and HTML entities; the catalog
is a full HTML page with nav/script/footer boilerplate. Everything that becomes
chunk text must pass through clean_text() first so no markup or escape sequences
leak into the vector store.
"""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

# Tags whose contents are boilerplate, not document text.
_DROP_TAGS = ("script", "style", "nav", "header", "footer")


def strip_html(raw: str) -> str:
    """Remove tags and boilerplate elements, returning plain text."""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(raw: str | None) -> str:
    """Unescape entities, strip any HTML, and collapse whitespace.

    Safe to call on already-clean text — it just normalizes whitespace then.
    """
    if not raw:
        return ""
    text = html.unescape(raw)
    # Only invoke the (relatively expensive) parser if it looks like markup.
    if "<" in text and ">" in text:
        text = strip_html(text)
        text = html.unescape(text)  # entities can survive inside tag text
    text = re.sub(r"\s+", " ", text).strip()
    return text
