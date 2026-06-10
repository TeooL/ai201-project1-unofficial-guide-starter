"""The Unofficial Guide — Gradio web interface (Milestone 5).

A browser chat UI over the same grounded-generation backend as the CLI. Run:

    python app_web.py

Then open the printed local URL (default http://127.0.0.1:7860).
"""

from __future__ import annotations

import gradio as gr

from src.embed import COLLECTION_NAME, build_index, get_client
from src.generate import answer


def _ensure_index() -> None:
    """Build the vector store on first run if it isn't there yet."""
    try:
        get_client().get_collection(COLLECTION_NAME)
    except Exception:
        build_index()


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["", "**Sources**"]
    for s in sources:
        lines.append(f"- `[{s['n']}]` {s['label']} — *distance {s['distance']}*")
    return "\n".join(lines)


def respond(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "Type a question about an SBU CSE professor or course above."
    result = answer(query)
    return result["answer"] + _format_sources(result["sources"])


EXAMPLES = [
    "What's the grading like in CSE 214 with Professor Mercer?",
    "Is Professor Rahman a good choice for CSE 114 if I'm new to programming?",
    "How do students describe Professor Osei's lectures and exams in CSE 220?",
    "Should I take CSE 214 with Mercer or Rahman?",
    "Which dining hall has the best food?",  # out-of-domain -> should decline
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="The Unofficial Guide — SBU CSE") as demo:
        gr.Markdown(
            "# The Unofficial Guide\n"
            "Ask about **Computer Science professors at Stony Brook University**. "
            "Answers come *only* from collected student reviews and cite their sources. "
            "If the reviews don't cover your question, the guide will say so rather than guess."
        )
        query = gr.Textbox(
            label="Your question",
            placeholder="e.g. What's grading like in CSE 214 with Professor Mercer?",
            lines=2,
        )
        ask = gr.Button("Ask", variant="primary")
        out = gr.Markdown(label="Answer")
        gr.Examples(examples=EXAMPLES, inputs=query)

        ask.click(respond, inputs=query, outputs=out)
        query.submit(respond, inputs=query, outputs=out)
    return demo


if __name__ == "__main__":
    _ensure_index()
    build_ui().launch()
