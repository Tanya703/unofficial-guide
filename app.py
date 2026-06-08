"""
Generation + Gradio interface for The Unofficial Guide (Milestone 5).

Pipeline stage 5 (Generation) from planning.md / the architecture diagram:
    query -> retrieve(top-k chunks) -> build a grounded prompt
          -> Groq llama-3.3-70b-versatile -> answer + programmatic source list

Grounding is enforced in TWO places, by design:

  1. SYSTEM PROMPT (enforced, not suggested): the model is told to answer using
     ONLY the provided context, to never use outside knowledge, and to return an
     EXACT refusal sentence when the context is insufficient.

  2. SOURCE ATTRIBUTION (programmatic, not LLM-generated): sources are collected
     from the retrieved chunks' own metadata and appended by THIS code after
     generation. The model is explicitly told NOT to write its own sources, so
     attribution is guaranteed and traceable regardless of what the LLM says.

Run:
    python app.py        # launches the Gradio UI at http://127.0.0.1:7860
"""

import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

from search import retrieve, TOP_K

# --- config ----------------------------------------------------------------
load_dotenv()  # read GROQ_API_KEY from .env (never committed; see .gitignore)

MODEL = "llama-3.3-70b-versatile"   # Groq free-tier, per planning.md Architecture

_API_KEY = os.environ.get("GROQ_API_KEY")
if not _API_KEY or _API_KEY == "your_key_here":
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and paste your free "
        "key from https://console.groq.com"
    )
client = Groq(api_key=_API_KEY)


# ---------------------------------------------------------------------------
# The grounding contract. This is the enforcement mechanism, not a suggestion:
# the model is restricted to the context and given an exact refusal string.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are The Unofficial Guide, a Q&A assistant about NYU Tandon \
Computer Engineering professors and courses. The CONTEXT in the user's message \
contains student reviews (RateMyProfessors) and Reddit posts.

Follow these rules WITHOUT EXCEPTION:

1. Answer using ONLY the information in the CONTEXT. Do not use any outside or \
prior knowledge, and do not guess or infer facts the context does not state.
2. If the CONTEXT does not contain enough information to answer the question, \
reply with EXACTLY this sentence and nothing else:
"I don't have enough information on that."
3. Never invent professor names, ratings, quotes, or facts that are not in the \
CONTEXT.
4. Represent the reviews faithfully. If reviews disagree, present both sides.
5. Do NOT write your own "Sources" or "Citations" section. Source attribution \
is added automatically after your answer, so just answer the question.

Answer concisely and directly."""


# ---------------------------------------------------------------------------
# Prompt assembly: hand the model the retrieved chunks as labeled context.
# ---------------------------------------------------------------------------
def _format_context(chunks):
    """Number each chunk and tag it with its source document so the model can
    ground its answer in specific reviews."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        src = c["metadata"].get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Programmatic source attribution: built from chunk metadata, NOT the LLM.
# ---------------------------------------------------------------------------
def _collect_sources(chunks):
    """Return the unique source documents behind the retrieved chunks (order
    preserved) as a list of citation strings. Because this is derived from
    metadata, attribution is guaranteed even if the model omits or invents one."""
    seen = {}
    for c in chunks:
        m = c["metadata"]
        src = m.get("source", "unknown")
        if src not in seen:
            label = m.get("title") or src           # human-readable when available
            url = m.get("url")
            seen[src] = f"{label} - {url}" if url else label
    return list(seen.values())


REFUSAL = "I don't have enough information on that."


def ask(query, k=TOP_K):
    """End-to-end RAG: retrieve -> grounded generation -> structured result.

    Returns {"answer": str, "sources": list[str]}. On a refusal (or empty/blank
    query) the sources list is empty -- we never attach sources to a non-answer.
    """
    query = (query or "").strip()
    if not query:
        return {"answer": "Please enter a question.", "sources": []}

    chunks = retrieve(query, k=k)
    if not chunks:
        return {"answer": REFUSAL, "sources": []}

    user_msg = (
        f"CONTEXT:\n{_format_context(chunks)}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,  # factual task: no creative drift away from the context
    )
    llm_answer = resp.choices[0].message.content.strip()

    # If the model refused, there is nothing to attribute -- don't fake sources.
    if llm_answer.lower().startswith("i don't have enough information"):
        return {"answer": REFUSAL, "sources": []}

    return {"answer": llm_answer, "sources": _collect_sources(chunks)}


def handle_query(question):
    """Gradio adapter: run ask() and split the result into the two output boxes
    (answer, sources). Sources are formatted here for display only."""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"]) or "—"
    return result["answer"], sources


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="The Unofficial Guide") as demo:
        gr.Markdown(
            "# The Unofficial Guide — NYU Tandon Computer Engineering\n"
            "Ask about CE professors and courses. Answers are grounded **only** in "
            "student reviews (RateMyProfessors) and Reddit threads — with sources cited. "
            "If the reviews don't cover it, the assistant says so instead of guessing."
        )
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. What do students say about Amit Patel's attendance policy?",
            lines=2,
        )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_box = gr.Textbox(label="Answer", lines=8)
        sources_box = gr.Textbox(label="Retrieved from (sources)", lines=4)

        ask_btn.click(handle_query, inputs=question, outputs=[answer_box, sources_box])
        question.submit(handle_query, inputs=question, outputs=[answer_box, sources_box])

        gr.Examples(
            examples=[
                "What do students say about Amit Patel's attendance policy?",
                "Is Fraida Fund's course a heavy workload, and who is it suited for?",
                "What are the main complaints about Chinmay Hegde's grading?",
                "What is the best dining hall at NYU?",  # out-of-scope -> should refuse
            ],
            inputs=question,
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
