# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

An unofficial guide to student experiences with Computer Engineering professors and courses at NYU Tandon, built from crowd-sourced reviews (RateMyProfessors) and student threads (Reddit r/nyu).

This knowledge is valuable because official channels tell you *what* a course covers, not *what it's like to take it* — the real workload, grading fairness, organization, and teaching quality. It's hard to find officially because NYU doesn't publish candid evaluations, and the honest signal is scattered across separate review pages and long Reddit threads. A RAG system collapses that into one searchable, attributable answer.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | RateMyProfessors |Reviews of a Specific Professor in Computer Engineering Department |https://www.ratemyprofessors.com/professor/2574177 |
| 2 | RateMyProfessors |Reviews of a Specific Professor in Computer Engineering Department | https://www.ratemyprofessors.com/professor/2793957|
| 3 | RateMyProfessors | Reviews of a Specific Professor in Computer Engineering Department| https://www.ratemyprofessors.com/professor/2609063|
| 4 | RateMyProfessors |Reviews of a Specific Professor in Computer Engineering Department |https://www.ratemyprofessors.com/professor/2822036 |
| 5 | RateMyProfessors |Reviews of a Specific Professor in Computer Engineering Department | https://www.ratemyprofessors.com/professor/3107171|
| 6 | RateMyProfessors|Reviews of a Specific Professor in Computer Engineering Department | https://www.ratemyprofessors.com/professor/2346919|
| 7 |RateMyProfessors  |Reviews of a Specific Professor in Computer Engineering Department |https://www.ratemyprofessors.com/professor/2700917 |
| 8 |RateMyProfessors  | Reviews of a Specific Professor in Computer Engineering Department|https://www.ratemyprofessors.com/professor/1103208 |
| 9 | Reddit|Awful Quality of Education at Tandon
 Thread on Reddit |https://www.reddit.com/r/nyu/comments/1g1ypkm/awful_quality_of_education_at_tandon/ |
| 10 | Reddit| Worries About Tandon thread on reddit |https://www.reddit.com/r/nyu/comments/g6xa4t/worries_about_tandon/ |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Strategy:** Recursive chunking.

**Chunk size:** ~800 characters max. Most reviews and comments are well under this, so each stays in one whole chunk; only a rare long Reddit comment gets split.

**Overlap:** ~100 characters. This only matters when a long comment is split — it keeps a sentence from losing its lead-in. Separate reviews never overlap.

**Reasoning:** Recursive splitting respects natural structure first (review/comment boundary, then paragraph, then sentence) and only falls back to smaller cuts when something is too big. Each review or comment is already a short, self-contained opinion, so the chunk should be the whole review, not a fixed slice that cuts a complaint in half. I avoided semantic because it's overkill and slow for reviews that are already tiny, pre-segmented opinions with no long topic shifts to detect. Each chunk also stores its metadata (professor name, rating, difficulty, would-take-again %, course, source URL) so questions like "which professor is rated highest?" can be answered.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `all-MiniLM-L6-v2` via sentence-transformers — free, local, fast, and built for short text, which fits my one-opinion-per-chunk.

**Top-k:** 5. Chunks are tiny, so I need several opinions to give a fair answer, not just one review.

**Production tradeoff reflection:** With no cost limit I'd consider a stronger model like `all-mpnet-base-v2` or a hosted one (OpenAI) for better matching on informal review text. Context length isn't a concern since chunks are short; the real tradeoffs are accuracy on slangy text vs. latency and cost

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Amit Patel's attendance policy? | Very strict — being even 5 minutes late counts as an absence; students call him "old school" and note he emailed mandating attendance or points are deducted. (Rating 3.7, difficulty 2.5) |
| 2 | Is Fraida Fund's course a heavy workload, and who is it suited for? | Yes — very heavy (homework, labs, hard exams); one review says "RUN! DON'T TAKE IT UNLESS YOU ARE A SENIOR SDE." High difficulty (4.2), only 61% would take again, though many call it comprehensive/excellent. |
| 3 | What are the main complaints about Chinmay Hegde's grading? | Unclear and harsh — points deducted for requirements never listed on the assignment sheet, and a grading system that "pits students against each other." Low rating (2.0), 37% would take again. |
| 4 | According to the Reddit thread, what specific complaints do students raise about the quality of education at Tandon? | Instruction quality feels subpar and not worth the ~$31k/semester tuition; disengaged professors, poorly structured content, and unfair, rubric-less grading with dismissive or no responses (the OP, a CE sophomore, had an OOP grade drop A→F over a concept taught before the deadline). Commenters add that ~30% of professors aren't good, that about half the school is "grumpy old tenured" Polytech-era profs, and that quality is mediocre and varies a lot — though some name standout professors (e.g., Campisi for embedded systems, Aronov, Sterling) and advise using RateMyProfessor, auditing classes, and Student Advocacy. |
| 5 | What concerns do prospective students raise about attending Tandon, and how do current students respond? | Worries: mediocre professors, a weaker/less-theoretical curriculum than CAS, low odds of top-company internships, and Tandon's reputation hurting job prospects. Current students largely reassure: professors vary (some weak older Poly-era profs, but new hires are good — check RateMyProfessor); the curriculum is standard and math/science is actually in-depth or harder than CAS since it's tailored to engineers; many got internships at Google, Apple, Microsoft, and Facebook (some as freshmen); and employers see the "NYU" name, so it won't hinder hiring. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Thin chunks.** Many reviews are only a sentence or two, so a single retrieved chunk may not carry enough context to answer well. .

2. **Cross-professor mix-ups.** Reviews are short and worded similarly ("disorganized," "tough grader"), so a query about one professor can retrieve another's reviews.
---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

![RAG pipeline architecture: PDF ingestion (pdfplumber) → recursive chunking (LangChain) → embeddings (all-MiniLM-L6-v2) + vector store (ChromaDB) → retrieval (top-k 5) → generation (Groq llama-3.3-70b-versatile)](assets/architecture.png)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->


I'll use **Claude** as my coding assistant. For each stage I give it the relevant planning.md section plus the assignment requirement, review the output against my spec instead of accepting it blindly, and correct anything that doesn't fit.

**Ingestion and chunking:** Input: my Documents + Chunking sections. Ask Claude for a `pdfplumber` loader that extracts text and attaches source metadata (professor name / source file), plus a recursive chunker (LangChain, ~800 chars / ~100 overlap). Expect: clean chunks, one review each. **Verify:** print 5 chunks and confirm they're self-contained with no leftover boilerplate, and that total chunk count lands in the 50–2,000 range.

**Embedding and retrieval:** Input: my Retrieval section + the diagram. Ask Claude to embed chunks with `all-MiniLM-L6-v2`, store them in ChromaDB with source metadata, and write a `retrieve(query, k=5)` function returning chunks + distance scores. Expect: working semantic search. **Verify:** run 3+ eval questions and confirm top results are on-topic with distance scores below 0.5.

**Generation and interface:** Input: my Architecture diagram + grounding requirement. Ask Claude to pass retrieved chunks to a Groq `llama-3.3-70b-versatile` call whose system prompt answers *only* from context (and says "I don't have enough information" otherwise), append source names programmatically, and wrap it in a Gradio UI. **Verify:** run all 5 eval questions end-to-end, confirm answers are grounded with visible citations, and that an out-of-scope question is refused.