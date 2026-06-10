# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system over student reviews of
Computer Science professors at Stony Brook University. Ask "what's grading like
in CSE 214 with Professor Mercer?" and get an answer synthesized **only** from
collected reviews, with citations — or an honest "I don't have enough feedback"
when the question isn't covered.

> **Data note:** This build runs on a **clearly-labeled synthetic sample corpus**
> with **fictional professor names** (so no invented reviews are attached to real
> faculty). The ingestion loaders are written against the real RateMyProfessors
> JSON, Reddit `.json`, and SBU catalog HTML shapes, so real scraped/saved data
> drops in without code changes. All results below are real runs of the pipeline
> against that synthetic corpus.

## Run it

```bash
pip install -r requirements.txt          # needs GROQ_API_KEY in .env (see .env.example)
python app.py                            # interactive CLI
python app_web.py                        # Gradio web UI at http://127.0.0.1:7860
python checkpoint_m4.py                  # retrieval test (distances per query)
```

---

## Domain

**Student reviews and experiences with Computer Science (CSE) professors at Stony
Brook University.** The system covers per-professor teaching style, grading
difficulty, workload, exam character, attendance/participation policies, and
whether a course is worth taking with a given instructor.

This knowledge is valuable because course registration is a high-stakes, time-boxed
decision and the *same* course can be a manageable or punishing semester depending
on who teaches it. It is hard to find through official channels: SBU's course
catalog and registrar tell you *what* a course is and *who* teaches it, but never
*what it's actually like*. SBU does not publish its internal course evaluations to
students, so the only useful signal lives scattered and unstructured across review
sites, Reddit, and word-of-mouth — never aggregated or searchable in one place. A
RAG system is a natural fit: it pulls the relevant student voices for a specific
professor/course and synthesizes them into a direct, cited answer.

---

## Document Sources

Sources span three perspectives: crowd-sourced **opinion** (the core corpus),
and **official** records used to normalize course numbers and ground attribution.
In this synthetic build, sources #1–#10 from the plan are represented by three
sample files whose formats mirror the real sources.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | RateMyProfessors — SBU CSE professor pages | Review site (JSON via GraphQL) | https://www.ratemyprofessors.com/search/professors/971 → `documents/ratemyprofessors_cse.json` (synthetic sample) |
| 2 | Reddit — r/SBU professor/course threads | Forum (`.json` API) | https://www.reddit.com/r/SBU/ → `documents/reddit_rSBU.json` (synthetic sample) |
| 3 | Reddit — r/StonyBrook discussion | Forum | https://www.reddit.com/r/StonyBrook/ (planned secondary source) |
| 4 | Coursicle — SBU professor pages | Aggregator | https://www.coursicle.com/stonybrook/professors/ (planned) |
| 5 | Niche — SBU student reviews | Review site | https://www.niche.com/colleges/stony-brook-university/reviews/ (planned) |
| 6 | SBU CSE official course catalog | Official HTML | https://www.cs.stonybrook.edu/students/Undergraduate-Studies/courses → `documents/sbu_cse_catalog.html` (synthetic sample) |
| 7 | SBU undergraduate bulletin — CSE | Official HTML | https://www.stonybrook.edu/sb/bulletin/current/academicprograms/cse/ (planned) |
| 8 | SBU CSE faculty directory | Official HTML | https://www.cs.stonybrook.edu/people/faculty (planned, for name↔course mapping) |

Three source files are live in this build: `ratemyprofessors_cse.json` (per-professor
review records), `reddit_rSBU.json` (thread comments as HTML), and
`sbu_cse_catalog.html` (a full catalog page with nav/script/footer boilerplate).

---

## Chunking Strategy

**Chunk size:** ~256 tokens (cap). The *primary* boundary is one record = one chunk
— a single review, a single Reddit comment, or a single catalog entry. Most reviews
are 1–4 sentences and fall well under the cap; the token limit only triggers on
unusually long Reddit posts.

**Overlap:** 0 between records. A ~40-token overlap is applied *only* when one
over-long post must be split across the cap (in this corpus, exactly one Reddit post
split into two parts, `t1_b1-0` / `t1_b1-1`, with the boundary phrase repeated).

**Why these choices fit your documents:** The corpus is review-heavy and
record-structured, not long-form prose. The two failure modes that matter here are
(1) splitting one review's verdict across a boundary and (2) merging reviews about
*different professors* into one chunk — both corrupt attribution, which is the whole
product. A fixed-size sliding window (the usual 512/50 default) would routinely glue
the tail of one professor's review onto the head of another's, and overlap would make
that cross-professor bleed *worse*. So each review/comment/entry becomes one
self-contained chunk with its metadata (professor, course, rating, difficulty,
source, date) prepended as an attribution header. **Preprocessing before chunking:**
HTML is stripped with BeautifulSoup (dropping `script`/`style`/`nav`/`header`/`footer`),
HTML entities are unescaped (`&amp;`, `&#39;`, `&mdash;`), and whitespace is collapsed
— so no markup or escape sequences reach the vector store. Empty bodies are filtered
at two points.

**Final chunk count:** 21 chunks (RateMyProfessors 10, Reddit r/SBU 7, SBU CSE
Catalog 4) from 20 source records; estimated chunk size min 33 / max 208 / avg ~66
tokens.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, ~256-token
input window). Embeddings are L2-normalized and stored in ChromaDB with the collection
configured for **cosine** distance (`hnsw:space="cosine"`), so a distance is
`1 − cosine_similarity` and relevant matches land well under 0.5. It runs locally with
no API key and no rate limits, it's fast, and its small input window is a near-perfect
match for our small review-level chunks — no chunk gets truncated by the encoder.

**Production tradeoff reflection:** If cost weren't a constraint, the biggest accuracy
win would *not* come from a larger embedding model — it would come from **hybrid
retrieval**. Professor names and course numbers are exact-match keys, but embeddings
treat "Chen" and "Chang" as near-neighbors, so pure semantic search can return the
wrong professor's reviews (see Failure Case). A metadata pre-filter (restrict to the
named professor/course before ranking) plus a BM25/keyword pass for exact name matching
*structurally* removes that failure mode, whereas a bigger model only mitigates it
probabilistically. Beyond that I'd weigh a stronger model (`bge-large-en`,
`voyage-3`, OpenAI `text-embedding-3-large`) for better discrimination on short, slangy
student text — but at the cost of per-query latency and API spend, which hurts an
interactive "who should I take?" tool. **Context length** is *not* a reason to switch
(chunks are tiny — MiniLM's 256-token window is a feature, not a limit), and
**multilingual** support is unnecessary (the corpus is English-only).

---

## Grounded Generation

Generation uses the **Groq API** (`llama-3.3-70b-versatile`). Grounding is enforced
both structurally and via the prompt.

**System prompt grounding instruction** (verbatim, abridged — see `src/generate.py`):

> You are The Unofficial Guide … answer questions about Computer Science professors at
> Stony Brook University using ONLY student reviews provided to you as numbered context.
> Rules:
> - Answer strictly from the numbered SOURCES below. Do not use any outside knowledge.
> - Cite the sources you used with bracketed numbers, e.g. [1], [3], inline in your answer.
> - When reviews disagree, say so and represent both sides rather than flattening them.
> - If the SOURCES do not contain enough information to answer the question, reply with
>   EXACTLY this and nothing else: "INSUFFICIENT_CONTEXT"
> - Be concise. Synthesize across reviews; don't just quote one.

**Structural grounding:** retrieved chunks are filtered by a cosine-distance gate
(`MAX_DISTANCE = 0.55`) *before* the LLM sees them. If nothing survives the gate, the
system refuses **without calling the LLM at all** — an out-of-domain question (e.g.
"which dining hall has the best food?" lands at distance 0.69; "parking ticket" at 0.81)
has no context to hallucinate from. The `INSUFFICIENT_CONTEXT` sentinel is a second
line of defense if weak-but-passing chunks don't actually address the question.

**How source attribution is surfaced in the response:** Context chunks are numbered
`[1]…[n]` and the model cites them inline. The response then lists **only the sources
it actually cited** (parsed from the inline `[n]` tags), each rendered as
`source — professor — course — date (source_file)`. Example from Q1:

```
… exams are "brutal" [3], and partial credit is rare [1] …
Sources:
  [1] RateMyProfessors — Daniel Mercer — CSE 214 — 2023-12-02  (ratemyprofessors_cse.json)
  [3] RateMyProfessors — Daniel Mercer — CSE 214 — 2023-09-14  (ratemyprofessors_cse.json)
```

---

## Evaluation Report

All five planning-doc questions were run end-to-end through the live system
(placeholder professor names resolved to the fictional professors in the corpus).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Grading difficulty of CSE 214 with Mercer? | Tough/strict grader, heavy workload, but fair if you keep up | Tough grading (difficulty 4.0–4.6), "brutal" exams, rare partial credit, but "fair but demanding" and you learn DS; flags the hard-but-worth-it disagreement. Cites [1–5], all Mercer/CSE 214. | Relevant | **Accurate** |
| 2 | Is Rahman good for CSE 114 beginners? | Beginner-friendly, clear, generous curve; recommend | Yes — explains slowly with examples, engaging live coding, generous curve. Cites two Rahman/CSE 114 reviews [1][2] + one Reddit comment [3]. | Partially relevant (see note) | **Accurate** |
| 3 | Osei's lecture style & exams in CSE 220? | Reads slides / flat delivery but good slides; cumulative, conceptual, assembly-heavy exams | Monotone, reads slides, but slides are good reference; exams cumulative & conceptual, require understanding assembly not memorization. Cites [1–3], all Osei/CSE 220. | Relevant | **Accurate** |
| 4 | Mercer vs Rahman for CSE 214 — who's preferred & why? | Comparative answer grounded in reviews for both | Balanced "depends on priorities" — Rahman clearer/nicer curve, Mercer harder but deeper learning. Reasonable, but rests on thin, asymmetric, and partly untagged evidence (see Failure Case). | Partially relevant | **Partially accurate** |
| 5 | Does attendance/participation matter in Whitfield's CSE 101? | Yes — clicker/participation points, attendance effectively required | Yes — in-class clicker questions count, so you must show up. Cites [1–3], all CSE 101. | Relevant | **Accurate** |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

*Note on Q2:* the answer is accurate and its core claims come from the two CSE 114
Rahman reviews, but retrieval also pulled a **CSE 214** Reddit comment ("Rahman's
curve is nicer") into a **CSE 114** question. The claim is still about Rahman and
broadly true, but it's a cross-course leak — an early symptom of the same root cause
as the Q4 failure below.

---

## Failure Case Analysis

**Question that failed:** Q4 — *"Between Professor Mercer and Professor Rahman, who do
students prefer for CSE 214, and why?"* (judged **partially accurate**).

**What the system returned:** A confident, balanced "it depends on your priorities"
answer — Rahman for clarity and a nicer curve, Mercer for deeper learning despite
difficulty. It reads well, but it **overstates the strength of the evidence**. The
preference signal rests largely on a *single* Reddit comment ("take Rahman… her curve
is nicer") plus Rahman's *one* CSE 214 review, against five Mercer data points. The
answer never flags that the evidence is thin and lopsided, and a user would reasonably
assume a robust consensus exists when it doesn't.

**Root cause (tied to a specific pipeline stage):** The **ingestion stage** leaves
every Reddit comment with `professor = None`, because a free-text comment ("I had Mercer
and Rahman explains it better") names professors in prose but isn't reliably tagged to
one in metadata. This has two concrete downstream effects:

1. The strongest preference evidence for Q4 lives in those untagged Reddit comments, so
   the system can't weight or verify it by professor — it leans on prose it can't attribute.
2. The **planned metadata pre-filter** *silently drops that evidence*. Demonstrated
   directly: for *"What do students say about Professor Rahman?"*, the Reddit comment
   praising Rahman is the 4th-best semantic hit (distance 0.458) **with no filter** —
   but adding `where={"professor": "Aisha Rahman"}` removes it, because `None` doesn't
   match `"Aisha Rahman"`. The filter that's supposed to *improve* precision instead
   throws away real, on-topic signal. (This same gap is why Q2 leaked a CSE 214 comment.)

This is an ingestion/metadata failure, not a retrieval-tuning or generation problem —
no amount of k-tuning or prompt-wording fixes a chunk that was never tagged with the
professor it discusses.

**What you would change to fix it:** Add a professor-resolution step to ingestion that
tags Reddit comments by matching mentioned names against the **faculty directory**
(source #8) and the thread's course context — populating `professor` (and a
`professor_mentions` list when a comment names several). Then make the metadata filter
*soft* rather than hard: prefer exact-professor chunks but fall back to course-scoped
semantic hits so a `None`-tagged-but-relevant comment isn't discarded. Finally, surface
an evidence-count signal (e.g. "based on 5 reviews for Mercer, 2 for Rahman") so thin or
asymmetric comparisons are flagged in the answer instead of presented as settled.

---

## Spec Reflection

**One way the spec helped you during implementation:** The Chunking Strategy section
of `planning.md` committed up front to *record-level* chunking with metadata headers
and no overlap, with an explicit rationale (avoid cross-professor contamination). That
decision propagated cleanly through every later stage: the chunk schema defined the
ChromaDB metadata fields, which defined the retrieval filter, which defined the
citation labels in generation. Because the boundary rule was decided before any code,
I never had to retrofit attribution — every chunk was single-professor and carried its
provenance from the moment it was created, and the Milestone 3 checkpoint (5 readable,
self-contained chunks) passed on the first run.

**One way your implementation diverged from the spec, and why:** The plan specified
**top-k = 8** for retrieval, reasoning that synthesizing student opinion needs many
voices. In practice I run the **checkpoint at k = 5** and generation at k = 8 *with a
0.55 distance gate* — and the gate matters more than k. On this small corpus, k = 8
without filtering pulled in wrong-professor chunks at distance > 0.5 (e.g. Q3 ranks 4–5
were Mercer/Rahman reviews on an Osei question). The spec treated k as the main lever for
coverage; implementation showed the *distance threshold* is the real lever for precision,
and that it doubles as the refusal mechanism (out-of-domain queries clear the gate with
zero chunks and refuse before the LLM is ever called). I kept k = 8 for recall but leaned
on the threshold the plan under-weighted.

---

## AI Usage

**Instance 1 — Embedding + retrieval code (Milestone 4)**

- *What I gave the AI:* the Retrieval Approach section of `planning.md` (model
  `all-MiniLM-L6-v2`, ChromaDB, top-k, metadata pre-filter) and the pipeline diagram, and
  asked it to implement the embedding step and a `retrieve()` function storing source
  metadata.
- *What it produced:* a working `src/embed.py` that embedded chunks and queried ChromaDB,
  but it created the collection with ChromaDB's **default L2 distance** and used Chroma's
  built-in embedding function.
- *What I changed or overrode:* I forced the collection to **cosine** space
  (`hnsw:space="cosine"`) and `normalize_embeddings=True` so distances matched the
  checkpoint's 0.5 threshold (L2 distances aren't comparable to that scale), and I embed
  chunks with my own `SentenceTransformer` instance rather than Chroma's embedder so the
  query and corpus provably use the same model. I also added a `_clean_meta` step because
  ChromaDB rejects `None`/non-scalar metadata, which the first version crashed on.

**Instance 2 — Grounding and citation behavior (Milestone 5)**

- *What I gave the AI:* the Anticipated Challenges section of `planning.md` (report
  disagreement; say "not enough feedback" instead of fabricating) and asked it to write a
  grounded generation function over Groq that cites sources.
- *What it produced:* a generator with a reasonable system prompt and a `Sources:` list,
  but it (a) relied **only** on the prompt to refuse and (b) listed **all** retrieved
  chunks as sources regardless of whether the answer used them.
- *What I changed or overrode:* I added a **structural refusal** — a distance gate that
  short-circuits to a canned refusal *without calling the LLM* when no chunk clears 0.55,
  so out-of-domain queries can't be talked into an answer. And I made attribution honest by
  parsing the inline `[n]` tags out of the response and listing **only the sources the
  answer actually cited** (verified: Q1 cited [1–5] and the list dropped the 3 unused
  chunks). The prompt-only version both over-trusted the model to refuse and over-claimed
  its sources.
