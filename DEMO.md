# Demo Runbook (3–5 min)

A shot list for the recorded demo. The required moments are tagged **[REQ]**.
Use the Gradio UI (`python app_web.py`) so citations render cleanly on screen;
the CLI (`python app.py`) works too.

Start screen recording, then:

## 0. Intro (~20s)
- One sentence: "A RAG system that answers questions about SBU CSE professors
  using only student reviews, with citations."
- Note honestly: "It's running on a synthetic sample corpus with fictional
  professor names; the loaders target the real RateMyProfessors/Reddit/catalog
  formats so real data drops in unchanged."

## 1. Query that works well — **[REQ: works well] [REQ: citations visible]**
- Ask: **"What's the grading like in CSE 214 with Professor Mercer?"**
- Point at the inline `[1]…[5]` tags and the **Sources** list. Say: "Every claim
  is cited, and the sources are all Mercer / CSE 214 reviews — retrieval and
  generation both did their job."

## 2. Second query, different professor — **[REQ: 3+ queries with citations]**
- Ask: **"Is Professor Rahman a good choice for CSE 114 if I'm new to programming?"**
- Note the citations are Rahman / CSE 114 reviews.

## 3. Third query showing disagreement handling — **[REQ: 3+ queries]**
- Ask: **"How do students describe Professor Osei's lectures and exams in CSE 220?"**
- Point out it synthesizes "reads slides / flat delivery BUT slides are good;
  exams are conceptual" across a Reddit post and two RMP reviews.

## 4. A query where the system struggles — **[REQ: failure, narrated]**
Pick ONE of these (both are in the README Failure Case):

- **Refusal path (cleanest to show):** Ask **"Which dining hall has the best food?"**
  → the system declines instead of inventing an answer. Narrate: "Out-of-domain
  queries land above the 0.55 distance gate, so it refuses *before* the LLM is even
  called — it can't hallucinate from context that isn't there."

- **Substantive failure (matches README):** Ask **"Between Mercer and Rahman, who
  do students prefer for CSE 214, and why?"** Narrate: "This looks confident, but
  it's only **partially accurate**. The preference rests on one Reddit comment and
  Rahman's single 214 review vs. five Mercer data points. Root cause is in
  ingestion: Reddit comments are stored with `professor=None`, so that evidence
  can't be attributed — and the professor metadata filter actually *drops* it.
  That's a pipeline bug, not a tuning problem."

> Recommended: show the refusal live (fast, visual), then verbally summarize the
> Q4 metadata failure while showing the README Failure Case section.

## 5. Evaluation report walkthrough — **[REQ: eval report]**
- Open `README.md` → **Evaluation Report** table. Walk the 5 rows: 3 Accurate,
  Q2 Accurate-with-a-cross-course-note, Q4 Partially accurate.
- Open **Failure Case Analysis** and read the root cause (Reddit `professor=None`
  → metadata filter drops real evidence) and the fix.

## 6. Wrap (~10s)
- "Retrieval is solid, generation is grounded and cites sources, and the main
  limitation is professor attribution on free-text Reddit comments — documented
  in the README." Stop recording.

---

### Pre-flight checklist
- [ ] `.env` has a valid `GROQ_API_KEY`
- [ ] `pip install -r requirements.txt` done
- [ ] `python checkpoint_m4.py` ran once (builds the index; first run downloads the
      ~80MB embedding model so it's cached before recording)
- [ ] `python app_web.py` opens and answers a test question
