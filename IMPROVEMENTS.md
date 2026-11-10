# AI-Cademic: Future Improvements

A running list of planned work. Each item notes its status. "App" means the latest Streamlit app, `UI_CADEMIC_v4.4_Alpha.py`.

Status key: **todo**, **in progress**, **on hold**, **done**

---

## 1. Retrieval quality

### 1.1 LLM-generated search query before embedding (query rewriting / HyDE): **todo**
Right now the app embeds the user's raw question and searches with that (`process_text` → `get_embedding(input_text)`). Questions are often short or worded differently from the papers, so their embeddings match the text poorly.

Plan: add an LLM step before retrieval that writes a better search string, then embed that instead.
- **Option A, query rewrite.** Turn the question into a focused, keyword-rich search query using the papers' own vocabulary. For example, "why is multimodal hard?" becomes "challenges of aligning and fusing text and image representations in multimodal models".
- **Option B, HyDE (Hypothetical Document Embeddings).** Have the LLM write a short passage that *would* answer the question, in the style of an academic paper, and embed that. Passages sit closer to passages in embedding space than questions do.
- **Option C, multi-query.** Generate 2–4 variant queries, search with each, and merge and dedupe the matches. This helps broad questions that touch several papers.
- Keep the user's original question for the final answer prompt. The rewritten string is for retrieval only.
- Show the generated search string in a debug expander so it can be checked.
- Evaluate against the raw-question baseline using the question set in 3.2, and pick A, B, or C (or A+C) based on the results.
- Cost and latency: one extra small LLM call per question. Use a cheap, fast model for this step.

### 1.2 Unlimited-OCR ingestion (Baidu Unlimited-OCR-2): **on hold**
Replace the current PDF → image → GPT-description pipeline with real document parsing. The model is 3B parameters and runs locally on the RTX 5070 Ti (12 GB).
- Done: `ingest_unlimited_ocr.py` (stages: ocr → chunk → embed), `requirements-ocr.txt`, and `.gitignore` entries. The chunk and embed stages were checked using the PDF text-layer baseline (`--backend pdftext`): 370 chunks, embed run as a dry run.
- Next, on hold since 2026-11-10: set up the Python 3.10 GPU venv and OCR one paper (`attention.pdf`).
- Then:
  - Check the raw output, including whether the tag cleanup in `clean_ocr_text` matches what the model actually emits.
  - Run all the papers and embed them into the `uocr` namespace and `dicembed_uocr.json`.
  - Compare against the current index using 3.2.
- Then point the app at the new data (see 1.3).
- Open question: whether OCR output describes figures and charts well enough, or whether the GPT image descriptions should be kept and merged in.

### 1.3 Switch the app to the new index: **todo** (after 1.2)
- Make the namespace and dictionary path settings (`st.secrets` or a constant) instead of hard-coding them.
- The new IDs are strings like `uocr-attention-p0003-c01`. The app already looks them up by string, so no lookup change is needed.
- Vectors now store their text in Pinecone metadata, so the app could use `include_metadata=True` and drop the 1.7 MB JSON file entirely.

### 1.4 Retrieval tuning: **todo**
- The app currently fetches the top 5 matches but uses only 3, with no similarity threshold. Make k configurable, and drop matches below a score cutoff so unrelated context isn't passed to the LLM.
- Consider reranking (a cross-encoder or an LLM reranker) over the top ~20 matches.
- Consider hybrid search (keyword/BM25 plus vectors) for exact terms, model names, and acronyms.

### 1.5 Citations in answers: **todo** (after 1.2)
Chunks from the OCR pipeline carry file, page, and section. Show the sources under each answer (for example, "attention.pdf, p.3, Encoder") and ask the LLM to cite them.

---

## 2. App and code quality

### 2.1 Upgrade the answer model: **todo**
The app calls `gpt-3.5-turbo` with `max_tokens=300` through a raw `requests.post`. Move to a current model through the `openai` SDK client the app already creates, and raise the token limit so answers aren't cut off.

### 2.2 Prompt structure: **todo**
The instructions and context are currently glued onto the user's question in a single user message (`question + instructions + context`). Split them into a system message (instructions) and a user message (question and labelled context blocks).

### 2.3 Error handling: **todo**
- Handle empty input. Submitting a blank question currently still runs a search.
- `get_llm_answer` returns a `{"error": ...}` dict on failure, and the app prints that dict as the answer. Show a friendly `st.error` instead.
- Handle Pinecone and embedding failures as well.

### 2.4 Conversation memory: **todo**
`st.session_state.conversation` is initialized but never used. Move to `st.chat_input` and `st.chat_message` with history. Follow-up questions also need the rewrite step from 1.1 to work, so it can turn "what about its limitations?" into a standalone query.

### 2.5 Dependency cleanup: **todo**
- The app imports many libraries it doesn't use (torch, transformers, fitz, PyPDF2, pdfplumber, langchain retrievers, and others), and imports `os` twice.
- `from pydantic import FieldSerializationInfo` (Pydantic v2) sits alongside `langchain_core.pydantic_v1`.
- Trim the imports and `requirements.txt` so the Streamlit Cloud deploy is faster and less fragile.

### 2.6 Security: rotate the old Pinecone key: **todo**
A Pinecone API key appears in a comment in the app (lines 53–54) and is in git history. Rotate it if it was ever live, and delete the comment.

### 2.7 Repo tidy-up: **todo**
- Older app versions (`UI_CADEMIC_v3.py`, `v4.py`, `v4.3_Alpha.py`) could move to an `archive/` folder.
- The README should point to the current app file and describe the ingestion pipeline.

---

## 3. Evaluation

### 3.1 Retrieval debug view: **todo**
Add an expander that shows the search string, the retrieved chunk IDs, their scores, and sources. This is needed to judge 1.1–1.4.

### 3.2 Question set for comparisons: **todo**
Write 15–20 questions with known answers spread across the papers. Use them to compare:
- the old index against the OCR index (1.2)
- the raw-question embedding against rewritten queries (1.1)
- retrieval settings (1.4)

Record retrieval hits (whether the correct chunk was in the top k) as well as answer quality.

---

## Suggested order
1. 2.3, 2.5, 2.6: quick fixes that cut risk.
2. 3.1 and 3.2: you can't measure the rest without them.
3. 1.1: query rewriting, which works on the current index right away.
4. 1.2 → 1.3 → 1.5: the OCR pipeline, switching the app over, and citations.
5. 2.1, 2.2, 2.4, 1.4: model, prompt, chat, and retrieval tuning.
