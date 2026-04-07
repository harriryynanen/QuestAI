# QuestAI

Lightweight Streamlit demo for a fictional business banking advisory Q&A assistant.

## What It Does Now

The current app supports three constrained paths:

- `retrieval`: markdown document retrieval with OpenAI-based synthesis grounded only in retrieved chunks
- `structured`: deterministic CSV querying for customer facts, filters, comparisons, counts, and existence checks
- `combined`: semantic planning plus evidence-based synthesis using retrieved document evidence and deterministic structured evidence

The app is intentionally scoped as decision support only. It does not make approvals, risk decisions, or real-world recommendations.

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables in `.env` if you want OpenAI-enabled routing and synthesis:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
ENABLE_LLM_FOR_RETRIEVAL=true
```

4. Start the app:

```bash
streamlit run app/main.py
```

## Data Included In The Repo

The repository already includes demo source files:

- markdown documents in `data/docs/`
- one CSV dataset in `data/structured/`

Supported file handling in the current version:

- document discovery: `.md`, `.txt`, `.pdf`
- retrieval content path: `.md`, `.txt`, `.pdf` for text-based content
- structured dataset loading: `.csv`

## Current Behavior

- Questions are first interpreted through an LLM-based semantic planning layer.
- If semantic planning is unavailable or unclear, a deterministic rule-based router remains as fallback.
- Retrieval answers are synthesized only from retrieved document chunks selected by the app.
- Structured answers are executed deterministically from CSV data after semantic planning.
- Combined answers use a controlled evidence flow:
  - retrieve document evidence
  - assemble structured customer evidence deterministically
  - synthesize a cautious answer from those evidence packs only
- Source references shown in the UI come from application metadata, not model-invented citations.
- If OpenAI is unavailable, the app still runs with deterministic fallbacks where possible.

## OpenAI Usage

OpenAI is used only for:

- semantic planning / routing
- retrieval answer synthesis
- combined evidence synthesis

OpenAI is not used to directly answer structured CSV questions from scratch. CSV execution remains deterministic for transparency and control.

For hosted Streamlit use, runtime settings can come from:

1. `st.secrets`
2. environment variables
3. safe defaults where applicable

## Project Structure

- `app/llm/`: OpenAI client and prompt builders
- `app/retrieval/`: markdown loading, chunking, and deterministic retrieval
- `app/structured/`: CSV loading, semantic planning entry point, and deterministic query execution
- `app/services/`: orchestration and fallback routing
- `app/ui/`: Streamlit presentation layer
- `app/config.py`: paths and runtime settings
- `data/`: constrained demo sources used by the app

## Current Limitations

- no embeddings or vector database
- no PDF text parsing
- no unrestricted agent behavior
- no production decision engine
- combined reasoning is still an MVP and intentionally cautious

## Next Likely Extensions

- improve retrieval ranking and lightweight evaluation coverage
- strengthen combined evidence reasoning and confidence labelling
- add broader structured alias coverage while keeping execution deterministic
