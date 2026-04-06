# QuestAI

Lightweight Streamlit demo for a fictional business banking advisory Q&A assistant.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app/main.py
```

## Current scope

This stage supports two grounded paths:

- markdown retrieval for document-oriented questions
- deterministic CSV querying for structured business questions

Combined reasoning across both sources is not implemented yet. OpenAI-based synthesis is the next step, not part of the current demo slice.

## Included demo data

The repository now includes demo markdown documents in `data/docs/` and a demo CSV in `data/structured/`.

Supported document discovery types:
- `.md`
- `.txt`
- `.pdf`

Current retrieval content path:
- `.md` only

Supported structured file type:
- `.csv`

## Current behavior

- Retrieval questions use markdown-only chunk retrieval with deterministic keyword matching.
- Structured questions use deterministic pandas-based querying for specific customer facts, simple comparisons, and simple filters.
- Combined routes return a controlled limitation message instead of faking combined reasoning.
- Answers show source references and should be treated as preliminary demo output, not decisions.

## Structure

- `app/retrieval/` contains markdown loading, chunking, and retrieval logic.
- `app/structured/` contains CSV loading and structured query logic.
- `app/services/router.py` keeps routing separate and explainable.
- `app/config.py` centralizes paths and lightweight settings.
- `data/` holds the constrained internal sources used by the demo.
