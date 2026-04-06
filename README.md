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

- markdown retrieval for document-oriented questions, with OpenAI used only for retrieval answer synthesis
- deterministic CSV querying for structured business questions

Combined reasoning across both sources is not implemented yet. OpenAI-based synthesis is now used only for retrieval answers, while structured CSV answers remain deterministic.

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

- Retrieval questions use markdown-only chunk retrieval with deterministic keyword matching, then optionally use OpenAI to synthesize a concise grounded answer from the retrieved chunks only.
- Structured questions use deterministic pandas-based querying for specific customer facts, simple comparisons, and simple filters.
- Combined routes return a controlled limitation message instead of faking combined reasoning.
- Answers show source references and should be treated as preliminary demo output, not decisions.
- The default retrieval synthesis model is `gpt-5.4-mini`, configurable through environment variables.

If `OPENAI_API_KEY` is missing, the app still runs and retrieval answers fall back to a deterministic chunk-based summary.

## OpenAI Setup

For local development, copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY`
- optionally `OPENAI_MODEL`

The app loads these values at runtime. If no API key is present, only the retrieval synthesis step falls back; structured CSV answers remain deterministic.

## Structure

- `app/retrieval/` contains markdown loading, chunking, and retrieval logic.
- `app/llm/` contains OpenAI client usage and retrieval prompt construction.
- `app/structured/` contains CSV loading and structured query logic.
- `app/services/router.py` keeps routing separate and explainable.
- `app/config.py` centralizes paths and lightweight settings.
- `data/` holds the constrained internal sources used by the demo.

This staged design keeps the deterministic retrieval and structured foundations visible, while making it straightforward to extend later with better ranking, prompt iteration, and combined reasoning.
