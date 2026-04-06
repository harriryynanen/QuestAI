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

This stage adds the first real markdown retrieval foundation. The app can now load repository markdown files, split them into lightweight chunks, and retrieve relevant passages with simple keyword overlap. CSV loading is available, but structured question answering is not implemented yet.

## Included demo data

The repository now includes demo markdown documents in `data/docs/` and a demo CSV in `data/structured/`.

Supported document discovery types:
- `.md`
- `.txt`
- `.pdf`

Current retrieval content path:
- `.md` only

Supported structured file type for loading:
- `.csv`

## Current behavior

- Retrieval questions use markdown-only chunk retrieval with deterministic keyword matching.
- Structured and combined routes still return placeholder answers.
- Answers are still partially placeholder-based and should be treated as preliminary demo output, not decisions.
