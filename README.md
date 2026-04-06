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

This stage validates file discovery and loading readiness for manually added source files. The question flow still uses simple routing and placeholder answer generation.

## Add your files

Place synthetic internal documents in `data/docs/`.

Supported document types for this stage:
- `.md`
- `.txt`
- `.pdf`

Place one synthetic CSV dataset in `data/structured/`.

Supported structured file type for this stage:
- `.csv`

No demo data is included. Add the files manually when you are ready to test discovery and loading.
