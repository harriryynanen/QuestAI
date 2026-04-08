# QuestAI

QuestAI is a lightweight AI-powered Business Q&A Assistant built for a recruitment assignment. It is designed as a constrained, explainable demo rather than a production system.

The application uses a chat-style Streamlit interface to answer questions in a fictional business banking advisory scenario using two data types:

- document content from `.md`, `.txt`, and `.pdf` files
- structured demo data from named `.csv` datasets

The solution is intentionally scoped around decision support, not automated decision-making. It can help an advisor explore policy guidance, inspect customer facts, and form cautious preliminary views from mixed evidence, but it does not make approvals, risk decisions, or production-grade financial recommendations.

QuestAI therefore supports grounded answers over documents plus multiple structured demo datasets, but structured extensibility is currently controlled rather than fully automatic.

## Why This Was Built

The assignment is a good fit for a common real-world AI engineering problem: combining unstructured documents, structured records, and LLM capabilities in a way that stays grounded, reviewable, and reasonably safe.

I built QuestAI to demonstrate a few practical engineering choices:

- use LLMs where interpretation and synthesis add value
- keep deterministic execution where exact answers are required
- make provenance visible in the UI
- keep the system small enough to explain end-to-end
- preserve graceful fallback behavior when AI capabilities are unavailable or uncertain

This is why the project is deliberately not a generic chatbot. It is a scoped assistant with explicit boundaries.

## What The Application Does

QuestAI supports three main question types:

1. Document-oriented questions  
   Example: “What does the policy say about tax arrears?”

2. Structured data questions  
   Example: “What is Harbor Foods Demo Oy’s equity ratio?” or “Which customers have tax arrears?”

3. Combined questions  
   Example: “Based on the policy and customer data, does this customer appear broadly aligned with FlexLine Demo?”

The UI is chat-like on purpose. For a demo, this makes the application easy to test interactively:

- the evaluator can ask multiple questions in sequence
- follow-up questions can reuse limited recent context
- answers remain the focal point
- technical transparency stays available through source chips, support levels, and expandable answer details

## Demo Domain And Sample Data

The project uses a fictional business banking advisory scenario. All included data is synthetic.

What the demo data contains:

- fictional product guidance
- fictional policy guidance
- fictional advisor notes
- fictional customer portfolio data in CSV form
- fictional advisory case pipeline data in CSV form

What it intentionally does not contain:

- real institutions
- real customers
- real products
- real decisions
- sensitive personal or financial production data

This matters for evaluation: the app is demonstrating architecture, reasoning boundaries, and explainability patterns, not domain accuracy for real financial operations.

## How The Solution Works

At a high level, the application answers a question through four steps:

1. Interpret the question
2. Choose the most suitable path
3. Gather evidence from the right source(s)
4. Return a grounded answer with explicit limitations

The implementation keeps document retrieval and structured execution separate on purpose.

- Document questions benefit from semantic interpretation and evidence synthesis.
- Structured CSV questions need deterministic execution so the answer remains precise and auditable.
- Combined questions use both sources, but still keep structured evidence generation deterministic.

This split is one of the main design decisions in the project.

## Architecture Overview

The repository is organized into small modules with clear responsibilities:

- `app/ui/`  
  Streamlit presentation layer, chat layout, answer cards, prompt chips, and provenance display.

- `app/services/`  
  Answer orchestration and routing logic. This layer coordinates the retrieval, structured, and combined flows.

- `app/llm/`  
  OpenAI client and prompt construction for semantic planning and answer synthesis.

- `app/retrieval/`  
  Document loading, chunking, local ranking, and retrieval helpers.

- `app/structured/`  
  CSV loading, semantic plan interpretation, and deterministic query execution.

- `app/config.py`  
  Runtime configuration and local/hosted settings.

- `app/models.py`  
  Shared data models used across the application.

This structure keeps the data flow visible:

- UI handles interaction
- services orchestrate
- retrieval handles documents
- structured handles CSV execution
- LLM code stays focused on planning and synthesis

## Data Ingestion And Retrieval Logic

### Supported document types

The app currently supports:

- `.md`
- `.txt`
- `.pdf`

These are loaded locally from `data/docs/`.

### How chunking works

Larger documents are broken into smaller text chunks before retrieval. In plain English, this means the app does not search an entire document as one large blob. Instead, it splits content into manageable pieces, usually around headings and paragraph boundaries, so that retrieval can point to specific sections rather than a whole file.

This makes the evidence more useful for two reasons:

- retrieval can surface the most relevant section instead of a large document
- answer synthesis can stay grounded in a small evidence set

### How retrieval works

Retrieval is intentionally lightweight and local. There is no vector database or external retrieval infrastructure.

The current retrieval pipeline:

1. load documents from disk
2. chunk them into smaller sections
3. normalize and tokenize the user query
4. score chunks with a lightweight ranking layer
5. keep the top matching chunks
6. synthesize an answer only from those retrieved chunks

The ranking layer is stronger than raw keyword overlap but still inspectable. It uses a lightweight combination of:

- token normalization
- weighted term overlap
- heading-aware scoring
- phrase match bonuses
- BM25-like local ranking
- conservative product query expansion for generic product questions

This was a deliberate choice. For a recruitment assignment, it improves retrieval quality without hiding the system behind a heavier black-box retrieval stack.

### Retrieval explainability

Retrieved chunks carry application-generated metadata such as:

- file name
- section heading
- chunk id
- match summary

Those match summaries are generated by the application logic, not invented by the model. The UI can therefore explain why a chunk was retrieved in understandable terms such as phrase matches, heading matches, or policy/product term matches.

## Structured Data Logic

Structured data is loaded from `.csv` files under `data/structured/`.

The current demo supports two named structured datasets:

- `customer_portfolio`
- `advisory_case_pipeline`

Structured answers stay deterministic by design. The model does not directly answer CSV questions from free-form memory. Instead:

1. the planner interprets the user’s intent
2. the planner uses local structured schema metadata to infer the target dataset and internal field name when possible
3. the structured engine maps that plan to a supported deterministic operation
4. the answer is executed directly against the selected DataFrame

This is a deliberate middle ground. Semantic planning is schema-aware, but execution still depends on known internal field names and explicit query logic rather than arbitrary dataframe generation.

Supported structured operations include:

- fact lookup
- filtering
- comparisons
- counts
- existence checks
- listing customers

This separation matters because structured questions often require exactness. A deterministic path keeps the behavior easier to validate and less likely to drift.

### Current structured-data limitation

QuestAI does not yet support arbitrary new structured datasets with zero code changes. Adding a genuinely new structured dataset still requires a small schema and execution extension so the planner knows the dataset/field surface and the deterministic layer knows how to execute it safely. That is intentional for this demo: reliability and inspectability are prioritized over fully automatic structured extensibility.

## Routing, Planning, And Fallback Logic

Routing is intentionally layered.

### Primary path

QuestAI first tries LLM-based semantic planning. This helps interpret natural language questions, including mixed or slightly ambiguous phrasing.

### Fallback path

If the semantic planner is unavailable or not confident enough, the app falls back to rule-based routing. This keeps the system usable even when OpenAI is missing or disabled.

### Why this matters

This design balances flexibility with control:

- LLM planning improves usability
- deterministic fallback preserves reliability
- uncertain cases do not silently turn into confident answers

### Conversational follow-ups

The chat UI keeps lightweight recent-session context. This is used narrowly, not as general memory.

Examples of supported follow-up behavior:

- count -> “Name those customers”
- filter result -> “Who are they?”
- grouped combined question -> “Which companies are not”

The implementation intentionally reuses only recent, clearly established scope. If the reference is too ambiguous, the app should fail cautiously rather than guess.

## Combined Questions

Combined questions are the most interesting path in the project because they mix unstructured and structured evidence.

The combined flow works like this:

1. identify that both documents and customer data are needed
2. retrieve the most relevant document evidence
3. build structured customer evidence deterministically
4. send only that evidence pack into LLM synthesis
5. return a cautious answer with visible sources and limitations

This is still decision support only. The app does not state final eligibility or approval decisions. It uses deliberately cautious language such as:

- broadly aligned based on available evidence
- caution / mixed signals
- not enough information

That wording is intentional and matches the scope of the demo.

## UI And User Interaction Model

The Streamlit interface is chat-style rather than form-style.

That was a deliberate product choice for the assignment:

- it feels more natural to evaluate interactively
- it allows follow-up questions in one session
- it shows how grounded answers behave over multiple turns

The UI preserves explainability without making technical detail dominate the screen:

- the main answer stays visually prominent
- source references are visible through compact citation markers
- route / support / synthesis badges remain visible
- `Why this answer` expands into full technical detail

Inside the expander, the evaluator can inspect:

- full source list
- retrieved evidence
- structured matches
- routing and planning detail
- limitations

Source references are shown from application metadata, not invented by the model. This is important: the app does not ask the model to hallucinate citations.

## Fallback Behavior

Fallback behavior is explicit at several stages.

### If OpenAI is unavailable

- semantic planning falls back to rule-based routing
- retrieval synthesis falls back to a deterministic summary of retrieved chunks
- combined synthesis falls back to a deterministic combined evidence summary where possible

### If routing is uncertain

- the app returns a cautious `unknown` route response
- it does not pretend to understand the question confidently

### If retrieval evidence is weak

- the app returns a low-support answer or a retrieval limitation message
- a missing retrieval match is treated as an evidence gap, not proof that no guidance exists

### If combined evidence is incomplete

- support level is capped
- limitations explicitly mention missing or incomplete information
- the app avoids final-decision wording

## Testing And Validation

The repository includes automated tests covering the main flows:

- routing and planning
- structured queries
- retrieval flow
- combined flow
- text and PDF ingestion
- retrieval scoring behavior

The test strategy is intentionally practical rather than exhaustive. The goal is to validate the constrained design choices and main edge cases of the demo.

This includes checks for:

- grounded source handling
- deterministic structured behavior
- safe fallback behavior
- conversational follow-up handling
- grouped combined assessment behavior

## Security And Data Safety

This project is not a production deployment, but the repository still follows basic security hygiene.

- no secrets are hardcoded in source files
- OpenAI credentials are expected via environment variables or Streamlit secrets
- `.env` and Streamlit secrets should stay out of version control
- the app uses only synthetic local demo data
- no real customer or institution data is included

The app also avoids exposing internal decisioning authority:

- it is not an approval engine
- it is not a credit decisioning engine
- it does not implement automated financial decision-making

## Key Technical Decisions And Tradeoffs

### 1. Separate retrieval from structured execution

Why:

- document questions and CSV questions have different reliability needs
- structured questions benefit from deterministic execution
- mixed questions still need a constrained evidence path

Tradeoff:

- the architecture is slightly more complex than a single chatbot flow
- but much easier to reason about and validate

### 2. Use LLMs for planning and synthesis, not for everything

Why:

- semantic planning helps interpret natural language
- synthesis improves readability of grounded evidence
- deterministic logic remains in control for exact data operations

Tradeoff:

- some user phrasing still needs heuristic support
- the system is intentionally narrower than a general agent

### 3. Keep retrieval local and lightweight

Why:

- better fit for a demo and recruitment assignment
- easier to explain than a heavier retrieval stack
- no infrastructure or vector database required

Tradeoff:

- retrieval quality is improved but still not production-grade
- ranking is transparent, but less semantically rich than embedding-based search

### 4. Keep the UI explainable

Why:

- the evaluator should be able to inspect how the answer was formed
- answer quality alone is not enough in this type of assignment

Tradeoff:

- even a cleaned-up UI still exposes technical detail
- that is intentional because explainability is part of the value here

### 5. Avoid agentic or multi-agent orchestration in this project

Why:

- the assignment benefits more from a clear, inspectable request flow than from autonomous tool orchestration
- deterministic structured execution and constrained retrieval are easier to validate than agent-style delegated reasoning
- a single-path orchestration layer makes fallback behavior, provenance, and failure analysis easier to explain to a reviewer

Tradeoff:

- the system is narrower than an agentic assistant and handles fewer open-ended workflows
- but the implementation remains more credible for this scope, because the reasoning boundaries, source flow, and failure modes stay visible

## What Is Implemented Now

Implemented in the current version:

- chat-style Streamlit UI
- `.md`, `.txt`, and `.pdf` document ingestion
- local chunk-based retrieval
- lightweight explainable ranking
- deterministic CSV querying
- LLM-based semantic planning
- LLM-based retrieval synthesis
- LLM-based combined evidence synthesis
- cautious grouped combined assessments
- visible source grounding and expandable technical details

## What Is Intentionally Out Of Scope

Deliberately out of scope for this assignment:

- production decisioning
- real customer data handling
- user authentication
- persistent chat history
- vector databases and retrieval infrastructure
- agent loops or autonomous workflows
- production observability and deployment automation
- policy enforcement beyond demo-level guardrails

## Failure Modes And Limitations

There are several places where the current system can still be wrong or incomplete.

- retrieval may miss relevant evidence if wording differs too much from the query
- PDF extraction quality depends on readable embedded text
- semantic planning can still misinterpret ambiguous questions
- combined answers depend on both document evidence and structured evidence being available
- grouped follow-up handling is intentionally narrow and not full conversational memory
- support levels are heuristic labels, not calibrated probabilities
- the product/policy logic is synthetic and intentionally simplified

Most importantly, the app should not be interpreted as making correct real-world financial judgments. It is a controlled demo for grounded AI-assisted decision support patterns.

## Extensibility And Future Improvements

If this were taken further, the next steps would be:

1. Improve retrieval quality further  
   Add lightweight evaluation datasets, better query expansion, and stronger chunk ranking before considering embeddings.

2. Evolve the structured-data layer carefully  
   Add better schema introspection or metadata generation, move more field descriptions into configuration, introduce a small generic structured operation registry, and strengthen validation and evaluation when new datasets are introduced.

3. Harden combined assessment behavior  
   Improve evidence assembly, bucket logic, and support-level calibration.

4. Add production readiness layers  
   Authentication, request logging, audit trails, monitoring, and safer configuration management.

5. Improve document handling  
   Better PDF extraction, richer document metadata, and more robust heading/section parsing.

6. Add evaluation and observability  
   Track retrieval quality, answer quality, failure cases, and follow-up handling over time.

## Local Run Instructions

### Requirements

- Python 3.11+ recommended
- dependencies from `requirements.txt`

### Setup

```bash
pip install -r requirements.txt
```

Set credentials through environment variables or Streamlit secrets if you want OpenAI-enabled planning and synthesis:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4-mini
ENABLE_LLM_FOR_RETRIEVAL=true
```

You can also use `.streamlit/secrets.toml` based on `.streamlit/secrets.toml.example`.

### Run

```bash
streamlit run app/main.py
```

### Run The Lightweight API

```bash
uvicorn app.api:app --reload
```

Available demo endpoints:

- `GET /health`
- `POST /answer`

### Tests

```bash
python -m pytest -q
```

## Hosted Demo Note

If a hosted Streamlit deployment is available, it uses the same constrained architecture as the local version. Runtime settings can be supplied through Streamlit secrets, environment variables, or safe defaults where appropriate.

## Closing Note

QuestAI is meant to demonstrate pragmatic AI engineering choices:

- use LLMs selectively
- keep exact paths deterministic
- keep answers grounded
- expose limitations clearly
- design for extension without overengineering

For a recruitment assignment, the goal is not to claim production completeness. The goal is to show judgment, scope control, explainability, and a clear path from a working demo toward a stronger system.
