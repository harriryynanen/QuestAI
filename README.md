# QuestAI

QuestAI is a lightweight AI-powered Business Q&A Assistant built for a recruitment assignment. It is designed as a constrained, explainable demo rather than a production system.

The current implementation combines:

- a chat-style Streamlit UI
- a lightweight FastAPI entrypoint
- local document retrieval over synthetic `.md`, `.txt`, and `.pdf` files
- deterministic structured querying over named demo `.csv` datasets
- a small LLM abstraction with OpenAI as the current implementation

The solution is intentionally scoped around decision support, not automated decision-making. It can help an advisor explore policy guidance, inspect customer or case facts, and form cautious preliminary views from mixed evidence, but it does not make approvals, risk decisions, or production-grade financial recommendations.

## Why This Was Built (What did you build and why?)

The assignment is a good fit for a common real-world AI engineering problem: combining unstructured documents, structured records, and LLM capabilities in a way that stays grounded, reviewable, and reasonably safe.

I built QuestAI to demonstrate a few practical engineering choices:

- use LLMs where interpretation and synthesis add value
- keep deterministic execution where exact answers are required
- make provenance visible in the UI
- keep the system small enough to explain end-to-end
- preserve graceful fallback behavior when AI capabilities are unavailable or uncertain

This is why the project is deliberately not a generic chatbot. It is a scoped assistant with explicit boundaries.

## Key Technical And AI Decisions (What key technical or AI-related decisions did you make, and why?)

- LLMs are used for semantic planning and evidence-based synthesis, not for unconstrained execution.
- Structured CSV answers stay deterministic so exact values come from the data rather than model generation.
- Retrieval is intentionally lightweight and local instead of using a vector database or heavier retrieval stack.
- Provenance and limitations are visible in the UI so answers stay inspectable.
- Fallback behavior is built in when LLM capabilities are unavailable or uncertain.
- The solution is intentionally scoped as constrained decision support rather than a generic autonomous agent.

## What The Application Does

QuestAI supports three main question types:

1. Document-oriented questions  
   Example: "What does the policy say about tax arrears?"

2. Structured data questions  
   Example: "What is Harbor Foods Demo Oy's equity ratio?" or "Who is the advisory owner of Harbor Foods Demo Oy?"

3. Combined questions  
   Example: "Based on the policy and customer data, does this customer appear broadly aligned with FlexLine Demo?"

The UI is chat-like on purpose. For a demo, this makes the application easy to test interactively and allows constrained recent-turn follow-ups when the scope is clear.

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

## Architecture Overview (How does the solution work at a high level?)

The repository is organized into small modules with clear responsibilities:

- `app/ui/`  
  Streamlit presentation layer, chat layout, answer cards, prompt chips, and provenance display.

- `app/api.py`  
  Lightweight FastAPI entrypoint that reuses the same service layer as the UI. It is a small integration-facing surface, not a separate backend architecture.

- `app/services/`  
  Answer orchestration, routing, fallback handling, combined-flow coordination, and constrained follow-up scope reuse.

- `app/llm/`  
  Prompt construction, a small LLM client abstraction, and the OpenAI-backed provider implementation used today.

- `app/retrieval/`  
  Document loading, chunking, local ranking, retrieval helpers, and lightweight corpus metadata used by scoring.

- `app/structured/`  
  CSV loading, schema metadata, semantic plan interpretation, deterministic query execution, and dataset-specific structured handlers such as the advisory case pipeline handler.

- `app/config.py` and `app/models.py`  
  Runtime configuration and shared typed models.

## Retrieval Logic

Retrieval is intentionally lightweight and local. There is no vector database or external retrieval infrastructure.

The current retrieval pipeline:

1. load documents from disk
2. chunk them into smaller sections
3. normalize and tokenize the query
4. score chunks with a lightweight ranking layer
5. keep the top matching chunks
6. synthesize an answer only from those retrieved chunks

The ranking layer is stronger than raw keyword overlap but still inspectable. It uses a lightweight combination of:

- token normalization
- weighted term overlap
- heading-aware scoring
- phrase match bonuses
- BM25-like local ranking
- conservative product query expansion
- narrow "other products / alternatives / besides X" retrieval support

Some retrieval hints are still demo-corpus specific by design, but they live in lightweight corpus metadata rather than directly inside the scoring rules.

This is intentionally still a heuristic local retriever, not a production semantic retrieval stack. It is meaningfully better than raw keyword matching for the demo, but still weaker than an embedding-based retrieval system with evaluation, indexing, and observability around it. That heavier path is a future option, not a current feature.

## Structured Data Logic

Structured data is loaded from `.csv` files under `data/structured/`.

The current demo supports two named structured datasets:

- `customer_portfolio`
- `advisory_case_pipeline`

Structured answers stay deterministic by design. The model does not directly answer CSV questions from free-form memory. Instead:

1. the planner interprets the user's intent
2. the planner uses local structured schema metadata to infer the target dataset and field when possible
3. the structured engine maps that plan to a supported deterministic operation
4. the answer is executed directly against the selected DataFrame

Supported structured operations include:

- fact lookup
- filtering
- comparisons
- counts
- existence checks
- customer listing
- some explicitly supported distinct-value listings, such as advisory case products

This is a deliberate middle ground. Semantic planning is schema-aware, but execution still depends on known internal field names and explicit query logic rather than arbitrary dataframe generation.

### Current structured-data limitation

QuestAI does not support arbitrary new structured datasets with zero code changes. Adding a genuinely new structured dataset still requires a small schema and execution extension so the planner knows the dataset and field surface and the deterministic layer knows how to execute it safely.

That is intentional for this demo: reliability and inspectability are prioritized over fully automatic structured extensibility.

A realistic next step would be to move more field descriptions and supported operations into configuration or generated metadata while keeping deterministic validation in front of any newly introduced dataset.

## Routing, Planning, And Fallbacks

Routing is intentionally layered.

- Primary path: QuestAI first tries LLM-based semantic planning.
- Fallback path: if semantic planning is unavailable or weak, the app falls back to rule-based routing.
- Execution fallback: retrieval synthesis and combined synthesis can fall back to deterministic summaries when OpenAI is unavailable.

This design balances flexibility with control:

- LLM planning improves usability
- deterministic fallback preserves reliability
- uncertain cases do not silently turn into confident answers

## Combined Flow

Combined questions mix unstructured and structured evidence:

1. identify that both documents and structured data are needed
2. retrieve the most relevant document evidence
3. build structured evidence deterministically
4. send only that evidence pack into LLM synthesis
5. return a cautious answer with visible sources and limitations

This is still decision support only. The app does not state final eligibility or approval decisions.

## Follow-ups And Scope Handling

QuestAI keeps lightweight recent-session context. This is used narrowly, not as general memory.

Examples of supported follow-up behavior:

- count -> "Name those customers"
- filter result -> "Who are they?"
- grouped combined scope reuse
- advisory owner lookup -> "Does Mika have other companies?"

The implementation intentionally reuses only recent, clearly established scope. If the reference is too ambiguous, the app should fail cautiously rather than guess.

## LLM Provider Position

OpenAI is the implemented provider today.

The code now depends on a small LLM interface rather than wiring `OpenAIAppClient` directly into the whole application. That makes future provider extension easier, but it should be understood as readiness rather than full multi-provider support.

## Lightweight Input Guardrails

QuestAI now applies small input guardrails before LLM-facing planning or synthesis. These are intentionally modest and aimed at demo-level hygiene rather than full security hardening.

Current safeguards include:

- a reasonable question length cap
- normalization of obviously problematic control characters
- safe rejection of clearly malformed control-character-heavy input
- prompt framing that treats the user question as data to analyze, not as instructions to override system behavior

This does not claim to solve prompt injection. It is simply a lightweight acknowledgement of LLM-facing risk in a constrained demo.

## Lightweight API

QuestAI also includes a small API entrypoint. The purpose is not to replace Streamlit, but to show that the same answer flow can be exposed to other systems through a simple HTTP surface.

The API:

- reuses the same `AnswerService`
- returns the same grounded answer shape at a JSON boundary
- is intentionally small and explicit

It is not a production backend. There is no authentication, persistence, or independent backend architecture behind it.

## Fallback Behavior

Fallback behavior is explicit at several stages.

If OpenAI is unavailable:

- semantic planning falls back to rule-based routing
- retrieval synthesis falls back to a deterministic summary of retrieved chunks
- combined synthesis falls back to a deterministic combined evidence summary where possible

If routing or evidence is weak:

- the app returns a cautious low-support or `unknown` response
- it does not treat missing evidence as proof that no guidance exists
- it avoids final-decision wording when evidence is incomplete

If the input itself is clearly malformed for safe LLM use:

- the app can reject it before planning or synthesis
- the response stays professional and explicit about the lightweight nature of the guardrail

## What Is Implemented Now

Implemented in the current version:

- chat-style Streamlit UI
- lightweight FastAPI entrypoint
- `.md`, `.txt`, and `.pdf` document ingestion
- local chunk-based retrieval
- lightweight explainable ranking
- deterministic CSV querying over multiple named datasets
- LLM-based semantic planning
- LLM-based retrieval synthesis
- LLM-based combined evidence synthesis
- lightweight LLM provider abstraction with OpenAI as the current implementation
- constrained recent-turn follow-up and scope reuse
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

## Failure Modes And Limitations (Where can this solution fail or produce incorrect results?)

There are several places where the current system can still be wrong or incomplete.

- retrieval may miss relevant evidence if wording differs too much from the query
- PDF extraction quality depends on readable embedded text
- semantic planning can still misinterpret ambiguous questions
- combined answers depend on both document evidence and structured evidence being available
- grouped follow-up handling is intentionally narrow and not full conversational memory
- support levels are heuristic labels, not calibrated probabilities
- the product and policy logic is synthetic and intentionally simplified
- prompt-injection hardening is only lightweight demo-level input hygiene, not full production protection
- audit trail and observability are still future production-path items rather than implemented features

Most importantly, the app should not be interpreted as making correct real-world financial judgments. It is a controlled demo for grounded AI-assisted decision support patterns.

## Next Steps Toward Production (What would you do next if this were taken toward production?)

- Add stronger retrieval infrastructure and evaluation before treating retrieval as production-grade.
- Add better observability and audit trail support around requests, evidence use, and failures.
- Add authentication and access control for real users and real data.
- Make structured-data onboarding more robust while keeping deterministic validation in place.
- Strengthen security hardening beyond the current lightweight guardrails.
- Add more production-ready backend and deployment maturity around operations and hosting.

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
LLM_PROVIDER=openai
```

You can also use `.streamlit/secrets.toml` based on `.streamlit/secrets.toml.example`.

### Run Streamlit

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

The API uses the same core service layer as the UI and is meant as a simple integration-facing entrypoint, not as a production backend.
