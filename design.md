# QuestAI Design Overview

## 1. Context and Purpose

Business experts often need quick answers based on internal guidance and simple customer-related data. In practice, relevant information is split between:

- textual instructions, policies, and product guidance
- structured records containing customer facts or simple business metrics

The demonstration context is a fictional business banking advisory scenario. The user role is similar to a relationship manager working with SME customers, but all materials, customer names, product names, and examples are synthetic and created only for demonstration purposes.

QuestAI is a lightweight Streamlit-based Business Q&A Assistant built for a recruitment assignment. It answers questions against a constrained set of fictional internal sources:

- text-based documents representing internal guidance
- named structured CSV datasets representing customer facts and advisory case data

The goal is not to simulate a production banking decision engine. The goal is to demonstrate clear scoping, grounded reasoning, uncertainty handling, and an extensible structure.

## 2. Design Goals

The current design is optimized for five goals:

1. keep the system explainable end-to-end
2. separate probabilistic interpretation from deterministic execution
3. keep the implementation lightweight enough for a recruitment assignment
4. preserve graceful fallback behavior when LLM support is missing or weak
5. make future extension possible without rewriting the core flow

These goals are more important here than raw model capability. The project intentionally favors controlled behavior over a more open-ended assistant architecture.

## 3. System Shape At A Glance

QuestAI uses a layered request flow:

1. UI captures the user question and recent conversation context
2. semantic planning interprets the question into a constrained plan
3. routing selects `retrieval`, `structured`, `combined`, or `unknown`
4. the selected execution path gathers evidence
5. the app returns an answer with support level, limitations, and visible provenance

The same orchestration layer is used by both the Streamlit UI and the lightweight API entrypoint.

The main architectural principle is that the application, not the model, controls:

- what data sources are available
- what operations are allowed
- how sources are attached to answers
- when fallback behavior is used

## 4. Module Responsibilities

### 4.1 `app/ui/`

The UI layer is responsible for Streamlit layout, chat rendering, prompt suggestions, answer cards, and provenance presentation. It does not perform routing, retrieval, or structured execution directly.

### 4.2 `app/api.py`

The API layer is intentionally minimal. It exposes a health endpoint and a single answer endpoint, reuses the same `AnswerService` as the UI, and exists to show integration readiness rather than to introduce a second backend architecture.

### 4.3 `app/services/`

The service layer orchestrates the answer flow:

- semantic planning
- routing fallback rules
- retrieval and structured dispatch
- combined evidence assembly
- final `AnswerResponse` creation

It also applies constrained recent-turn follow-up handling through a dedicated conversation scope resolver.

The service layer now also applies lightweight question guardrails before LLM-facing planning or synthesis so obviously malformed inputs can be normalized or rejected early.

### 4.4 `app/llm/`

The LLM layer is intentionally narrow. It is used for:

- semantic planning / intent interpretation
- retrieval answer synthesis
- combined evidence synthesis

It is not used as an unrestricted reasoning engine and does not directly execute structured CSV operations. The code now depends on a small LLM client abstraction, with OpenAI as the current concrete provider implementation.

The prompt construction is also intentionally separated by task in `app/llm/prompts.py`:

`build_semantic_plan_messages(...)`
`build_retrieval_messages(...)`
`build_combined_answer_messages(...)`

These are not interchangeable prompts. Each one supports a different stage of the request lifecycle.

**`build_semantic_plan_messages(...)`** is used first, during question interpretation. It does not answer the user’s question. Instead, it asks the model to produce a constrained semantic plan such as route, operation, field, dataset, and confidence. This is the point where the LLM helps with semantic interpretation rather than execution.

**`build_retrieval_messages(...)`** is used only after document retrieval has already selected grounded chunks. Its job is to turn the retrieved evidence into a bounded synthesis prompt so the model answers only from the provided document context.

**`build_combined_answer_messages(...)`** is used only in the combined path, after both document evidence and deterministic structured evidence have already been assembled. Its role is to produce a cautious synthesis from that explicit evidence pack rather than letting the model infer freely from outside knowledge.

This separation is a deliberate quality choice. It makes the LLM layer easier to test, easier to explain, and less likely to blur planning, retrieval synthesis, and combined evidence synthesis into one opaque step.


### 4.5 `app/retrieval/`

The retrieval layer handles document loading, chunking, local ranking, and explainable match summaries. Lightweight demo-corpus metadata is now separated from the scoring logic so that retrieval behavior and corpus-specific hints are easier to reason about independently.

### 4.6 `app/structured/`

The structured layer handles:

- CSV loading
- schema-aware semantic plan interpretation
- deterministic DataFrame execution
- grouped customer-level evidence assembly for combined questions
- dataset-specific handlers such as the advisory case pipeline handler

This is where exact data operations stay controlled. The current implementation supports multiple named datasets, including `customer_portfolio` and `advisory_case_pipeline`.

### 4.7 `app/models.py`

Shared dataclasses and typed response objects live here. This keeps boundaries between modules explicit and reviewer-friendly.

## 5. Question Paths

QuestAI currently supports three intentional execution paths.

### 5.1 Retrieval Path

Used for document-oriented questions such as policy and product guidance.

Flow:

1. retrieve relevant chunks from local documents
2. select a small grounded evidence set
3. synthesize an answer from those chunks only
4. attach sources from application metadata

### 5.2 Structured Path

Used for CSV-backed questions such as:

- point facts
- comparisons
- filters
- counts
- existence checks
- customer lists
- some explicitly supported distinct-value listings

This path stays deterministic. The model can help identify the intent, but it does not invent structured answers.

### 5.3 Combined Path

Used when the question needs both policy or product guidance and structured customer facts.

Flow:

1. retrieve relevant document evidence
2. build structured evidence deterministically
3. combine them into an explicit evidence pack
4. synthesize a cautious answer from that pack only

This path is still deliberately constrained. It is not an agent and does not perform arbitrary multi-step tool selection.

## 6. Data Handling Strategy

### 6.1 Documents

Documents are loaded from `data/docs/` and currently support:

- `.md`
- `.txt`
- `.pdf`

PDF support is text-extraction based. The system does not attempt advanced OCR or document understanding beyond recoverable text.

### 6.2 Chunking

Chunking is intentionally simple and readable:

- markdown headings are treated as section boundaries
- empty lines also trigger chunk flushing
- long sections are split into smaller pieces by character budget

This is sufficient for a small demo while keeping the retrieval logic easy to explain.

### 6.3 Structured Data

Structured data is loaded from a controlled set of named CSV datasets under `data/structured/`.

Today that means:

- `customer_portfolio`
- `advisory_case_pipeline`

The planner sees compact schema metadata for those datasets and can infer dataset plus field before deterministic execution runs. This is more flexible than pure keyword routing, but still far from arbitrary zero-code structured onboarding.

## 7. Retrieval Design

The retrieval design is intentionally lightweight and local. There is no vector database, remote index, or external retrieval infrastructure.

Retrieval uses a small explainable scoring layer built around:

- token normalization
- BM25-like local term weighting
- heading-aware scoring
- phrase bonuses
- multi-term coverage bonuses
- conservative query expansion for generic product questions
- narrow alternative-product retrieval support using corpus metadata and overview or sibling-section biasing

This is a deliberate middle ground between naive keyword matching and a heavier semantic search stack.

Each retrieved chunk includes a machine-generated match summary, which is useful both for debugging and evaluator transparency.

This remains a local heuristic retriever. It is stronger than raw keyword overlap, but it is still not equivalent to a production semantic retrieval system with embeddings, indexing strategy, evaluation, and monitoring.

## 8. Structured Execution Design

Structured queries are kept deterministic for credibility and control.

Why this matters:

- exact values should come from the CSV, not from model generation
- filters and comparisons should be reproducible
- reviewer trust improves when the data path is explicit

The structured engine maps supported semantic plans into direct DataFrame operations and returns field-level or row-level source metadata where appropriate.

This has evolved into a planner-first structured flow: the LLM helps resolve route, dataset, operation, and field against known schema metadata, while execution remains explicit and deterministic. That keeps natural language support broader without turning the structured path into free-form model execution.

Structured onboarding is still controlled. Adding a genuinely new dataset still requires a small schema and execution extension rather than arbitrary zero-code ingestion.

## 9. Combined Assessment Design

Combined assessment is the narrowest AI reasoning part of the system and therefore the most constrained.

Single-customer combined questions mix:

- retrieved policy or product guidance
- deterministic customer facts
- LLM synthesis over the assembled evidence

The current implementation also supports grouped preliminary views over a previously established customer scope. Those outputs use deliberately cautious buckets such as:

- broadly aligned based on available evidence
- caution / mixed signals
- not enough information

The bucket logic is not meant to be complete financial decisioning. It is a controlled demo approximation with explicit limitations.

## 10. Conversational Context Handling

QuestAI supports lightweight conversational continuity, but not full conversational memory.

The design intent is:

- reuse recent context when the reference is clear
- avoid broad hidden memory behavior
- stay cautious when scope cannot be resolved safely

Current examples include:

- count -> "Name those customers"
- filtered result -> "Who are they?"
- grouped combined scope reuse
- advisory owner reverse lookup when the previous structured value is clear

This is implemented as narrow follow-up resolution inside the service layer, with recent-scope handling extracted into a dedicated helper rather than embedded as a general memory subsystem.

## 11. Routing And Fallback Design

Routing follows a layered strategy.

- Preferred route source: semantic planning through the LLM.
- Deterministic fallback: if planning is unavailable or weak, use the rule-based router.
- Execution fallback: retrieval and combined synthesis can fall back to deterministic summaries.

There is also a lightweight pre-planning guardrail step for clearly malformed or unsafe-to-process input. This is intentionally small and should be read as risk awareness, not as comprehensive prompt-injection protection.

This avoids turning ambiguity or missing AI support into a false confident answer.

## 12. Provenance And Explainability

A core design choice in QuestAI is that provenance belongs to the application layer, not to the model.

That means:

- visible citations are generated from application metadata
- full source lists come from the app's own source bookkeeping
- retrieved evidence details come from actual chunk metadata
- support levels and limitations are application-level response properties

This reduces the chance of model-invented citations and keeps answer traceability inspectable.

## 13. Safety And Scope Boundaries

This repository is intentionally bounded in several ways.

- All data is synthetic.
- No real customer or institution data is included.
- No production decisioning is implemented.
- No hidden approval logic exists.
- No autonomous agent loop is used.

The project is meant to support careful review of constrained AI-assisted decision support patterns, not to simulate a production financial platform.

The current prompt design also treats user questions and recent conversation content as untrusted input to classify or answer from, not as instructions to alter the system’s rules. This is supported by lightweight input guardrails and explicit prompt framing. It is not full production-grade prompt-injection hardening, but it is a deliberate demo-level safeguard.


## 14. Current Limitations

Current limitations are deliberate and should be understood as part of the demo scope:

- retrieval quality is improved but still local and heuristic
- PDF support depends on extractable text
- semantic planning can still fail on unusual phrasing
- support levels are heuristic labels, not calibrated scores
- grouped follow-up resolution is narrow and context-window based
- only a small set of structured operations is supported
- the combined assessment logic is intentionally cautious and simplified
- OpenAI is the only real provider implementation today even though the app now depends on a small LLM interface
- retrieval is still local and heuristic rather than a production semantic retrieval stack
- structured onboarding still requires small code changes for new named datasets
- prompt-injection hardening is only lightweight demo-level guardrailing
- observability and audit trail support are still missing

These are acceptable tradeoffs for a recruitment assignment, but they would need to be addressed before production use.

## 15. Extension Path

The code is structured to allow incremental improvement without rewriting the application.

Reasonable next steps include:

- stronger retrieval evaluation and ranking refinement
- broader semantic plan coverage with deterministic execution preserved
- more configuration- or metadata-driven structured onboarding for new named datasets
- richer combined evidence calibration
- improved PDF and document parsing
- stronger logging, observability, and auditability
- authentication and production deployment hardening

The key design intent is that future work can be added by strengthening existing modules rather than replacing the overall shape of the system.

## 16. Current Design Shape

The current codebase reflects a few targeted refactors that materially shaped the design:

- planner-first schema-aware structured dataset selection instead of growing dataset-specific routing rules
- multi-dataset structured support with explicit schema metadata
- extracted conversation scope resolver for constrained follow-up handling
- advisory case pipeline execution separated from customer portfolio execution
- retrieval corpus metadata separated from scoring logic
- more consistent `app...` package imports across UI, API, and tests
- lightweight question guardrails ahead of LLM-facing planning and synthesis

These are not separate subsystems so much as cleanup steps that made the current architecture easier to explain and extend without changing its basic shape.
