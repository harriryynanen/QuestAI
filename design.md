# QuestAI Design Overview

## 1. Context and Purpose

Business experts often need quick answers based on internal guidance and simple customer-related data. In practice, relevant information is split between:
- textual instructions, policies, and product guidance
- structured records containing customer facts or simple business metrics

The demonstration context is a **fictional business banking advisory scenario**. The user role is similar to a relationship manager working with SME customers, but all materials, customer names, product names, and examples are synthetic and created only for demonstration purposes.

QuestAI is a lightweight Streamlit-based Business Q&A Assistant built for a recruitment assignment. It answers questions against a constrained set of fictional internal sources:

- text-based documents representing internal guidance
- one structured CSV dataset representing customer facts

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

The main architectural principle is that the application, not the model, controls:

- what data sources are available
- what operations are allowed
- how sources are attached to answers
- when fallback behavior is used

## 4. Module Responsibilities

The codebase is split by responsibility rather than by framework layer alone.

### 4.1 `app/ui/`

The UI layer is responsible for:

- Streamlit layout and chat rendering
- answer cards, badges, and provenance presentation
- prompt suggestions and composer interaction
- session-local conversation state

It does not perform routing, retrieval, or structured execution logic directly.

### 4.2 `app/services/`

The service layer orchestrates the answer flow:

- calls the semantic planner
- applies routing fallback rules
- invokes retrieval and structured execution
- assembles combined evidence
- builds the final `AnswerResponse`

This is the coordination layer of the application.

### 4.3 `app/llm/`

The LLM layer is intentionally narrow. It is used for:

- semantic planning / intent interpretation
- retrieval answer synthesis
- combined evidence synthesis

It is not used as an unrestricted reasoning engine and does not directly execute structured CSV operations.

### 4.4 `app/retrieval/`

The retrieval layer handles:

- document loading
- chunking
- lightweight local ranking
- match summaries for explainability

This layer is local and inspectable by design.

### 4.5 `app/structured/`

The structured layer handles:

- CSV loading
- semantic plan interpretation for structured queries
- deterministic execution over tabular data
- grouped customer-level evidence assembly for combined questions

This is where exact data operations stay controlled.

### 4.6 `app/models.py`

Shared dataclasses and typed response objects live here. This keeps the boundaries between modules explicit and reviewer-friendly.

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

This path stays deterministic. The model can help identify the intent, but it does not invent structured answers.

### 5.3 Combined Path

Used when the question needs both policy/product guidance and customer facts.

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

Structured customer data is loaded from the first CSV file found under `data/structured/`. This is intentionally simple for the demo and keeps the source of truth obvious.

## 7. Retrieval Design

The retrieval design is intentionally lightweight and local. There is no vector database, remote index, or external retrieval infrastructure.

### 7.1 Why This Choice

This assignment does not need production-grade retrieval infrastructure. A local retriever is easier to:

- review
- explain
- test
- extend incrementally

### 7.2 Current Ranking Approach

Retrieval now uses a small explainable scoring layer built around:

- token normalization
- BM25-like local term weighting
- heading-aware scoring
- phrase bonuses
- multi-term coverage bonuses
- conservative query expansion for generic product questions

This is a deliberate middle ground between naive keyword matching and a heavier semantic search stack.

### 7.3 Retrieval Explainability

Each retrieved chunk includes a machine-generated match summary, for example:

- matched terms
- heading match
- phrase match
- general policy section
- product section boost

That metadata is useful both for debugging and for evaluator transparency.

## 8. Structured Execution Design

Structured queries are kept deterministic for credibility and control.

Why this matters:

- exact values should come from the CSV, not from model generation
- filters and comparisons should be reproducible
- reviewer trust improves when the data path is explicit

The structured engine therefore maps supported semantic plans into direct DataFrame operations. It also returns field-level or row-level source metadata where appropriate.

## 9. Combined Assessment Design

Combined assessment is the narrowest “AI reasoning” part of the system and therefore the most constrained.

### 9.1 Single-Customer Combined Questions

These combine:

- retrieved policy/product guidance
- deterministic customer facts
- LLM synthesis over the assembled evidence

### 9.2 Grouped Combined Questions

The current implementation also supports scoped grouped preliminary views, for example:

- assessing a previously identified customer set against a product
- grouping customers into cautious buckets such as:
  - broadly aligned based on available evidence
  - caution / mixed signals
  - not enough information

The bucket logic is not meant to be complete financial decisioning. It is a controlled demo approximation with explicit limitations.

### 9.3 Guardrails

The grouped path uses deterministic guardrails where practical:

- missing key information lowers confidence
- caution flags can prevent a positive preliminary view
- output wording avoids final eligibility or approval language

## 10. Conversational Context Handling

QuestAI supports lightweight conversational continuity, but not full conversational memory.

The design intent is:

- reuse recent context when the reference is clear
- avoid broad hidden memory behavior
- stay cautious when scope cannot be resolved safely

Current examples:

- `How many customers are there?` -> `Name those customers`
- filtered result -> `Who are they?`
- grouped combined result -> `Which companies are not`

This is implemented as narrow follow-up resolution inside the service layer rather than as a general memory subsystem.

## 11. Routing And Fallback Design

Routing follows a layered strategy.

### 11.1 Preferred Route Source

The application first tries semantic planning through the LLM.

If the planner returns a confident, valid route, the system uses it directly.

### 11.2 Deterministic Fallback

If planning is unavailable, weak, or unclear:

- the rule-based router is used as fallback
- if that is also unclear, the app returns a controlled `unknown` response

This avoids turning ambiguity into a false confident answer.

### 11.3 Execution Fallbacks

Fallback behavior is explicit at multiple stages:

- retrieval synthesis can fall back to deterministic evidence summarization
- combined synthesis can fall back to deterministic evidence assembly output
- grouped combined answers can return targeted limitation messages if scope or evidence is incomplete

## 12. Provenance And Explainability

A core design choice in QuestAI is that provenance belongs to the application layer, not to the model.

That means:

- visible citations are generated from application metadata
- full source lists come from the app’s own source bookkeeping
- retrieved evidence details come from actual chunk metadata
- support levels and limitations are application-level response properties

This reduces the chance of model-invented citations and keeps answer traceability inspectable.

## 13. UI Design Rationale

The UI is chat-style on purpose.

This is useful in the assignment context because it lets an evaluator:

- test multiple question types quickly
- explore follow-up behavior
- inspect answer grounding turn by turn

The answer is kept visually central, while technical details remain available behind `Why this answer`. This is a tradeoff between readability and transparency.

The UI intentionally exposes:

- citation markers
- support levels
- route and synthesis badges
- expandable provenance details

This is part of the design, not just presentation polish.

## 14. Safety And Scope Boundaries

This repository is intentionally bounded in several ways.

- All data is synthetic.
- No real customer or institution data is included.
- No production decisioning is implemented.
- No hidden approval logic exists.
- No autonomous agent loop is used.

The project is meant to support careful review of constrained AI-assisted decision support patterns, not to simulate a production financial platform.

## 15. Current Limitations

Current limitations are deliberate and should be understood as part of the demo scope:

- retrieval quality is improved but still local and heuristic
- PDF support depends on extractable text
- semantic planning can still fail on unusual phrasing
- support levels are heuristic labels, not calibrated scores
- grouped follow-up resolution is narrow and context-window based
- only a small set of structured operations is supported
- the combined assessment logic is intentionally cautious and simplified

These are acceptable tradeoffs for a recruitment assignment, but they would need to be addressed before production use.

## 16. Extension Path

The code is structured to allow incremental improvement without rewriting the application.

Reasonable next steps include:

- stronger retrieval evaluation and ranking refinement
- broader semantic plan coverage with deterministic execution preserved
- richer combined evidence calibration
- improved PDF and document parsing
- stronger logging, observability, and auditability
- authentication and production deployment hardening

The key design intent is that future work can be added by strengthening existing modules rather than replacing the overall shape of the system.
