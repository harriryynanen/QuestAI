# QuestAI Design Overview

## 1. Context and Purpose

Business experts often need quick answers based on internal guidance and simple customer-related data. In practice, relevant information is split between:
- textual instructions, policies, and product guidance
- structured records containing customer facts or simple business metrics

The demonstration context is a **fictional business banking advisory scenario**. The user role is similar to a relationship manager working with SME customers, but all materials, customer names, product names, and examples are synthetic and created only for demonstration purposes.

QuestAI is a lightweight Streamlit-based Business Q&A Assistant built for a recruitment assignment. It answers questions against a constrained set of fictional internal sources:

- markdown documents representing internal guidance
- one structured CSV dataset representing customer facts

The goal is not to simulate a production banking decision engine. The goal is to demonstrate clear scoping, grounded reasoning, uncertainty handling, and an extensible structure.

## 2. Current Architecture

The current system uses a modular three-step pattern:

1. semantic planning
2. constrained execution
3. grounded answer generation

### 2.1 Semantic Planning

An OpenAI-based planner interprets the user question into a compact JSON plan. The plan can include:

- `route`: `retrieval` | `structured` | `combined` | `unknown`
- `operation`: `fact` | `filter` | `comparison` | `count` | `exists` | `policy_lookup` | `product_guidance` | `preliminary_assessment` | `unknown`
- `customer_name`
- `field_name`
- `product_name`
- `document_topic`
- `needs_documents`
- `needs_structured_data`
- `confidence`
- `reason`

This planning layer improves flexibility over pure keyword routing, while keeping the system constrained and inspectable.

### 2.2 Deterministic Execution

The planner does not execute tools directly.

Execution remains separated by source type:

- `retrieval`:
  - load markdown documents
  - split into chunks
  - retrieve top chunks with deterministic scoring
- `structured`:
  - load CSV with pandas
  - execute deterministic fact/filter/comparison/count/exists logic
- `combined`:
  - assemble both document evidence and structured evidence
  - keep the evidence pack explicit

### 2.3 Answer Generation

- Retrieval answers use OpenAI synthesis only after relevant chunks are selected.
- Structured answers remain deterministic and are built directly from CSV results.
- Combined answers use OpenAI only to synthesize across an explicit evidence pack assembled by the app.

This preserves the principle that LLMs interpret and summarize, while the application controls what evidence is available.

## 3. Data Sources

### 3.1 Unstructured Source

Markdown files in `data/docs/` act as synthetic internal guidance. These are used for:

- policy questions
- product guidance questions
- written-rule interpretation

### 3.2 Structured Source

The CSV file in `data/structured/` acts as a synthetic customer portfolio dataset. It is used for:

- customer fact lookups
- filters
- comparisons
- counts
- existence checks

## 4. Routing And Fallback Strategy

Routing now follows this order:

1. try semantic planning through the LLM
2. if the returned route is confident and valid, use it
3. otherwise fall back to the deterministic rule-based router
4. if still unclear, return a controlled unclear-response

The rule-based router remains useful for robustness, but it is no longer the primary behavior.

## 5. Combined Evidence Flow

Combined questions are now handled through a constrained MVP flow:

1. semantic planner identifies a combined question
2. retrieval layer fetches relevant markdown chunks
3. structured layer assembles relevant customer evidence deterministically
4. answer service builds a combined evidence pack
5. OpenAI synthesizes a cautious answer using only:
   - retrieved document evidence
   - structured evidence summary
   - explicit missing-information notes

This is still not an autonomous agent. The app decides what evidence is gathered and what operations are allowed.

## 6. Source Grounding

Source references shown to the user come from application metadata, not model-generated citations.

Examples:

- document source: file name + section heading
- structured source: CSV file + row/column/filter reference

The model may generate explanation text, but source references remain under application control.

## 7. Uncertainty Handling

The app is designed to avoid fake certainty.

Responses always include:

- Answer
- Sources Used
- Support Level
- Limitations

When support is weak, missing, or ambiguous, the app says so explicitly. Combined answers are framed as preliminary support, not final decisions.

## 8. Why This Design Is Credible For The Assignment

This design aims to show:

- modular separation of concerns
- constrained and grounded LLM usage
- deterministic structured execution
- explicit fallback behavior
- easy extensibility without overengineering

The code makes it clear where each responsibility lives:

- planning in `app/llm/` and `app/structured/planner.py`
- retrieval in `app/retrieval/`
- structured execution in `app/structured/query_engine.py`
- orchestration in `app/services/answer_service.py`
- UI in `app/ui/streamlit_app.py`

## 9. Current Limitations

The current system still has deliberate limits:

- no embeddings or vector database
- markdown retrieval only for actual content use
- no PDF parsing
- no unrestricted tool use
- no production-grade decision engine
- combined reasoning remains cautious and lightweight

These limits are intentional to keep the MVP understandable and honest.

## 10. Near-Term Next Steps

The most sensible next improvements are:

- add lightweight evaluation coverage for routing, retrieval, structured, and combined examples
- improve combined confidence labelling and evidence presentation
- expand structured alias coverage and planner robustness without giving up deterministic execution
