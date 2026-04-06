# Business Q&A Assistant – Design Overview

## 1. Problem Definition

Business experts often need quick answers based on internal guidance and simple customer-related data. In practice, relevant information is split between:
- textual instructions, policies, and product guidance
- structured records containing customer facts or simple business metrics

This solution provides a single interface where a user can ask natural-language questions and receive concise, well-reasoned answers grounded only in the provided sample data.

The demonstration context is a **fictional business banking advisory scenario**. The user role is similar to a relationship manager working with SME customers, but all materials, customer names, product names, and examples are synthetic and created only for demonstration purposes.

The focus is on:
- fast access to relevant information
- concise answers suitable for busy expert users
- visible source references
- explicit handling of uncertainty and missing information

---

## 2. Data Sources

The system operates on a limited, well-defined sample dataset.

### Unstructured Data (RAG)
2–3 synthetic internal documents, for example:
- a financing product guideline
- an internal eligibility policy summary
- an FAQ or internal advisory note

These documents simulate the kind of internal written material an expert may need to consult.

### Structured Data (Tool)
1 synthetic CSV or JSON dataset containing simple customer-level facts, for example:
- customer name
- segment
- turnover
- EBITDA
- equity ratio
- existing products
- other simple eligibility-relevant fields

### Metadata
Where possible, the system preserves lightweight metadata such as:
- document name
- section heading
- page reference (if available)
- chunk identifier

This metadata is used to make answers more transparent.

---

## 3. Supported Question Types

### 3.1 Document-based Questions (Retrieval)
Questions about rules, instructions, or product guidance, for example:
- “What does the policy say about minimum equity ratio for Product Alpha Demo?”
- “How should an advisor interpret the basic eligibility guidance for export financing?”

→ Answered using document retrieval over the synthetic documents.

### 3.2 Structured Data Questions
Questions about specific customer facts or simple aggregations, for example:
- “What is Demo Manufacturing Ltd’s latest equity ratio?”
- “Which demo customer has the highest turnover in the sample data?”

→ Answered using deterministic structured-data processing.

### 3.3 Combined Questions (Optional but Valuable)
Questions that require both policy context and customer data, for example:
- “Based on the provided policy and sample customer data, does Demo Manufacturing Ltd appear to meet the basic criteria for Product Alpha Demo?”

→ Uses both document retrieval and structured data.

The answer is positioned as **decision support**, not as an automated banking decision.

---

## 4. Out of Scope (Explicit Limitations)

To keep the solution focused, reliable, and aligned with the assignment scope:

- no real-time integrations with operational systems
- no write actions or workflow automation
- no general knowledge outside the provided sample data
- no production-grade credit decisioning
- no long-term user memory

The system only answers based on the provided dataset and does not claim to replace formal review or approval processes.

---

## 5. System Behavior: Retrieval vs Structured Data

The system uses simple and transparent routing logic.

### Retrieval (RAG)
Used when the question is about:
- policy
- instructions
- product guidance
- interpretation of written material

The system searches the indexed document chunks semantically and provides relevant context to the LLM.

### Structured Data Tool
Used when the question is about:
- specific customer facts
- simple filters
- aggregations or comparisons
- numeric values

The system answers these using deterministic data handling (for example, pandas-based filtering or aggregation).

### Combined Use
If the question requires both written guidance and customer facts, the system can use both paths and then synthesize a final answer.

The routing logic is intentionally kept simple, explainable, and easy to extend.

---

## 6. Answer Format and Uncertainty Handling

Each response should include the following sections.

### Answer
A concise answer in plain language.

### Sources Used
Examples:
- `product_policy_demo.md`, section “Eligibility criteria”
- `customer_metrics_demo.csv`, row or filtered result reference

### Support Level
The response is labelled as one of:
- **Directly supported**
- **Partially supported**
- **Not sufficiently supported**

### Missing Information / Limitations
If the available evidence is incomplete, the system should say so clearly. Example:

> I could not find sufficient support in the provided sample data to answer this confidently.

The system avoids guessing and does not use external knowledge.

---

## 7. Design Principles

- clarity over complexity
- source-grounded answers
- explicit uncertainty handling
- modular separation of concerns
- easy extensibility

The architecture should make it straightforward to:
- add new documents without changing core logic
- add new structured datasets with limited code changes
- add new tools through a modular routing layer
- replace the vector store or model through configuration

---

## 8. Data Safety and Domain Framing

To avoid any ambiguity, the demonstration uses only **synthetic sample material**:
- no real customer names
- no real product names
- no real internal documents
- no direct reference to any real financial institution

All names should clearly indicate that the data is fictional test material, for example:
- `Demo Manufacturing Ltd`
- `North Harbor Test Oy`
- `Product Alpha Demo`
- `Eligibility Guide – Sample Only`

This keeps the project safe, portable, and appropriate for a recruitment exercise.
