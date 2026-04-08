from models import RetrievedChunk
from structured.schema_metadata import (
    SUPPORTED_SEMANTIC_FIELD_NAMES,
    build_structured_schema_prompt_text,
)


def build_retrieval_messages(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    context_blocks = []
    for item in retrieved_chunks:
        heading = item.chunk.section_heading or "No heading"
        context_blocks.append(
            "\n".join(
                [
                    f"Source: {item.chunk.file_name}",
                    f"Section: {heading}",
                    f"Chunk ID: {item.chunk.chunk_id}",
                    f"Matched terms: {', '.join(item.matched_terms)}",
                    f"Context: {item.chunk.text}",
                ]
            )
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    instructions = (
        "You are helping with a fictional business banking advisory demo. "
        "Answer only from the provided markdown context. "
        "Do not use outside knowledge. "
        "If the context is insufficient, say so clearly. "
        "Do not claim approval, eligibility, suitability, or final decisions. "
        "Keep the answer concise, professional, and grounded in the provided text only. "
        "Return JSON with exactly these keys: answer, support_level, limitations. "
        "support_level must be one of: low, medium, high."
    )

    user_message = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Use only this context. If it does not fully answer the question, say what is supported and what remains uncertain."
    )

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]


def build_semantic_plan_messages(
    question: str,
    conversation_context: str | None = None,
) -> list[dict[str, str]]:
    schema_text = build_structured_schema_prompt_text()
    supported_field_names = ", ".join(SUPPORTED_SEMANTIC_FIELD_NAMES)
    instructions = (
        "Interpret the user's question into a constrained semantic plan for a fictional business banking advisory demo. "
        "Do not answer the question. "
        "Return JSON with exactly these keys: route, operation, customer_name, field_name, product_name, document_topic, comparison_direction, filter_value, needs_documents, needs_structured_data, confidence, reason, structured_dataset. "
        "route must be one of: retrieval, structured, combined, unknown. "
        "operation must be one of: fact, filter, comparison, count, list, exists, policy_lookup, product_guidance, preliminary_assessment, unknown. "
        "confidence must be one of: low, medium, high. "
        "field_name should use only supported internal column names when known, otherwise null. "
        f"Supported field_name values include: {supported_field_names}. "
        "structured_dataset must be one of: customer_portfolio, advisory_case_pipeline, or null. "
        "comparison_direction must be highest, lowest, or null. "
        "Use retrieval for questions about policies, guides, guidance, criteria, product descriptions, or document interpretation. "
        "Use structured for customer facts, case facts, filters, counts, list requests, existence checks, or comparisons from CSV data. "
        "Use combined when the question explicitly needs both document guidance and structured customer data together, including questions about which of a referenced customer group appear aligned with a product or criteria. "
        "Use unknown when the question is too unclear to map safely. "
        "If recent conversation context is provided, use it only to resolve follow-up references such as 'those customers', 'they', or 'that customer'. "
        "Do not repeat the previous operation blindly just because similar wording appeared in the context. "
        "Support modest English and Finnish business wording variation. "
        f"{schema_text}"
    )

    if conversation_context:
        user_message = (
            f"Current question:\n{question}\n\n"
            f"Recent conversation context:\n{conversation_context}\n\n"
            "Interpret only into a safe semantic plan. Use the context only to resolve references. "
            "Do not invent missing data and do not answer the question itself."
        )
    else:
        user_message = (
            f"Current question:\n{question}\n\n"
            "Interpret only into a safe semantic plan. Do not invent missing data and do not answer the question itself."
        )

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]


def build_combined_answer_messages(
    question: str,
    document_evidence: list[str],
    structured_evidence: str,
    missing_information: list[str],
) -> list[dict[str, str]]:
    instructions = (
        "You are helping with a fictional business banking advisory demo. "
        "Write a cautious combined answer using only the provided evidence pack. "
        "Do not use outside knowledge. "
        "Do not claim approval, eligibility, suitability, or final credit decisions. "
        "Clearly distinguish what is supported by documents, what is supported by structured data, and what remains uncertain. "
        "Return JSON with exactly these keys: answer, support_level, limitations. "
        "support_level must be one of: low, medium, high."
    )

    document_text = "\n\n".join(document_evidence) if document_evidence else "No document evidence retrieved."
    missing_text = "\n".join(f"- {item}" for item in missing_information) if missing_information else "- No explicit missing-information note."
    user_message = (
        f"Question:\n{question}\n\n"
        f"Document evidence:\n{document_text}\n\n"
        f"Structured evidence:\n{structured_evidence}\n\n"
        f"Missing information notes:\n{missing_text}\n\n"
        "Use only this evidence pack."
    )

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]
