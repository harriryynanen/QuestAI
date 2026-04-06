from models import RetrievedChunk


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


def build_intent_classification_messages(question: str) -> list[dict[str, str]]:
    instructions = (
        "Classify the user's question into exactly one route for a fictional business banking advisory demo. "
        "Do not answer the question. "
        "Return JSON with exactly these keys: route, confidence, reason. "
        "route must be one of: retrieval, structured, combined, unknown. "
        "confidence must be one of: low, medium, high. "
        "Use retrieval for questions about policies, written guidance, product descriptions, or document interpretation. "
        "Use structured for questions about customer facts, rankings, filters, numeric values, or yes/no fields from CSV data. "
        "Use combined only when the question explicitly requires both document guidance and customer-specific structured data. "
        "Use unknown when the intent is too unclear to route confidently."
    )

    user_message = (
        f"Question:\n{question}\n\n"
        "Classify only the intent. Do not use outside knowledge and do not invent missing context."
    )

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]
