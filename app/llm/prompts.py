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
