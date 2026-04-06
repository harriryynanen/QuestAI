from models import Route


class SimpleRouter:
    DOCUMENT_INTENT_KEYWORDS = (
        "policy",
        "guideline",
        "guide",
        "instruction",
        "what does",
        "what do the documents say",
        "what does the policy say",
    )

    STRUCTURED_INTENT_KEYWORDS = (
        "which customer",
        "which customers",
        "highest",
        "lowest",
        "largest",
        "minimum",
        "maximum",
        "missing",
        "does ",
        "how many",
    )

    def __init__(
        self,
        retrieval_keywords: tuple[str, ...],
        structured_keywords: tuple[str, ...],
    ) -> None:
        self.retrieval_keywords = retrieval_keywords
        self.structured_keywords = structured_keywords

    def classify(self, question: str) -> Route:
        normalized_question = question.lower()

        matches_retrieval = any(
            keyword in normalized_question for keyword in self.retrieval_keywords
        )
        matches_structured = any(
            keyword in normalized_question for keyword in self.structured_keywords
        )
        document_intent = any(
            keyword in normalized_question for keyword in self.DOCUMENT_INTENT_KEYWORDS
        )
        structured_intent = any(
            keyword in normalized_question for keyword in self.STRUCTURED_INTENT_KEYWORDS
        )

        if matches_retrieval and matches_structured and document_intent and not structured_intent:
            return "retrieval"
        if matches_retrieval and matches_structured:
            return "combined"
        if matches_structured:
            return "structured"
        return "retrieval"
