from models import Route


class SimpleRouter:
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

        if matches_retrieval and matches_structured:
            return "combined"
        if matches_structured:
            return "structured"
        return "retrieval"
