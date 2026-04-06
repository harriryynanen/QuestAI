import re

from models import DocumentChunk, RetrievedChunk


class KeywordRetriever:
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "say",
        "the",
        "to",
        "what",
        "when",
        "with",
    }

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        chunks: list[DocumentChunk],
    ) -> list[RetrievedChunk]:
        question_terms = self._tokenize(question)
        if not question_terms:
            return []

        results: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_terms = set(self._tokenize(chunk.text))
            matched_terms = sorted(question_terms & chunk_terms)
            if not matched_terms:
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=len(matched_terms),
                    matched_terms=matched_terms,
                )
            )

        results.sort(
            key=lambda item: (
                -item.score,
                item.chunk.file_name,
                item.chunk.section_heading or "",
                item.chunk.chunk_id,
            )
        )
        return results[: self.top_k]

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {
            token for token in tokens
            if len(token) > 2 and token not in self.STOPWORDS
        }
