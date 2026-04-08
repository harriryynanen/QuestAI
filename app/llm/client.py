from typing import Protocol

from app.models import (
    CombinedEvidence,
    RetrievalSynthesisResult,
    RetrievedChunk,
    SemanticPlanningResult,
)


class LLMClient(Protocol):
    """Minimal LLM interface used by QuestAI services."""

    def plan_question(
        self,
        question: str,
        conversation_context: str | None = None,
    ) -> SemanticPlanningResult: ...

    def synthesize_retrieval_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RetrievalSynthesisResult: ...

    def synthesize_combined_answer(
        self,
        question: str,
        evidence: CombinedEvidence,
        document_evidence: list[str],
    ) -> RetrievalSynthesisResult: ...

    def is_available(self) -> bool: ...

    def get_status(self) -> tuple[str, str]: ...
