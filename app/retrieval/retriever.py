"""Lightweight, explainable retrieval over chunked local documents."""

from __future__ import annotations

from models import DocumentChunk, RetrievedChunk
from retrieval.scoring import RetrievalScoringHelper


class KeywordRetriever:
    """Rank chunks with local BM25-style scoring plus explainable bonuses."""

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        chunks: list[DocumentChunk],
    ) -> list[RetrievedChunk]:
        """Return the strongest locally ranked chunks for a question."""
        query_profile = RetrievalScoringHelper.build_query_profile(question)
        if not query_profile.term_weights:
            return []

        corpus_profile = RetrievalScoringHelper.build_corpus_profile(chunks)
        results: list[RetrievedChunk] = []
        for chunk_profile in corpus_profile.chunk_profiles:
            score, matched_terms, match_summary = RetrievalScoringHelper.score_chunk(
                query=query_profile,
                chunk_profile=chunk_profile,
                corpus=corpus_profile,
            )
            if not matched_terms or score < RetrievalScoringHelper.MIN_SCORE:
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk_profile.chunk,
                    score=score,
                    matched_terms=sorted(matched_terms),
                    match_summary=match_summary,
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
