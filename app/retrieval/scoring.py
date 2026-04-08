"""Lightweight, explainable scoring helpers for local document retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.models import DocumentChunk
from app.retrieval.corpus_metadata import (
    DEMO_GENERAL_POLICY_HEADINGS,
    DEMO_KNOWN_PHRASES,
    DEMO_PRODUCT_TERMS,
)


@dataclass(frozen=True)
class QueryProfile:
    """Normalized retrieval query features used for chunk scoring."""

    normalized_question: str
    term_weights: dict[str, float]
    phrases: list[str]
    expanded_terms: list[str]
    has_specific_product: bool
    asks_product_alternatives: bool
    referenced_product_terms: tuple[str, ...]


@dataclass(frozen=True)
class ChunkProfile:
    """Preprocessed token statistics for a retrieval chunk."""

    chunk: DocumentChunk
    body_term_counts: Counter[str]
    heading_term_counts: Counter[str]
    normalized_length: int


@dataclass(frozen=True)
class CorpusProfile:
    """Corpus-wide token statistics for BM25-like scoring."""

    chunk_profiles: list[ChunkProfile]
    document_frequencies: Counter[str]
    average_chunk_length: float


class RetrievalScoringHelper:
    """Explainable local retrieval scoring with BM25-style term weighting."""

    PRODUCT_TERMS = DEMO_PRODUCT_TERMS
    GENERIC_PRODUCT_TERMS = {"product", "products", "service", "services", "offering", "offerings"}
    ALTERNATIVE_PRODUCT_TERMS = {"alternative", "alternatives", "other", "besides"}
    POLICY_TERMS = {"policy", "criteria", "eligibility", "guide", "guideline", "instruction"}
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
        "does",
        "about",
        "based",
        "provided",
        "there",
        "their",
        "them",
        "this",
        "that",
        "into",
        "used",
        "only",
        "just",
    }
    HEADING_WEIGHT = 1.8
    K1 = 1.5
    B = 0.75
    PHRASE_BONUS = 2.5
    REPEATED_PHRASE_BONUS = 0.8
    MULTI_TERM_COVERAGE_BONUS = 0.5
    GENERAL_POLICY_HEADING_BONUS = 1.1
    PRODUCT_SECTION_BOOST = 0.9
    PRODUCT_OVERVIEW_BOOST = 1.4
    SIBLING_PRODUCT_SECTION_BOOST = 1.1
    REFERENCED_PRODUCT_SECTION_PENALTY = 4.0
    PRODUCT_SPECIFIC_PENALTY = 1.0
    MIN_SCORE = 1.25
    KNOWN_PHRASES = DEMO_KNOWN_PHRASES
    GENERAL_POLICY_HEADINGS = DEMO_GENERAL_POLICY_HEADINGS

    @classmethod
    def build_query_profile(cls, question: str) -> QueryProfile:
        """Normalize the question and add conservative query expansion."""
        normalized_question = question.lower()
        base_terms = cls.tokenize(question)
        term_weights = {term: 1.0 for term in base_terms}
        phrases = [phrase for phrase in cls.KNOWN_PHRASES if phrase in normalized_question]
        expanded_terms: list[str] = []

        referenced_product_terms = tuple(
            sorted(term for term in cls.PRODUCT_TERMS if term in normalized_question)
        )
        has_specific_product = bool(referenced_product_terms)
        asks_for_other_products = cls._asks_for_other_products(
            normalized_question,
            base_terms,
        )
        asks_about_products = any(term in base_terms for term in cls.GENERIC_PRODUCT_TERMS) or (
            has_specific_product and asks_for_other_products
        )
        asks_product_alternatives = asks_about_products and asks_for_other_products

        if asks_about_products and not has_specific_product:
            for term in sorted(cls.PRODUCT_TERMS):
                expanded_terms.append(term)
                term_weights.setdefault(term, 0.45)
        elif asks_product_alternatives and referenced_product_terms:
            for term in sorted(cls.PRODUCT_TERMS - set(referenced_product_terms)):
                expanded_terms.append(term)
                term_weights.setdefault(term, 0.5)

        return QueryProfile(
            normalized_question=normalized_question,
            term_weights=term_weights,
            phrases=phrases,
            expanded_terms=expanded_terms,
            has_specific_product=has_specific_product,
            asks_product_alternatives=asks_product_alternatives,
            referenced_product_terms=referenced_product_terms,
        )

    @classmethod
    def build_corpus_profile(cls, chunks: list[DocumentChunk]) -> CorpusProfile:
        """Preprocess chunks and corpus statistics used for ranking."""
        chunk_profiles: list[ChunkProfile] = []
        document_frequencies: Counter[str] = Counter()

        for chunk in chunks:
            body_term_counts = Counter(cls.tokenize(chunk.text))
            heading_term_counts = Counter(cls.tokenize(chunk.section_heading or ""))
            unique_terms = set(body_term_counts) | set(heading_term_counts)
            document_frequencies.update(unique_terms)
            normalized_length = sum(body_term_counts.values()) + sum(heading_term_counts.values())
            chunk_profiles.append(
                ChunkProfile(
                    chunk=chunk,
                    body_term_counts=body_term_counts,
                    heading_term_counts=heading_term_counts,
                    normalized_length=max(normalized_length, 1),
                )
            )

        average_chunk_length = (
            sum(profile.normalized_length for profile in chunk_profiles) / len(chunk_profiles)
            if chunk_profiles
            else 1.0
        )
        return CorpusProfile(
            chunk_profiles=chunk_profiles,
            document_frequencies=document_frequencies,
            average_chunk_length=max(average_chunk_length, 1.0),
        )

    @classmethod
    def score_chunk(
        cls,
        query: QueryProfile,
        chunk_profile: ChunkProfile,
        corpus: CorpusProfile,
    ) -> tuple[float, list[str], str]:
        """Score one chunk and return score, matched terms, and a human summary."""
        matched_terms: list[str] = []
        notes: list[str] = []
        score = 0.0

        heading_text = (chunk_profile.chunk.section_heading or "").lower()
        body_text = chunk_profile.chunk.text.lower()
        body_and_heading = f"{heading_text} {body_text}".strip()
        total_chunks = len(corpus.chunk_profiles)

        for term, weight in query.term_weights.items():
            body_tf = chunk_profile.body_term_counts.get(term, 0)
            heading_tf = chunk_profile.heading_term_counts.get(term, 0)
            if not body_tf and not heading_tf:
                continue

            matched_terms.append(term)
            weighted_tf = body_tf + (heading_tf * cls.HEADING_WEIGHT)
            doc_freq = corpus.document_frequencies.get(term, 0)
            idf = math.log(1 + ((total_chunks - doc_freq + 0.5) / (doc_freq + 0.5)))
            normalization = cls.K1 * (
                1 - cls.B + cls.B * (chunk_profile.normalized_length / corpus.average_chunk_length)
            )
            bm25 = idf * ((weighted_tf * (cls.K1 + 1)) / (weighted_tf + normalization))
            score += bm25 * weight

        if matched_terms:
            notes.append(f"matched terms: {', '.join(matched_terms)}")

        heading_matches = [
            term for term in matched_terms if chunk_profile.heading_term_counts.get(term, 0)
        ]
        if heading_matches:
            notes.append(f"heading match: {', '.join(heading_matches)}")

        matched_phrases = [phrase for phrase in query.phrases if phrase in body_and_heading]
        if matched_phrases:
            score += cls.PHRASE_BONUS * len(matched_phrases)
            notes.append(f"phrase match: {', '.join(matched_phrases)}")
            repeated = [
                phrase for phrase in matched_phrases if body_and_heading.count(phrase) > 1
            ]
            if repeated:
                score += cls.REPEATED_PHRASE_BONUS * len(repeated)
                notes.append(f"repeated concept: {', '.join(repeated)}")

        if len(matched_terms) >= 2:
            coverage_ratio = len(matched_terms) / max(len(query.term_weights), 1)
            score += cls.MULTI_TERM_COVERAGE_BONUS * coverage_ratio * len(matched_terms)
            notes.append("multi-term coverage")

        if heading_text in cls.GENERAL_POLICY_HEADINGS:
            score += cls.GENERAL_POLICY_HEADING_BONUS
            notes.append("general policy section")

        if query.expanded_terms:
            expanded_matches = [term for term in query.expanded_terms if term in matched_terms]
            if expanded_matches:
                notes.append(f"query expansion: {', '.join(expanded_matches)}")

        if query.asks_product_alternatives and heading_text == "overview":
            score += cls.PRODUCT_OVERVIEW_BOOST
            notes.append("product overview boost")

        if any(term in heading_text for term in cls.PRODUCT_TERMS):
            if query.has_specific_product or query.expanded_terms:
                score += cls.PRODUCT_SECTION_BOOST
                notes.append("product section boost")
            else:
                score -= cls.PRODUCT_SPECIFIC_PENALTY
                notes.append("product-specific section penalty")

            if query.asks_product_alternatives:
                sibling_matches = [
                    term
                    for term in query.expanded_terms
                    if term in heading_text or term in matched_terms
                ]
                if sibling_matches:
                    score += cls.SIBLING_PRODUCT_SECTION_BOOST
                    notes.append(f"sibling product boost: {', '.join(sorted(sibling_matches))}")

                referenced_matches = [
                    term for term in query.referenced_product_terms if term in heading_text
                ]
                if referenced_matches:
                    score -= cls.REFERENCED_PRODUCT_SECTION_PENALTY
                    notes.append(
                        f"referenced product penalty: {', '.join(sorted(referenced_matches))}"
                    )

        if any(term in matched_terms for term in cls.POLICY_TERMS):
            notes.append("policy terms matched")

        if any(term in matched_terms for term in cls.PRODUCT_TERMS):
            notes.append("product terms matched")

        if not notes:
            notes.append("token overlap")

        return score, matched_terms, "; ".join(notes)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Tokenize text with lightweight normalization for retrieval matching."""
        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
        normalized_tokens: list[str] = []
        for token in raw_tokens:
            normalized = cls._normalize_token(token)
            if len(normalized) <= 2 or normalized in cls.STOPWORDS:
                continue
            normalized_tokens.append(normalized)
        return normalized_tokens

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Apply conservative normalization without hiding the logic."""
        replacements = {
            "products": "product",
            "services": "service",
        }
        if token in replacements:
            return replacements[token]
        if token.endswith("ies") and len(token) > 4:
            return f"{token[:-3]}y"
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            return token[:-1]
        return token

    @classmethod
    def _asks_for_other_products(
        cls,
        normalized_question: str,
        base_terms: list[str],
    ) -> bool:
        if any(term in base_terms for term in cls.ALTERNATIVE_PRODUCT_TERMS):
            return True

        comparative_phrases = (
            "other products than",
            "other product than",
            "besides",
            "other than",
        )
        return any(phrase in normalized_question for phrase in comparative_phrases)
