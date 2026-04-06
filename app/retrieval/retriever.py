import re

from models import DocumentChunk, RetrievedChunk


class KeywordRetriever:
    EXACT_PHRASE_BONUS = 6.0
    HEADING_TERM_BONUS = 2.0
    POLICY_TERM_BONUS = 2.5
    PRODUCT_TERM_BONUS = 1.5
    GENERAL_POLICY_HEADING_BONUS = 3.0
    REPEATED_CONCEPT_BONUS = 1.5
    PRODUCT_SPECIFIC_PENALTY = 2.5
    MULTI_TERM_BONUS = 1.0
    MIN_SCORE = 2.0
    POLICY_TERMS = {"policy", "criteria", "eligibility", "guide", "guideline", "instruction"}
    PRODUCT_TERMS = {"flexline", "invoicebridge", "assetgrow", "demo"}
    GENERAL_POLICY_HEADINGS = {
        "general exclusions",
        "advisor reminder",
        "missing-information rules",
        "escalation flags",
        "response framing",
        "example 2: tax arrears",
    }
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
    }

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        chunks: list[DocumentChunk],
    ) -> list[RetrievedChunk]:
        normalized_question = question.lower()
        question_terms = self._tokenize(question)
        question_phrases = self._extract_phrases(normalized_question)
        question_has_product = any(term in normalized_question for term in self.PRODUCT_TERMS)
        if not question_terms:
            return []

        results: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_terms = set(self._tokenize(chunk.text))
            heading_terms = set(self._tokenize(chunk.section_heading or ""))
            matched_terms = sorted(question_terms & chunk_terms)
            if not matched_terms:
                continue

            score = self._score_chunk(
                question=normalized_question,
                matched_terms=matched_terms,
                chunk=chunk,
                heading_terms=heading_terms,
                question_phrases=question_phrases,
                question_has_product=question_has_product,
            )
            if score < self.MIN_SCORE:
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                    match_summary=self._build_match_summary(
                        question=normalized_question,
                        matched_terms=matched_terms,
                        chunk=chunk,
                        heading_terms=heading_terms,
                        question_phrases=question_phrases,
                        question_has_product=question_has_product,
                    ),
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

    def _score_chunk(
        self,
        question: str,
        matched_terms: list[str],
        chunk: DocumentChunk,
        heading_terms: set[str],
        question_phrases: list[str],
        question_has_product: bool,
    ) -> float:
        score = 0.0
        chunk_text = chunk.text.lower()
        heading_text = (chunk.section_heading or "").lower()

        for term in matched_terms:
            score += 1.0
            if term in heading_terms:
                score += self.HEADING_TERM_BONUS
            if term in self.POLICY_TERMS and term in chunk_text:
                score += self.POLICY_TERM_BONUS
            if term in self.PRODUCT_TERMS and (term in chunk_text or term in heading_text):
                score += self.PRODUCT_TERM_BONUS

        if len(matched_terms) >= 2:
            score += self.MULTI_TERM_BONUS * len(matched_terms)

        for phrase in question_phrases:
            phrase_count = chunk_text.count(phrase) + heading_text.count(phrase)
            if phrase_count:
                score += self.EXACT_PHRASE_BONUS
                if phrase_count > 1:
                    score += self.REPEATED_CONCEPT_BONUS * (phrase_count - 1)

        if heading_text in self.GENERAL_POLICY_HEADINGS:
            score += self.GENERAL_POLICY_HEADING_BONUS

        if not question_has_product and any(term in heading_text for term in self.PRODUCT_TERMS):
            score -= self.PRODUCT_SPECIFIC_PENALTY

        return score

    def _build_match_summary(
        self,
        question: str,
        matched_terms: list[str],
        chunk: DocumentChunk,
        heading_terms: set[str],
        question_phrases: list[str],
        question_has_product: bool,
    ) -> str:
        match_notes: list[str] = []
        chunk_text = chunk.text.lower()
        heading_text = (chunk.section_heading or "").lower()

        matched_phrases = [phrase for phrase in question_phrases if phrase in chunk_text or phrase in heading_text]
        if matched_phrases:
            match_notes.append(f"phrase match: {', '.join(matched_phrases)}")
            repeated_phrases = [
                phrase for phrase in matched_phrases
                if chunk_text.count(phrase) + heading_text.count(phrase) > 1
            ]
            if repeated_phrases:
                match_notes.append(f"repeated concept: {', '.join(repeated_phrases)}")

        heading_matches = [term for term in matched_terms if term in heading_terms]
        if heading_matches:
            match_notes.append(f"heading match: {', '.join(heading_matches)}")

        if heading_text in self.GENERAL_POLICY_HEADINGS:
            match_notes.append("general policy section")

        if any(term in self.POLICY_TERMS for term in matched_terms):
            match_notes.append("policy terms matched")

        if any(term in self.PRODUCT_TERMS for term in matched_terms):
            match_notes.append("product terms matched")

        if not question_has_product and any(term in heading_text for term in self.PRODUCT_TERMS):
            match_notes.append("product-specific section penalty")

        if not match_notes:
            match_notes.append("token overlap")

        return "; ".join(match_notes)

    def _extract_phrases(self, question: str) -> list[str]:
        phrases = [
            "tax arrears",
            "unresolved tax arrears",
            "payment delays",
            "missing financial statements",
            "financial statements",
            "equity ratio",
            "debt to ebitda",
            "invoicebridge demo",
            "flexline demo",
            "assetgrow demo",
        ]
        return [phrase for phrase in phrases if phrase in question]

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {
            token for token in tokens
            if len(token) > 2 and token not in self.STOPWORDS
        }
