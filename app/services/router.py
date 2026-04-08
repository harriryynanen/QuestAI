from models import Route, RoutingDecision


class RuleBasedRouter:
    ADVISORY_OWNER_QUERY_PHRASES = (
        "who is the advisory owner",
        "advisory owner of",
        "advisory_owner",
        "who owns",
        "owner of",
        "case owner",
        "responsible",
        "responsible for",
        "responsible about",
        "customer responsible",
    )

    DOCUMENT_INTENT_PHRASES = (
        "what does the policy say",
        "what does the guide say",
        "what does the product guide say",
        "what does the eligibility policy say",
        "how does the guide describe",
        "what are the criteria",
        "what are the basic screening criteria",
    )

    DOCUMENT_CONTEXT_KEYWORDS = (
        "policy",
        "product guide",
        "eligibility policy",
        "guideline",
        "guide",
        "instruction",
        "criteria",
        "product",
    )

    STRUCTURED_QUERY_PHRASES = (
        "which customer",
        "which customers",
        "which case",
        "which cases",
        "what is",
        "how many",
        "who owns",
        "who is",
    )

    STRUCTURED_DATA_KEYWORDS = (
        "turnover",
        "revenue",
        "equity ratio",
        "ebitda",
        "debt to ebitda",
        "years in operation",
        "financial statements",
        "tax arrears",
        "payment delays",
        "largest customer share",
        "customer concentration",
        "highest",
        "largest",
        "missing",
        "lowest",
        "interested in",
        "advisory owner",
        "advisory_owner",
        "case owner",
        "customer responsible",
        "responsible",
        "responsible for",
        "owner of",
        "support level",
        "escalation flag",
        "next action",
        "preliminary status",
        "open cases",
    )

    COMBINED_PHRASES = (
        "based on the policy and the customer data",
        "based on the provided documents and data",
        "based on the policy and sample customer data",
        "based on the policy and customer data",
        "using the policy and sample customer data",
        "appear to meet the",
        "preliminary view",
    )

    CUSTOMER_ROW_HINTS = (
        "customer data",
        "sample customer data",
        "customer row",
        "customer portfolio",
        "customer facts",
        "company ",
        "companies ",
        "customers ",
        "customer ",
        " oy",
        " ltd",
        " ab",
    )

    def __init__(
        self,
        retrieval_keywords: tuple[str, ...],
        structured_keywords: tuple[str, ...],
    ) -> None:
        self.retrieval_keywords = retrieval_keywords
        self.structured_keywords = structured_keywords

    def classify(self, question: str) -> RoutingDecision:
        normalized_question = question.lower()

        document_first = self._is_document_first_question(normalized_question)
        structured_first = self._is_structured_question(normalized_question)
        combined = self._is_combined_question(normalized_question)
        matches_retrieval = any(keyword in normalized_question for keyword in self.retrieval_keywords)
        matches_structured = any(keyword in normalized_question for keyword in self.structured_keywords)

        if combined:
            return RoutingDecision(
                route="combined",
                confidence="medium",
                reason="Rule-based router detected explicit cross-source wording.",
                method="rules",
            )
        if document_first:
            return RoutingDecision(
                route="retrieval",
                confidence="high",
                reason="Rule-based router detected document-first intent.",
                method="rules",
            )
        if structured_first:
            return RoutingDecision(
                route="structured",
                confidence="high",
                reason="Rule-based router detected structured data intent.",
                method="rules",
            )
        if matches_retrieval and not matches_structured:
            return RoutingDecision(
                route="retrieval",
                confidence="medium",
                reason="Rule-based router found retrieval-oriented keywords.",
                method="rules",
            )
        if matches_structured and not matches_retrieval:
            return RoutingDecision(
                route="structured",
                confidence="medium",
                reason="Rule-based router found structured-data keywords.",
                method="rules",
            )
        if matches_retrieval and matches_structured:
            return RoutingDecision(
                route="retrieval",
                confidence="low",
                reason="Rule-based router found mixed keywords and defaulted to the more useful single-source retrieval path.",
                method="rules",
            )
        return RoutingDecision(
            route="unknown",
            confidence="low",
            reason="Rule-based router could not route the question confidently.",
            method="rules",
        )

    def _is_document_first_question(self, question: str) -> bool:
        if any(phrase in question for phrase in self.DOCUMENT_INTENT_PHRASES):
            return True

        has_document_context = any(
            keyword in question for keyword in self.DOCUMENT_CONTEXT_KEYWORDS
        )
        has_customer_row_signal = any(
            hint in question for hint in self.CUSTOMER_ROW_HINTS
        )
        return has_document_context and not has_customer_row_signal

    def _is_structured_question(self, question: str) -> bool:
        has_structured_phrase = any(
            phrase in question for phrase in self.STRUCTURED_QUERY_PHRASES
        )
        has_structured_data_keyword = any(
            keyword in question for keyword in self.STRUCTURED_DATA_KEYWORDS
        )
        has_advisory_owner_signal = any(
            phrase in question for phrase in self.ADVISORY_OWNER_QUERY_PHRASES
        )
        has_customer_row_signal = any(
            hint in question for hint in self.CUSTOMER_ROW_HINTS
        )
        return (
            has_structured_data_keyword
            and (has_structured_phrase or has_customer_row_signal or has_advisory_owner_signal)
        )

    def _is_combined_question(self, question: str) -> bool:
        has_combined_phrase = any(
            phrase in question for phrase in self.COMBINED_PHRASES
        )
        asks_fit_view = (
            ("fit for" in question or "aligned with" in question or "meet the" in question)
            and any(hint in question for hint in self.CUSTOMER_ROW_HINTS)
            and any(keyword in question for keyword in self.DOCUMENT_CONTEXT_KEYWORDS)
        )
        return has_combined_phrase or asks_fit_view


def is_confident_routing_decision(decision: RoutingDecision) -> bool:
    return decision.route != "unknown" and decision.confidence in {"medium", "high"}
