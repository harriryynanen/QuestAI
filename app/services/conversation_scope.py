from app.models import SemanticPlanningResult, SemanticQueryPlan


class ConversationScopeResolver:
    KNOWN_PRODUCTS = ("AssetGrow Demo", "FlexLine Demo", "InvoiceBridge Demo")

    def resolve_structured_follow_up(
        self,
        question: str,
        conversation_turns: list[dict[str, object]] | None,
    ) -> SemanticPlanningResult | None:
        if not conversation_turns:
            return None

        previous_response = conversation_turns[-1]["response"]
        if previous_response.route != "structured":
            return None

        normalized = question.lower()
        reference_terms = ("those", "them", "they", "that customer", "same ones", "same customer")
        list_terms = ("name", "list", "show", "who are")
        if any(term in normalized for term in list_terms) and any(
            term in normalized for term in reference_terms
        ):
            return SemanticPlanningResult(
                plan=SemanticQueryPlan(
                    route="structured",
                    operation="list",
                    customer_name=None,
                    field_name=None,
                    product_name=None,
                    document_topic=None,
                    comparison_direction=None,
                    filter_value=None,
                    needs_documents=False,
                    needs_structured_data=True,
                    confidence="high",
                    reason="Follow-up references previously identified structured customers.",
                    method="heuristic_fallback",
                ),
                status="success",
                failure_reason=None,
            )

        if (
            previous_response.matched_customer_name
            and "product" in normalized
            and any(term in normalized for term in ("they", "that customer", "the company"))
        ):
            return SemanticPlanningResult(
                plan=SemanticQueryPlan(
                    route="structured",
                    operation="fact",
                    customer_name=previous_response.matched_customer_name,
                    field_name="requested_product_interest",
                    product_name=None,
                    document_topic=None,
                    comparison_direction=None,
                    filter_value=None,
                    needs_documents=False,
                    needs_structured_data=True,
                    confidence="medium",
                    reason="Follow-up refers to the previously matched structured customer.",
                    method="heuristic_fallback",
                ),
                status="success",
                failure_reason=None,
            )

        reverse_lookup_follow_up = self._resolve_advisory_owner_reverse_lookup_follow_up(
            question=question,
            previous_response=previous_response,
        )
        if reverse_lookup_follow_up is not None:
            return reverse_lookup_follow_up

        return None

    def resolve_customer_names_from_context(
        self,
        question: str,
        conversation_turns: list[dict[str, object]] | None,
        dataframe,
    ) -> list[str] | None:
        if not conversation_turns:
            return None

        if dataframe is None or "customer_name" not in dataframe.columns:
            return None

        customer_names = dataframe["customer_name"].astype(str).tolist()
        previous_response = conversation_turns[-1]["response"]
        previous_question = str(conversation_turns[-1]["question"])
        normalized = question.lower()

        if previous_response.route == "structured":
            if any(term in normalized for term in ("those", "them", "they", "same ones")):
                return previous_response.matched_customer_names
            return None

        if previous_response.route == "combined":
            if not (
                self.looks_like_group_reference(question)
                or self.looks_like_elliptical_group_follow_up(question)
            ):
                return None

            if previous_response.matched_customer_names:
                return previous_response.matched_customer_names

            if self.question_implies_full_dataset_scope(previous_question):
                return customer_names

        return None

    def resolve_group_combined_follow_up(
        self,
        question: str,
        conversation_turns: list[dict[str, object]] | None,
    ) -> SemanticPlanningResult | None:
        normalized = question.lower()
        previous_question = None
        previous_response = None
        if conversation_turns:
            previous_turn = conversation_turns[-1]
            previous_question = str(previous_turn["question"])
            previous_response = previous_turn["response"]

        product_name = self.extract_product_name(question)
        if product_name is None and previous_question:
            product_name = self.extract_product_name(previous_question)
        combined_terms = (
            "aligned",
            "fit",
            "suitable",
            "eligible",
            "criteria",
            "preliminary view",
            "service",
            "product",
        )
        scope_terms = ("those customers", "these customers", "listed customers", "those", "these")
        if previous_response is not None and previous_response.route == "combined":
            if product_name and self.looks_like_elliptical_group_follow_up(question):
                return SemanticPlanningResult(
                    plan=SemanticQueryPlan(
                        route="combined",
                        operation="preliminary_assessment",
                        customer_name=None,
                        field_name=None,
                        product_name=product_name,
                        document_topic=f"{product_name} criteria",
                        comparison_direction=None,
                        filter_value=None,
                        needs_documents=True,
                        needs_structured_data=True,
                        confidence="high",
                        reason="Short follow-up reuses the immediately previous grouped combined scope and product context.",
                        method="heuristic_fallback",
                    ),
                    status="success",
                    failure_reason=None,
                )

        if not product_name or not any(term in normalized for term in combined_terms):
            return None
        if not any(term in normalized for term in scope_terms):
            return None

        return SemanticPlanningResult(
            plan=SemanticQueryPlan(
                route="combined",
                operation="preliminary_assessment",
                customer_name=None,
                field_name=None,
                product_name=product_name,
                document_topic=f"{product_name} criteria",
                comparison_direction=None,
                filter_value=None,
                needs_documents=True,
                needs_structured_data=True,
                confidence="high",
                reason="Follow-up references a previously identified customer group for product alignment review.",
                method="heuristic_fallback",
            ),
            status="success",
            failure_reason=None,
        )

    def extract_product_name(self, question: str) -> str | None:
        normalized = question.lower()
        matches = [product for product in self.KNOWN_PRODUCTS if product.lower() in normalized]
        if len(matches) == 1:
            return matches[0]
        return None

    def looks_like_group_reference(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            term in normalized
            for term in (
                "those customers",
                "these customers",
                "listed customers",
                "those companies",
                "these companies",
                "those",
                "these",
            )
        )

    def looks_like_elliptical_group_follow_up(self, question: str) -> bool:
        normalized = question.lower()
        if "which" not in normalized:
            return False

        negative_terms = (
            " are not",
            " not ",
            "not eligible",
            "do not fit",
            "don't fit",
            "not aligned",
            "do not align",
        )
        reference_terms = ("companies", "customers", "ones", "which are", "which companies", "which customers")
        return any(term in normalized for term in negative_terms) and any(
            term in normalized for term in reference_terms
        )

    def question_implies_full_dataset_scope(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            phrase in normalized
            for phrase in (
                "all of the companies",
                "all companies",
                "all of the customers",
                "all customers",
                "companies in the data",
                "customers in the data",
                "companies in data",
                "customers in data",
            )
        )

    def asks_for_negative_group_subset(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            phrase in normalized
            for phrase in (
                "which companies are not",
                "which customers are not",
                "which ones are not",
                "which are not eligible",
                "which companies are not eligible",
                "which customers are not eligible",
                "which ones do not fit",
                "which companies do not fit",
                "which customers do not fit",
                "which ones are not aligned",
            )
        )

    def _resolve_advisory_owner_reverse_lookup_follow_up(
        self,
        question: str,
        previous_response,
    ) -> SemanticPlanningResult | None:
        if previous_response.structured_dataset != "advisory_case_pipeline":
            return None
        if previous_response.matched_field_name != "advisory_owner":
            return None
        if not previous_response.matched_field_value:
            return None
        if not self._looks_like_related_company_follow_up(question):
            return None
        if not self._references_previous_field_value(question, previous_response.matched_field_value):
            return None

        return SemanticPlanningResult(
            plan=SemanticQueryPlan(
                route="structured",
                operation="filter",
                customer_name=previous_response.matched_customer_name,
                field_name="advisory_owner",
                product_name=None,
                document_topic=None,
                comparison_direction=None,
                filter_value=previous_response.matched_field_value,
                needs_documents=False,
                needs_structured_data=True,
                confidence="high",
                reason="Follow-up reuses the previously resolved advisory owner value for a deterministic reverse lookup.",
                method="heuristic_fallback",
                structured_dataset="advisory_case_pipeline",
            ),
            status="success",
            failure_reason=None,
        )

    def _looks_like_related_company_follow_up(self, question: str) -> bool:
        normalized = question.lower()
        return "other compan" in normalized and any(
            phrase in normalized for phrase in ("does ", "what other companies")
        )

    def _references_previous_field_value(
        self,
        question: str,
        field_value: str,
    ) -> bool:
        normalized_question = question.lower()
        normalized_value = field_value.lower()
        if normalized_value in normalized_question:
            return True

        value_tokens = [token for token in normalized_value.split() if len(token) > 2]
        if not value_tokens:
            return False
        return value_tokens[0] in normalized_question
