import pandas as pd

from app.models import SemanticQueryPlan, StructuredQueryResult
from app.structured.schema_metadata import get_structured_field_aliases


class AdvisoryCaseQueryHandler:
    def __init__(
        self,
        *,
        normalize_text,
        match_customer,
        format_value,
        field_label,
    ) -> None:
        self._normalize_text = normalize_text
        self._match_customer = match_customer
        self._format_value = format_value
        self._field_label = field_label
        self.field_aliases = get_structured_field_aliases("advisory_case_pipeline")
        self.owner_aliases = self.field_aliases["advisory_owner"]

    def answer(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        plan: SemanticQueryPlan | None,
    ) -> StructuredQueryResult:
        if plan is not None and plan.operation != "unknown":
            planned_result = self._execute_plan(plan, dataframe, dataset_file_name)
            if planned_result is not None:
                return planned_result

        return self._answer_with_heuristics(question, dataframe, dataset_file_name)

    def _execute_plan(
        self,
        plan: SemanticQueryPlan,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
    ) -> StructuredQueryResult | None:
        customer_match = (
            self._match_customer(plan.customer_name, dataframe)
            if plan.customer_name
            else {"status": "none", "matches": []}
        )

        if customer_match["status"] == "ambiguous":
            return StructuredQueryResult(
                answer=(
                    "The customer name is ambiguous. Matching case rows: "
                    f"{', '.join(customer_match['matches'])}."
                ),
                sources_used=[dataset_file_name],
                support_level="low",
                limitations="Please ask again with the full customer name.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "fact":
            if plan.field_name not in {
                "advisory_owner",
                "next_action",
                "requested_product",
                "support_level",
                "preliminary_status",
                "case_id",
            }:
                return None
            if customer_match["status"] != "single":
                return StructuredQueryResult(
                    answer="I could not match the customer case safely in the advisory case pipeline.",
                    sources_used=[dataset_file_name],
                    support_level="low",
                    limitations="Ask using the full customer name from the current case pipeline.",
                    matched_customer_name=None,
                    matched_customer_names=None,
                    matched_field_name=plan.field_name,
                    planning_method=plan.method,
                    planning_reason=plan.reason,
                )
            row = dataframe.loc[dataframe["customer_name"] == customer_match["matches"][0]].iloc[0]
            return self._build_fact_result(
                row=row,
                field_name=plan.field_name,
                dataset_file_name=dataset_file_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "filter":
            if plan.field_name not in {
                "escalation_flag",
                "support_level",
                "preliminary_status",
                "requested_product",
                "advisory_owner",
            }:
                return None
            matches = self._filter_matches(
                dataframe=dataframe,
                field_name=plan.field_name,
                filter_value=plan.filter_value,
                product_name=plan.product_name,
            )
            if matches is None:
                return None
            if plan.field_name == "advisory_owner":
                owner_name = plan.filter_value or ""
                anchor_customer = plan.customer_name
                if anchor_customer:
                    matches = matches[
                        ~matches["customer_name"].astype(str).str.lower().eq(anchor_customer.lower())
                    ]
                return self._build_owner_relation_result(
                    matches=matches,
                    owner_name=owner_name,
                    anchor_customer=anchor_customer,
                    dataset_file_name=dataset_file_name,
                    planning_method=plan.method,
                    planning_reason=plan.reason,
                )
            return self._build_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "count":
            matches = self._filter_matches(
                dataframe=dataframe,
                field_name="requested_product",
                filter_value=plan.filter_value,
                product_name=plan.product_name,
            )
            if matches is None:
                matches = dataframe
            if "preliminary_status" in dataframe.columns and (
                plan.filter_value == "open" or "open" in (plan.reason or "").lower()
            ):
                matches = matches[
                    matches["preliminary_status"].astype(str).str.lower().eq("open")
                ]
            if plan.product_name:
                matches = matches[
                    matches["requested_product"].astype(str).str.lower().eq(plan.product_name.lower())
                ]
            return StructuredQueryResult(
                answer=f"The current advisory case pipeline contains {len(matches.index)} matching cases.",
                sources_used=[f"{dataset_file_name} | rows counted"],
                support_level="high",
                limitations="This count reflects only the current advisory case CSV rows.",
                matched_customer_name=None,
                matched_customer_names=matches["customer_name"].astype(str).tolist(),
                matched_field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "list" and plan.field_name == "requested_product":
            return self._build_distinct_product_list_result(
                dataframe=dataframe,
                dataset_file_name=dataset_file_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        return None

    def _answer_with_heuristics(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
    ) -> StructuredQueryResult:
        normalized = self._normalize_text(question)
        field_name = self._match_field(question)
        customer_match = self._match_customer(question, dataframe)

        list_result = self._try_list_question(
            question=normalized,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
        )
        if list_result is not None:
            return list_result

        count_result = self._try_count_question(
            question=normalized,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
        )
        if count_result is not None:
            return count_result

        if customer_match["status"] == "ambiguous":
            return StructuredQueryResult(
                answer=(
                    "The customer name is ambiguous. Matching case rows: "
                    f"{', '.join(customer_match['matches'])}."
                ),
                sources_used=[dataset_file_name],
                support_level="low",
                limitations="Please ask again with the full customer name.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=field_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser found multiple advisory case matches.",
            )

        if customer_match["status"] == "single" and field_name in {
            "advisory_owner",
            "next_action",
            "requested_product",
            "support_level",
            "preliminary_status",
            "case_id",
        }:
            row = dataframe.loc[dataframe["customer_name"] == customer_match["matches"][0]].iloc[0]
            return self._build_fact_result(
                row=row,
                field_name=field_name,
                dataset_file_name=dataset_file_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched an advisory case fact question.",
            )

        filter_result = self._try_filter_question(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            field_name=field_name,
        )
        if filter_result is not None:
            return filter_result

        if customer_match["status"] == "none" and (
            field_name == "advisory_owner"
            or any(phrase in normalized for phrase in ("who owns", "next action", "case"))
        ):
            return StructuredQueryResult(
                answer="I could not match the customer case in the advisory case pipeline.",
                sources_used=[dataset_file_name],
                support_level="low",
                limitations="Try using the full customer name as it appears in the case pipeline CSV.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=field_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser recognized a case-style question but found no safe customer match.",
            )

        return StructuredQueryResult(
            answer="I could not map this advisory case question to a supported deterministic query pattern.",
            sources_used=[dataset_file_name],
            support_level="low",
            limitations="This step currently supports case owner lookups, next actions, simple case filters, open-case counts by product, and distinct product listings.",
            matched_customer_name=None,
            matched_customer_names=None,
            matched_field_name=field_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser could not map the advisory case question safely.",
        )

    def _try_list_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
    ) -> StructuredQueryResult | None:
        asks_for_listing = any(term in question for term in ("what", "which", "list", "show"))
        asks_about_products = "product" in question
        references_case_data = any(
            phrase in question
            for phrase in (
                "in the data",
                "in the advisory case data",
                "in the case pipeline",
                "in the pipeline",
            )
        )

        if not (asks_for_listing and asks_about_products and references_case_data):
            return None

        return self._build_distinct_product_list_result(
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser matched a distinct advisory product listing request.",
        )

    def _try_count_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
    ) -> StructuredQueryResult | None:
        if "how many" not in question or "case" not in question:
            return None

        product_name = self._extract_case_product_name(question, dataframe)
        matches = dataframe
        if "open" in question:
            matches = matches[matches["preliminary_status"].astype(str).str.lower().eq("open")]
        if product_name is not None:
            matches = matches[
                matches["requested_product"].astype(str).str.lower().eq(product_name.lower())
            ]

        if product_name and "open" in question:
            answer = f"There are {len(matches.index)} open {product_name} cases in the current advisory case pipeline."
        else:
            answer = f"The current advisory case pipeline contains {len(matches.index)} matching cases."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | rows counted"],
            support_level="high",
            limitations="This count reflects only the current advisory case CSV rows.",
            matched_customer_name=None,
            matched_customer_names=matches["customer_name"].astype(str).tolist(),
            matched_field_name="requested_product" if product_name else "preliminary_status" if "open" in question else None,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser matched an advisory case count question.",
        )

    def _try_filter_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        field_name: str | None,
    ) -> StructuredQueryResult | None:
        normalized = self._normalize_text(question)
        if "which" not in normalized:
            return None

        if field_name == "escalation_flag" or "escalation flag" in normalized:
            matches = dataframe[
                dataframe["escalation_flag"].astype(str).str.lower().eq("yes")
            ]
            return self._build_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                field_name="escalation_flag",
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched an escalation-flag case filter.",
            )

        if field_name == "support_level" and (
            "not enough information" in normalized
            or "not_enough_information" in question.lower()
        ):
            matches = dataframe[
                dataframe["support_level"].astype(str).str.lower().eq("not_enough_information")
            ]
            return self._build_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                field_name="support_level",
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched a support-level case filter.",
            )

        return None

    def _build_fact_result(
        self,
        row: pd.Series,
        field_name: str,
        dataset_file_name: str,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult:
        customer_name = str(row["customer_name"])
        value = self._format_value(field_name, row[field_name])
        label = self._field_label(field_name)
        if field_name == "advisory_owner":
            answer = f"{customer_name}'s case is owned by {value}."
        elif field_name == "next_action":
            answer = f"The next action for {customer_name}'s case is: {value}."
        else:
            answer = f"{customer_name}'s case {label} is {value}."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | row: {customer_name} | column: {field_name}"],
            support_level="high",
            limitations="This answer is taken directly from the current advisory case CSV row.",
            matched_customer_name=customer_name,
            matched_customer_names=[customer_name],
            matched_field_name=field_name,
            matched_field_value=str(row[field_name]),
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _build_owner_relation_result(
        self,
        matches: pd.DataFrame,
        owner_name: str,
        anchor_customer: str | None,
        dataset_file_name: str,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult:
        case_names = matches["customer_name"].astype(str).tolist()
        if matches.empty:
            answer = f"{owner_name} does not have other companies in the current advisory case pipeline."
        else:
            answer = f"Other companies for {owner_name}: {', '.join(case_names)}."

        limitations = "This result reflects only the current advisory case CSV rows."
        if anchor_customer:
            limitations += f" The previously discussed company, {anchor_customer}, was excluded from the reverse lookup."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | filter: advisory_owner"],
            support_level="high",
            limitations=limitations,
            matched_customer_name=None,
            matched_customer_names=case_names,
            matched_field_name="advisory_owner",
            matched_field_value=owner_name,
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _build_filter_result(
        self,
        matches: pd.DataFrame,
        dataset_file_name: str,
        field_name: str,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult:
        case_names = matches["customer_name"].astype(str).tolist()
        if matches.empty:
            answer = f"No cases matched the filter: {self._field_label(field_name)}."
        else:
            answer = f"Matching cases: {', '.join(case_names)}."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | filter: {field_name}"],
            support_level="high",
            limitations="This result reflects only the current advisory case CSV rows.",
            matched_customer_name=None,
            matched_customer_names=case_names,
            matched_field_name=field_name,
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _build_distinct_product_list_result(
        self,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult:
        products = sorted(
            {
                str(product).strip()
                for product in dataframe["requested_product"].dropna().astype(str).tolist()
                if str(product).strip()
            }
        )
        if not products:
            answer = "No requested products are listed in the current advisory case pipeline."
        else:
            answer = f"Products in the current advisory case pipeline: {', '.join(products)}."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | distinct values: requested_product"],
            support_level="high",
            limitations="This list is taken directly from distinct requested_product values in the current advisory case CSV rows.",
            matched_customer_name=None,
            matched_customer_names=None,
            matched_field_name="requested_product",
            matched_field_value=None,
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _filter_matches(
        self,
        dataframe: pd.DataFrame,
        field_name: str,
        filter_value: str | None,
        product_name: str | None,
    ) -> pd.DataFrame | None:
        if field_name == "requested_product":
            target = product_name or filter_value
            if not target:
                return None
            return dataframe[
                dataframe["requested_product"].astype(str).str.lower().eq(target.lower())
            ]

        if field_name == "escalation_flag":
            target = (filter_value or "yes").lower()
            return dataframe[
                dataframe["escalation_flag"].astype(str).str.lower().eq(target)
            ]

        if field_name == "advisory_owner":
            if not filter_value:
                return None
            return dataframe[
                dataframe["advisory_owner"].astype(str).str.lower().eq(filter_value.lower())
            ]

        if field_name in {"support_level", "preliminary_status"}:
            if not filter_value:
                return None
            return dataframe[
                dataframe[field_name].astype(str).str.lower().eq(filter_value.lower())
            ]

        return None

    def _match_field(self, question: str) -> str | None:
        normalized = self._normalize_text(question)
        if any(
            self._normalize_text(alias) in normalized for alias in self.owner_aliases
        ):
            return "advisory_owner"

        best_field: str | None = None
        best_length = -1
        for field_name, aliases in self.field_aliases.items():
            for alias in aliases:
                normalized_alias = self._normalize_text(alias)
                if normalized_alias in normalized and len(normalized_alias) > best_length:
                    best_field = field_name
                    best_length = len(normalized_alias)
        return best_field

    def _extract_case_product_name(self, question: str, dataframe: pd.DataFrame) -> str | None:
        normalized = question.lower()
        products = dataframe["requested_product"].dropna().astype(str).unique().tolist()
        matches = [product for product in products if product.lower() in normalized]
        if len(matches) == 1:
            return matches[0]
        return None
