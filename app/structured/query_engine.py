import re
from difflib import SequenceMatcher

import pandas as pd

from app.models import (
    CombinedEvidence,
    CustomerAssessment,
    SemanticQueryPlan,
    StructuredDatasetName,
    StructuredQueryResult,
)
from app.structured.advisory_case_handler import AdvisoryCaseQueryHandler
from app.structured.customer_portfolio_assessment import CustomerPortfolioAssessmentHandler
from app.structured.schema_metadata import (
    get_structured_field_aliases,
    get_structured_field_label,
)


class StructuredQueryEngine:
    FILTER_DEFINITIONS: dict[str, tuple[str, str]] = {
        "has_tax_arrears": ("yes", "customers with tax arrears"),
        "payment_delays_12m": ("repeated", "customers with repeated payment delays"),
        "latest_financials_available": ("no", "customers missing latest financial statements"),
    }

    def __init__(self) -> None:
        self.field_aliases = get_structured_field_aliases("customer_portfolio")
        self.advisory_case_handler = AdvisoryCaseQueryHandler(
            normalize_text=self._normalize_text,
            match_customer=self._match_customer,
            format_value=self._format_value,
            field_label=self._field_label,
        )
        self.customer_portfolio_assessment_handler = CustomerPortfolioAssessmentHandler(
            match_customer=self._match_customer,
            format_value=self._format_value,
            field_label=self._field_label,
        )

    def answer(
        self,
        question: str,
        dataframe: pd.DataFrame | None,
        dataset_file_name: str | None,
        plan: SemanticQueryPlan | None = None,
        resolved_customer_names: list[str] | None = None,
        dataset_name: StructuredDatasetName = "customer_portfolio",
    ) -> StructuredQueryResult:
        if dataframe is None or dataset_file_name is None:
            return StructuredQueryResult(
                answer="No structured dataset is available for this question.",
                sources_used=[],
                support_level="low",
                limitations="A CSV file must be present and readable before structured questions can be answered.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=None,
                planning_method=plan.method if plan is not None else "heuristic_fallback",
                planning_reason=plan.reason if plan is not None else None,
            )

        if dataset_name == "advisory_case_pipeline":
            return self.advisory_case_handler.answer(
                question=question,
                dataframe=dataframe,
                dataset_file_name=dataset_file_name,
                plan=plan,
            )

        if plan is not None and plan.operation != "unknown":
            planned_result = self._execute_plan(
                plan,
                dataframe,
                dataset_file_name,
                resolved_customer_names=resolved_customer_names,
            )
            if planned_result is not None:
                return planned_result

        return self._answer_with_heuristics(
            question,
            dataframe,
            dataset_file_name,
            resolved_customer_names=resolved_customer_names,
        )

    def build_combined_evidence(
        self,
        question: str,
        dataframe: pd.DataFrame | None,
        dataset_file_name: str | None,
        plan: SemanticQueryPlan | None = None,
    ) -> CombinedEvidence:
        return self.customer_portfolio_assessment_handler.build_combined_evidence(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            plan=plan,
        )

    def build_group_combined_evidence(
        self,
        dataframe: pd.DataFrame | None,
        dataset_file_name: str | None,
        customer_names: list[str],
        product_name: str | None,
    ) -> tuple[list[CustomerAssessment], list[str], list[str]]:
        return self.customer_portfolio_assessment_handler.build_group_combined_evidence(
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            customer_names=customer_names,
            product_name=product_name,
        )

    def _answer_with_heuristics(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        resolved_customer_names: list[str] | None = None,
    ) -> StructuredQueryResult:
        normalized = self._normalize_text(question)
        field_name = self._match_field(question)
        customer_match = self._match_customer(question, dataframe)

        count_result = self._try_count_question(
            question=normalized,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
        )
        if count_result is not None:
            return count_result

        list_result = self._try_list_question(
            question=normalized,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            resolved_customer_names=resolved_customer_names,
        )
        if list_result is not None:
            return list_result

        exists_result = self._try_exists_question(
            question=normalized,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            field_name=field_name,
        )
        if exists_result is not None:
            return exists_result

        if customer_match["status"] == "ambiguous":
            return StructuredQueryResult(
                answer=(
                    "The customer name is ambiguous. Matching rows: "
                    f"{', '.join(customer_match['matches'])}."
                ),
                sources_used=[dataset_file_name],
                support_level="low",
                limitations="Please ask again with the full customer name.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=field_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser found multiple customer matches.",
            )

        if customer_match["status"] == "single":
            if field_name is None:
                return StructuredQueryResult(
                    answer="I found the customer, but I could not map the requested field to a supported CSV column.",
                    sources_used=[f"{dataset_file_name} | row: {customer_match['matches'][0]}"],
                    support_level="low",
                    limitations="This step supports only explicitly mapped business fields.",
                    matched_customer_name=str(customer_match["matches"][0]),
                    matched_customer_names=[str(customer_match["matches"][0])],
                    matched_field_name=None,
                    planning_method="heuristic_fallback",
                    planning_reason="Heuristic parser matched a customer but not a supported field.",
                )

            row = dataframe.loc[dataframe["customer_name"] == customer_match["matches"][0]].iloc[0]
            return self._build_fact_result(
                row=row,
                field_name=field_name,
                dataset_file_name=dataset_file_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched a customer fact question.",
            )

        filter_result = self._try_filter_question(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            field_name=field_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser matched a filter question.",
        )
        if filter_result is not None:
            return filter_result

        comparison_result = self._try_comparison_question(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            field_name=field_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser matched a comparison question.",
        )
        if comparison_result is not None:
            return comparison_result

        if customer_match["status"] == "none" and self._looks_like_customer_fact_question(question):
            return StructuredQueryResult(
                answer="I could not match the customer name in the structured dataset.",
                sources_used=[dataset_file_name],
                support_level="low",
                limitations="Try using the full customer name as it appears in the CSV.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=field_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser recognized a customer-style question but found no safe match.",
            )

        return StructuredQueryResult(
            answer="I could not map this structured question to a supported deterministic query pattern.",
            sources_used=[dataset_file_name],
            support_level="low",
            limitations="This step supports customer facts, simple comparisons, filters, counts, and existence checks only.",
            matched_customer_name=None,
            matched_customer_names=None,
            matched_field_name=field_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser could not map the question safely.",
        )

    def _execute_plan(
        self,
        plan: SemanticQueryPlan,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        resolved_customer_names: list[str] | None = None,
    ) -> StructuredQueryResult | None:
        customer_match = (
            self._match_customer(plan.customer_name, dataframe)
            if plan.customer_name
            else {"status": "none", "matches": []}
        )

        if customer_match["status"] == "ambiguous":
            return StructuredQueryResult(
                answer=(
                    "The customer name is ambiguous. Matching rows: "
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

        if plan.operation == "count":
            return StructuredQueryResult(
                answer=f"The current dataset contains {len(dataframe.index)} customers.",
                sources_used=[f"{dataset_file_name} | rows counted"],
                support_level="high",
                limitations="This count reflects only the current CSV rows.",
                matched_customer_name=None,
                matched_customer_names=dataframe["customer_name"].astype(str).tolist(),
                matched_field_name=None,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "list":
            names_to_list = resolved_customer_names
            if names_to_list is None:
                names_to_list = dataframe["customer_name"].astype(str).tolist()

            if not names_to_list:
                return StructuredQueryResult(
                    answer="I could not determine which customers you mean from the recent conversation.",
                    sources_used=[dataset_file_name],
                    support_level="low",
                    limitations="The follow-up reference was ambiguous or the referenced result set was empty.",
                    matched_customer_name=None,
                    matched_customer_names=[],
                    matched_field_name=None,
                    planning_method=plan.method,
                    planning_reason=plan.reason,
                )

            return StructuredQueryResult(
                answer=f"Customer names in scope: {', '.join(names_to_list)}.",
                sources_used=[f"{dataset_file_name} | listed rows"],
                support_level="high",
                limitations="This list is taken directly from the current CSV rows in scope.",
                matched_customer_name=names_to_list[0] if len(names_to_list) == 1 else None,
                matched_customer_names=names_to_list,
                matched_field_name=None,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "exists":
            if plan.field_name is None:
                return None
            matches = self._filter_matches(
                dataframe=dataframe,
                field_name=plan.field_name,
                filter_value=plan.filter_value,
                product_name=plan.product_name,
            )
            if matches is None:
                return None
            if matches.empty:
                answer = "No. No matching customers were found in the current dataset."
            else:
                answer = f"Yes. Matching customers: {', '.join(matches['customer_name'].astype(str).tolist())}."
            return StructuredQueryResult(
                answer=answer,
                sources_used=[f"{dataset_file_name} | exists check: {plan.field_name}"],
                support_level="high",
                limitations="This answer reflects only the current CSV rows.",
                matched_customer_name=None,
                matched_customer_names=matches["customer_name"].astype(str).tolist(),
                matched_field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "fact":
            if plan.field_name is None or customer_match["status"] != "single":
                return None
            if plan.field_name not in dataframe.columns:
                return StructuredQueryResult(
                    answer="I found the customer, but the planned field does not exist in the selected structured dataset.",
                    sources_used=[f"{dataset_file_name} | row: {customer_match['matches'][0]}"],
                    support_level="low",
                    limitations="The semantic plan pointed to a field that is not available in this CSV dataset.",
                    matched_customer_name=str(customer_match["matches"][0]),
                    matched_customer_names=[str(customer_match["matches"][0])],
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
            if plan.field_name is None:
                return None
            matches = self._filter_matches(
                dataframe=dataframe,
                field_name=plan.field_name,
                filter_value=plan.filter_value,
                product_name=plan.product_name,
            )
            if matches is None:
                return None
            filter_label = self._field_label(plan.field_name)
            if plan.product_name and plan.field_name == "requested_product_interest":
                filter_label = f"requested product interest = {plan.product_name}"
            return self._build_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                filter_label=filter_label,
                field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "comparison":
            if plan.field_name is None:
                return None
            if plan.field_name not in {
                "latest_revenue_eur",
                "ebitda_eur",
                "equity_ratio_pct",
                "largest_customer_share_pct",
            }:
                return None
            direction = plan.comparison_direction or "highest"
            ranked = dataframe.sort_values(by=plan.field_name, ascending=direction == "lowest")
            if ranked.empty:
                return None
            row = ranked.iloc[0]
            customer_name = str(row["customer_name"])
            label = self._field_label(plan.field_name)
            value = self._format_value(plan.field_name, row[plan.field_name])
            return StructuredQueryResult(
                answer=f"{customer_name} has the {direction} {label}: {value}.",
                sources_used=[f"{dataset_file_name} | sorted by: {plan.field_name} | row: {customer_name}"],
                support_level="high",
                limitations="This answer is based only on the current CSV values and does not include document context.",
                matched_customer_name=customer_name,
                matched_customer_names=[customer_name],
                matched_field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
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
        if field_name not in row.index:
            return StructuredQueryResult(
                answer="I found the customer, but the requested field does not exist in the selected structured dataset.",
                sources_used=[f"{dataset_file_name} | row: {customer_name}"],
                support_level="low",
                limitations="This deterministic step can only answer fields that are present in the current CSV row.",
                matched_customer_name=customer_name,
                matched_customer_names=[customer_name],
                matched_field_name=field_name,
                planning_method=planning_method,
                planning_reason=planning_reason,
            )
        value = row[field_name]
        label = self._field_label(field_name)

        if pd.isna(value):
            return StructuredQueryResult(
                answer=f"{customer_name} does not have a value for {label} in the current dataset.",
                sources_used=[f"{dataset_file_name} | row: {customer_name}"],
                support_level="low",
                limitations="The dataset row exists, but this field is empty.",
                matched_customer_name=customer_name,
                matched_customer_names=[customer_name],
                matched_field_name=field_name,
                planning_method=planning_method,
                planning_reason=planning_reason,
            )

        formatted_value = self._format_value(field_name, value)
        if field_name in {"has_tax_arrears", "latest_financials_available"}:
            answer = f"{customer_name}: {label} = {formatted_value}."
        else:
            answer = f"{customer_name} has {label} of {formatted_value}."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | row: {customer_name} | column: {field_name}"],
            support_level="high",
            limitations="This answer is taken directly from the CSV and does not include document-based interpretation.",
            matched_customer_name=customer_name,
            matched_customer_names=[customer_name],
            matched_field_name=field_name,
            matched_field_value=str(value),
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _try_count_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
    ) -> StructuredQueryResult | None:
        count_phrases = (
            "how many customers",
            "customer count",
            "kuinka monta asiakasta",
            "asiakasta minulla on",
        )
        if any(phrase in question for phrase in count_phrases):
            return StructuredQueryResult(
                answer=f"The current dataset contains {len(dataframe.index)} customers.",
                sources_used=[f"{dataset_file_name} | rows counted"],
                support_level="high",
                limitations="This count reflects only the current CSV rows.",
                matched_customer_name=None,
                matched_customer_names=dataframe["customer_name"].astype(str).tolist(),
                matched_field_name=None,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched a count question.",
            )
        return None

    def _try_list_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        resolved_customer_names: list[str] | None,
    ) -> StructuredQueryResult | None:
        list_phrases = (
            "name those customers",
            "name them",
            "name those",
            "list the customers",
            "list customers",
            "show the customer names",
            "show customer names",
            "who are they",
            "name those customer",
        )
        if not any(phrase in question for phrase in list_phrases):
            return None

        has_reference = any(term in question for term in ("those", "them", "they", "same"))
        if has_reference and resolved_customer_names is None:
            return StructuredQueryResult(
                answer="I could not determine which customers you mean from the recent conversation.",
                sources_used=[dataset_file_name],
                support_level="low",
                limitations="Please restate which customer group you want listed.",
                matched_customer_name=None,
                matched_customer_names=None,
                matched_field_name=None,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser detected a follow-up list request without a safe reference.",
            )

        names = resolved_customer_names or dataframe["customer_name"].astype(str).tolist()
        return StructuredQueryResult(
            answer=f"Customer names in scope: {', '.join(names)}.",
            sources_used=[f"{dataset_file_name} | listed rows"],
            support_level="high",
            limitations="This list is taken directly from the current CSV rows in scope.",
            matched_customer_name=names[0] if len(names) == 1 else None,
            matched_customer_names=names,
            matched_field_name=None,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser matched a customer listing request.",
        )

    def _try_exists_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        field_name: str | None,
    ) -> StructuredQueryResult | None:
        exists_phrases = (
            "does any customer",
            "is there any customer",
            "is there any",
            "any customer",
            "onko",
            "loytyyko",
        )
        if not any(phrase in question for phrase in exists_phrases):
            return None
        if field_name is None:
            return None

        matches = self._filter_matches(
            dataframe=dataframe,
            field_name=field_name,
            filter_value=None,
            product_name=None,
        )
        if matches is None:
            return None

        if matches.empty:
            answer = "No. No matching customers were found in the current dataset."
        else:
            answer = f"Yes. Matching customers: {', '.join(matches['customer_name'].astype(str).tolist())}."

        return StructuredQueryResult(
            answer=answer,
            sources_used=[f"{dataset_file_name} | exists check: {field_name}"],
            support_level="high",
            limitations="This answer reflects only the current CSV rows.",
            matched_customer_name=None,
            matched_customer_names=matches["customer_name"].astype(str).tolist(),
            matched_field_name=field_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser matched an existence question.",
        )

    def _try_comparison_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        field_name: str | None,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult | None:
        normalized = self._normalize_text(question)
        if field_name is None:
            return None
        if field_name not in {
            "latest_revenue_eur",
            "ebitda_eur",
            "equity_ratio_pct",
            "largest_customer_share_pct",
        }:
            return None

        if any(term in normalized for term in ("highest", "largest", "max", "maximum", "suurin", "korkein")):
            descriptor = "highest"
            ranked = dataframe.sort_values(by=field_name, ascending=False)
        elif any(term in normalized for term in ("lowest", "smallest", "min", "minimum", "pienin", "matalin")):
            descriptor = "lowest"
            ranked = dataframe.sort_values(by=field_name, ascending=True)
        else:
            return None

        if ranked.empty:
            return None

        row = ranked.iloc[0]
        customer_name = str(row["customer_name"])
        label = self._field_label(field_name)
        value = self._format_value(field_name, row[field_name])
        return StructuredQueryResult(
            answer=f"{customer_name} has the {descriptor} {label}: {value}.",
            sources_used=[f"{dataset_file_name} | sorted by: {field_name} | row: {customer_name}"],
            support_level="high",
            limitations="This answer is based only on the current CSV values and does not include document context.",
            matched_customer_name=customer_name,
            matched_customer_names=[customer_name],
            matched_field_name=field_name,
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _try_filter_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        field_name: str | None,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult | None:
        normalized = self._normalize_text(question)

        if field_name == "requested_product_interest":
            product_name = self._extract_product_name(question, dataframe)
            if product_name is None:
                return StructuredQueryResult(
                    answer="I could not determine which product interest to filter on.",
                    sources_used=[dataset_file_name],
                    support_level="low",
                    limitations="Ask using a product name that appears in the dataset.",
                    matched_customer_name=None,
                    matched_customer_names=None,
                    matched_field_name=field_name,
                    planning_method=planning_method,
                    planning_reason=planning_reason,
                )

            matches = dataframe[
                dataframe["requested_product_interest"].astype(str).str.lower().eq(product_name.lower())
            ]
            return self._build_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                filter_label=f"requested product interest = {product_name}",
                field_name=field_name,
                planning_method=planning_method,
                planning_reason=planning_reason,
            )

        listing_terms = ("which customer", "which customers", "who", "which firms", "which companies", "mitka asiakkaat")
        for column_name, (expected_value, answer_phrase) in self.FILTER_DEFINITIONS.items():
            if column_name != field_name and answer_phrase not in normalized:
                continue
            if answer_phrase in normalized or (
                field_name == column_name and any(term in normalized for term in listing_terms)
            ):
                matches = dataframe[dataframe[column_name].astype(str).str.lower().eq(expected_value)]
                return self._build_filter_result(
                    matches=matches,
                    dataset_file_name=dataset_file_name,
                    filter_label=answer_phrase,
                    field_name=column_name,
                    planning_method=planning_method,
                    planning_reason=planning_reason,
                )

        return None

    def _build_filter_result(
        self,
        matches: pd.DataFrame,
        dataset_file_name: str,
        filter_label: str,
        field_name: str,
        planning_method: str,
        planning_reason: str | None,
    ) -> StructuredQueryResult:
        if matches.empty:
            return StructuredQueryResult(
                answer=f"No customers matched the filter: {filter_label}.",
                sources_used=[f"{dataset_file_name} | filter: {field_name}"],
                support_level="high",
                limitations="This result reflects only the current CSV rows.",
                matched_customer_name=None,
                matched_customer_names=[],
                matched_field_name=field_name,
                planning_method=planning_method,
                planning_reason=planning_reason,
            )

        names = matches["customer_name"].astype(str).tolist()
        return StructuredQueryResult(
            answer=f"Matching customers: {', '.join(names)}.",
            sources_used=[f"{dataset_file_name} | filter: {field_name}"],
            support_level="high",
            limitations="This result reflects only the current CSV rows and does not include policy interpretation.",
            matched_customer_name=None,
            matched_customer_names=names,
            matched_field_name=field_name,
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
        if field_name == "requested_product_interest":
            target = product_name or filter_value
            if not target:
                return None
            return dataframe[
                dataframe["requested_product_interest"].astype(str).str.lower().eq(target.lower())
            ]

        if field_name in self.FILTER_DEFINITIONS:
            target = (filter_value or self.FILTER_DEFINITIONS[field_name][0]).lower()
            return dataframe[dataframe[field_name].astype(str).str.lower().eq(target)]

        return None

    def _match_field(self, question: str) -> str | None:
        normalized = self._normalize_text(question)
        best_field: str | None = None
        best_length = -1

        for field_name, aliases in self.field_aliases.items():
            for alias in aliases:
                normalized_alias = self._normalize_text(alias)
                if normalized_alias in normalized and len(normalized_alias) > best_length:
                    best_field = field_name
                    best_length = len(normalized_alias)

        return best_field

    def _match_customer(self, question: str | None, dataframe: pd.DataFrame) -> dict[str, str | list[str]]:
        if not question:
            return {"status": "none", "matches": []}

        question_normalized = self._normalize_text(question)
        if not question_normalized:
            return {"status": "none", "matches": []}

        question_tokens = set(question_normalized.split())
        collapsed_question = question_normalized.replace(" ", "")
        customer_names = dataframe["customer_name"].astype(str).tolist()

        exact_matches = [
            name for name in customer_names if self._normalize_text(name) in question_normalized
        ]
        if len(exact_matches) == 1:
            return {"status": "single", "matches": exact_matches}
        if len(exact_matches) > 1:
            return {"status": "ambiguous", "matches": exact_matches}

        tolerant_matches: list[str] = []
        ignored_tokens = {"oy", "ltd", "ab", "demo", "sample", "mock", "test"}

        for name in customer_names:
            normalized_name = self._normalize_text(name)
            collapsed_name = normalized_name.replace(" ", "")
            name_tokens = [token for token in normalized_name.split() if len(token) > 2 and token not in ignored_tokens]

            if collapsed_name and collapsed_name in collapsed_question:
                tolerant_matches.append(name)
                continue

            if name_tokens and all(token in question_tokens for token in name_tokens[:2]):
                tolerant_matches.append(name)
                continue

            if name_tokens:
                matched_token_count = sum(1 for token in name_tokens if token in question_tokens)
                if matched_token_count >= max(1, len(name_tokens) - 1):
                    tolerant_matches.append(name)
                    continue

            similarity = SequenceMatcher(None, collapsed_name, collapsed_question).ratio()
            if similarity >= 0.84 and len(collapsed_name) >= 8:
                tolerant_matches.append(name)

        unique_matches = sorted(set(tolerant_matches))
        if len(unique_matches) == 1:
            return {"status": "single", "matches": unique_matches}
        if len(unique_matches) > 1:
            return {"status": "ambiguous", "matches": unique_matches}

        return {"status": "none", "matches": []}

    def _extract_product_name(self, question: str, dataframe: pd.DataFrame) -> str | None:
        normalized = question.lower()
        products = dataframe["requested_product_interest"].dropna().astype(str).unique().tolist()
        matches = [product for product in products if product.lower() in normalized]
        if len(matches) == 1:
            return matches[0]
        return None

    def _field_label(self, field_name: str) -> str:
        return get_structured_field_label(field_name)

    def _format_value(self, field_name: str, value: object) -> str:
        if field_name in {"latest_revenue_eur", "ebitda_eur"}:
            return f"EUR {float(value):,.0f}"
        if field_name == "requested_amount_eur":
            return f"EUR {float(value):,.0f}"
        if field_name in {
            "ebitda_margin_pct",
            "equity_ratio_pct",
            "b2b_invoicing_pct",
            "export_sales_pct",
            "largest_customer_share_pct",
        }:
            return f"{float(value):.1f}%"
        if field_name == "years_in_operation":
            return str(int(value))
        return str(value)

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower()
        translation = str.maketrans({"ä": "a", "ö": "o", "å": "a", "é": "e"})
        lowered = lowered.translate(translation)
        lowered = re.sub(r"\b([a-z0-9]+):n\b", r"\1", lowered)
        return re.sub(r"[^a-z0-9]+", " ", lowered).strip()

    def _looks_like_customer_fact_question(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        fact_phrases = ("what is", "does ", "how many", "mika on", "onko", "which customer")
        return any(phrase in normalized for phrase in fact_phrases)
