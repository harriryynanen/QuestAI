import re
from difflib import SequenceMatcher

import pandas as pd

from models import (
    CombinedEvidence,
    CustomerAssessment,
    SemanticQueryPlan,
    StructuredDatasetName,
    StructuredQueryResult,
)


class StructuredQueryEngine:
    ADVISORY_OWNER_ALIASES: tuple[str, ...] = (
        "advisory owner",
        "advisory_owner",
        "who owns",
        "case owner",
    )

    ADVISORY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "case_id": ("case id", "case"),
        "advisory_owner": (
            "advisory owner",
            "advisory_owner",
            "owner",
            "case owner",
            "who owns",
        ),
        "requested_product": ("requested product", "product"),
        "preliminary_status": ("preliminary status", "status", "open"),
        "support_level": ("support level",),
        "missing_information_flags": ("missing information", "missing information flags"),
        "escalation_flag": ("escalation flag", "escalated", "escalation"),
        "next_action": ("next action", "action"),
    }

    FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "latest_revenue_eur": ("turnover", "revenue", "liikevaihto"),
        "ebitda_eur": ("ebitda", "kayttokate"),
        "ebitda_margin_pct": ("ebitda margin", "kayttokateprosentti"),
        "equity_ratio_pct": ("equity ratio", "omavaraisuusaste"),
        "debt_to_ebitda": ("debt to ebitda", "velka suhteessa ebitdaan", "velka suhteessa kayttokatteeseen"),
        "years_in_operation": ("years in operation", "years in business", "toimintavuodet", "kuinka monta vuotta"),
        "b2b_invoicing_pct": ("b2b invoicing share", "b2b share", "invoicing share"),
        "export_sales_pct": ("export sales share", "export share", "vientiosuus"),
        "has_tax_arrears": ("tax arrears", "unresolved tax arrears", "verovelka", "vero velka"),
        "latest_financials_available": (
            "latest financial statements available",
            "financial statements available",
            "latest financial statements",
            "missing financial statements",
            "financial statement",
            "tilinpaatos",
            "tilinpaatokset",
        ),
        "payment_delays_12m": ("payment delays", "repeated payment delays", "maksuviiveet"),
        "largest_customer_share_pct": (
            "largest customer share",
            "customer concentration",
            "largest customer concentration",
            "asiakaskeskittyma",
        ),
        "requested_product_interest": ("interested in", "requested product interest", "kiinnostunut tuotteesta"),
    }

    FILTER_DEFINITIONS: dict[str, tuple[str, str]] = {
        "has_tax_arrears": ("yes", "customers with tax arrears"),
        "payment_delays_12m": ("repeated", "customers with repeated payment delays"),
        "latest_financials_available": ("no", "customers missing latest financial statements"),
    }

    PRODUCT_FIELDS: dict[str, tuple[str, ...]] = {
        "FlexLine Demo": (
            "years_in_operation",
            "equity_ratio_pct",
            "latest_financials_available",
            "has_tax_arrears",
            "payment_delays_12m",
            "ebitda_eur",
        ),
        "InvoiceBridge Demo": (
            "b2b_invoicing_pct",
            "latest_financials_available",
            "has_tax_arrears",
            "largest_customer_share_pct",
            "payment_delays_12m",
            "requested_product_interest",
        ),
        "AssetGrow Demo": (
            "years_in_operation",
            "equity_ratio_pct",
            "latest_financials_available",
            "has_tax_arrears",
            "ebitda_eur",
            "debt_to_ebitda",
        ),
    }

    GENERIC_COMBINED_FIELDS: tuple[str, ...] = (
        "latest_revenue_eur",
        "ebitda_eur",
        "equity_ratio_pct",
        "debt_to_ebitda",
        "years_in_operation",
        "b2b_invoicing_pct",
        "largest_customer_share_pct",
        "has_tax_arrears",
        "latest_financials_available",
        "payment_delays_12m",
        "requested_product_interest",
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
            return self._answer_advisory_case_pipeline(
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
        if dataframe is None or dataset_file_name is None:
            return CombinedEvidence(
                summary="No structured dataset is available.",
                sources_used=[],
                missing_information=["Structured customer data is not available."],
            )

        customer_name = plan.customer_name if plan is not None else None
        product_name = plan.product_name if plan is not None else None
        customer_match = self._match_customer(customer_name or question, dataframe)

        if customer_match["status"] == "ambiguous":
            return CombinedEvidence(
                summary=(
                    "Multiple customer rows could match the question: "
                    f"{', '.join(customer_match['matches'])}."
                ),
                sources_used=[dataset_file_name],
                missing_information=["The customer name is ambiguous."],
            )

        if customer_match["status"] != "single":
            return CombinedEvidence(
                summary="No single customer row could be matched from the structured dataset.",
                sources_used=[dataset_file_name],
                missing_information=["A specific customer match is required for combined analysis."],
            )

        matched_customer = str(customer_match["matches"][0])
        row = dataframe.loc[dataframe["customer_name"] == matched_customer].iloc[0]
        relevant_fields = self.PRODUCT_FIELDS.get(product_name or "", self.GENERIC_COMBINED_FIELDS)

        evidence_lines: list[str] = [f"Customer: {matched_customer}"]
        sources_used: list[str] = []
        missing_information: list[str] = []

        for field_name in relevant_fields:
            if field_name not in dataframe.columns:
                continue

            value = row[field_name]
            label = self._field_label(field_name)
            if pd.isna(value):
                missing_information.append(f"Missing structured value: {label}.")
                continue

            evidence_lines.append(f"{label}: {self._format_value(field_name, value)}")
            sources_used.append(
                f"{dataset_file_name} | row: {matched_customer} | column: {field_name}"
            )

        if product_name:
            evidence_lines.append(f"Requested product interest: {product_name}")
            current_interest = str(row.get("requested_product_interest", ""))
            if current_interest and current_interest.lower() != product_name.lower():
                missing_information.append(
                    f"The dataset shows requested product interest as {current_interest}, not {product_name}."
                )

        if not sources_used:
            missing_information.append("No structured fields could be assembled for the matched customer.")

        return CombinedEvidence(
            summary="\n".join(evidence_lines),
            sources_used=sources_used,
            missing_information=missing_information,
        )

    def build_group_combined_evidence(
        self,
        dataframe: pd.DataFrame | None,
        dataset_file_name: str | None,
        customer_names: list[str],
        product_name: str | None,
    ) -> tuple[list[CustomerAssessment], list[str], list[str]]:
        if dataframe is None or dataset_file_name is None:
            return [], [], ["Structured customer data is not available."]

        product_fields = self.PRODUCT_FIELDS.get(product_name or "", self.GENERIC_COMBINED_FIELDS)
        assessments: list[CustomerAssessment] = []
        merged_sources: list[str] = []
        missing_information: list[str] = []

        for customer_name in customer_names:
            matches = dataframe.loc[dataframe["customer_name"] == customer_name]
            if matches.empty:
                assessments.append(
                    CustomerAssessment(
                        customer_name=customer_name,
                        bucket="not_enough_information",
                        reason="No matching customer row was found in the current dataset.",
                        sources_used=[dataset_file_name],
                    )
                )
                continue

            row = matches.iloc[0]
            assessment = self._assess_customer_alignment(
                row=row,
                dataset_file_name=dataset_file_name,
                product_name=product_name,
                relevant_fields=product_fields,
            )
            assessments.append(assessment)
            for source in assessment.sources_used:
                if source not in merged_sources:
                    merged_sources.append(source)

        if not assessments:
            missing_information.append("No customer rows were available for group assessment.")

        return assessments, merged_sources, missing_information

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

    def _answer_advisory_case_pipeline(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        plan: SemanticQueryPlan | None,
    ) -> StructuredQueryResult:
        if plan is not None and plan.operation != "unknown":
            planned_result = self._execute_advisory_plan(plan, dataframe, dataset_file_name)
            if planned_result is not None:
                return planned_result

        return self._answer_advisory_with_heuristics(question, dataframe, dataset_file_name)

    def _execute_advisory_plan(
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
            if plan.field_name not in {"advisory_owner", "next_action", "requested_product", "support_level", "preliminary_status", "case_id"}:
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
            return self._build_advisory_fact_result(
                row=row,
                field_name=plan.field_name,
                dataset_file_name=dataset_file_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "filter":
            if plan.field_name not in {"escalation_flag", "support_level", "preliminary_status", "requested_product"}:
                return None
            matches = self._filter_advisory_matches(
                dataframe=dataframe,
                field_name=plan.field_name,
                filter_value=plan.filter_value,
                product_name=plan.product_name,
            )
            if matches is None:
                return None
            return self._build_advisory_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                field_name=plan.field_name,
                planning_method=plan.method,
                planning_reason=plan.reason,
            )

        if plan.operation == "count":
            matches = self._filter_advisory_matches(
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

        return None

    def _answer_advisory_with_heuristics(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
    ) -> StructuredQueryResult:
        normalized = self._normalize_text(question)
        field_name = self._match_advisory_field(question)
        customer_match = self._match_customer(question, dataframe)

        count_result = self._try_advisory_count_question(
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

        if customer_match["status"] == "single" and field_name in {"advisory_owner", "next_action", "requested_product", "support_level", "preliminary_status", "case_id"}:
            row = dataframe.loc[dataframe["customer_name"] == customer_match["matches"][0]].iloc[0]
            return self._build_advisory_fact_result(
                row=row,
                field_name=field_name,
                dataset_file_name=dataset_file_name,
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched an advisory case fact question.",
            )

        filter_result = self._try_advisory_filter_question(
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
            limitations="This step currently supports case owner lookups, next actions, simple case filters, and open-case counts by product.",
            matched_customer_name=None,
            matched_customer_names=None,
            matched_field_name=field_name,
            planning_method="heuristic_fallback",
            planning_reason="Heuristic parser could not map the advisory case question safely.",
        )

    def _try_advisory_count_question(
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

    def _try_advisory_filter_question(
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
            return self._build_advisory_filter_result(
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
            return self._build_advisory_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                field_name="support_level",
                planning_method="heuristic_fallback",
                planning_reason="Heuristic parser matched a support-level case filter.",
            )

        return None

    def _build_advisory_fact_result(
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
            planning_method=planning_method,
            planning_reason=planning_reason,
        )

    def _build_advisory_filter_result(
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

    def _filter_advisory_matches(
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

        if field_name in {"support_level", "preliminary_status"}:
            if not filter_value:
                return None
            return dataframe[
                dataframe[field_name].astype(str).str.lower().eq(filter_value.lower())
            ]

        return None

    def _match_advisory_field(self, question: str) -> str | None:
        normalized = self._normalize_text(question)
        if any(
            self._normalize_text(alias) in normalized for alias in self.ADVISORY_OWNER_ALIASES
        ):
            return "advisory_owner"

        best_field: str | None = None
        best_length = -1
        for field_name, aliases in self.ADVISORY_FIELD_ALIASES.items():
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

    def _match_field(self, question: str) -> str | None:
        normalized = self._normalize_text(question)
        best_field: str | None = None
        best_length = -1

        for field_name, aliases in self.FIELD_ALIASES.items():
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
        labels = {
            "latest_revenue_eur": "turnover",
            "ebitda_eur": "EBITDA",
            "ebitda_margin_pct": "EBITDA margin",
            "equity_ratio_pct": "equity ratio",
            "debt_to_ebitda": "debt to EBITDA",
            "years_in_operation": "years in operation",
            "b2b_invoicing_pct": "B2B invoicing share",
            "export_sales_pct": "export sales share",
            "has_tax_arrears": "tax arrears",
            "latest_financials_available": "latest financial statements available",
            "payment_delays_12m": "payment delays",
            "largest_customer_share_pct": "largest customer share",
            "requested_product_interest": "requested product interest",
            "case_id": "case ID",
            "advisory_owner": "advisory owner",
            "requested_product": "requested product",
            "preliminary_status": "preliminary status",
            "support_level": "support level",
            "missing_information_flags": "missing information flags",
            "escalation_flag": "escalation flag",
            "next_action": "next action",
        }
        return labels[field_name]

    def _assess_customer_alignment(
        self,
        row: pd.Series,
        dataset_file_name: str,
        product_name: str | None,
        relevant_fields: tuple[str, ...],
    ) -> CustomerAssessment:
        customer_name = str(row["customer_name"])
        sources_used = [
            f"{dataset_file_name} | row: {customer_name} | column: {field_name}"
            for field_name in relevant_fields
            if field_name in row.index and not pd.isna(row[field_name])
        ]

        financials_value = str(row.get("latest_financials_available", "")).lower()
        tax_arrears_value = str(row.get("has_tax_arrears", "")).lower()
        payment_delays_value = str(row.get("payment_delays_12m", "")).lower()

        missing_required = []
        for field_name in ("latest_financials_available", "equity_ratio_pct", "years_in_operation"):
            if field_name in relevant_fields and (field_name not in row.index or pd.isna(row[field_name])):
                missing_required.append(self._field_label(field_name))

        if financials_value == "no" and "latest_financials_available" in relevant_fields:
            missing_required.append("latest financial statements")

        if missing_required:
            return CustomerAssessment(
                customer_name=customer_name,
                bucket="not_enough_information",
                reason=f"Missing or unavailable key information: {', '.join(sorted(set(missing_required)))}.",
                sources_used=sources_used or [f"{dataset_file_name} | row: {customer_name}"],
            )

        caution_reasons: list[str] = []
        if tax_arrears_value == "yes":
            caution_reasons.append("tax arrears are present")
        if payment_delays_value == "repeated":
            caution_reasons.append("repeated payment delays are visible")
        if financials_value == "no":
            caution_reasons.append("latest financial statements are unavailable")

        if "debt_to_ebitda" in row.index and not pd.isna(row.get("debt_to_ebitda")):
            try:
                if float(row["debt_to_ebitda"]) > 4.0:
                    caution_reasons.append("debt to EBITDA is elevated")
            except (TypeError, ValueError):
                pass

        if caution_reasons:
            return CustomerAssessment(
                customer_name=customer_name,
                bucket="caution",
                reason=", ".join(caution_reasons).capitalize() + ".",
                sources_used=sources_used or [f"{dataset_file_name} | row: {customer_name}"],
            )

        positive_signals: list[str] = []
        if "equity_ratio_pct" in row.index and not pd.isna(row.get("equity_ratio_pct")):
            try:
                if float(row["equity_ratio_pct"]) >= 25.0:
                    positive_signals.append("equity ratio is solid")
            except (TypeError, ValueError):
                pass
        if "years_in_operation" in row.index and not pd.isna(row.get("years_in_operation")):
            try:
                if float(row["years_in_operation"]) >= 3:
                    positive_signals.append("operating history is established")
            except (TypeError, ValueError):
                pass
        if "ebitda_eur" in row.index and not pd.isna(row.get("ebitda_eur")):
            try:
                if float(row["ebitda_eur"]) > 0:
                    positive_signals.append("EBITDA is positive")
            except (TypeError, ValueError):
                pass
        if product_name and str(row.get("requested_product_interest", "")).lower() == product_name.lower():
            positive_signals.append("requested product interest matches")

        if len(positive_signals) >= 2:
            return CustomerAssessment(
                customer_name=customer_name,
                bucket="broadly_aligned",
                reason=", ".join(positive_signals[:3]).capitalize() + ".",
                sources_used=sources_used or [f"{dataset_file_name} | row: {customer_name}"],
            )

        return CustomerAssessment(
            customer_name=customer_name,
            bucket="caution",
            reason="Available signals are mixed and do not support a stronger preliminary view.",
            sources_used=sources_used or [f"{dataset_file_name} | row: {customer_name}"],
        )

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
