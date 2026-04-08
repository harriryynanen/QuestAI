import pandas as pd

from app.models import CombinedEvidence, CustomerAssessment, SemanticQueryPlan


class CustomerPortfolioAssessmentHandler:
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

    def __init__(
        self,
        *,
        match_customer,
        format_value,
        field_label,
    ) -> None:
        self._match_customer = match_customer
        self._format_value = format_value
        self._field_label = field_label

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
