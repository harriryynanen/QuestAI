import re

import pandas as pd

from models import StructuredQueryResult


class StructuredQueryEngine:
    FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "latest_revenue_eur": ("turnover", "revenue"),
        "ebitda_eur": ("ebitda",),
        "ebitda_margin_pct": ("ebitda margin", "ebitda margin pct"),
        "equity_ratio_pct": ("equity ratio",),
        "debt_to_ebitda": ("debt to ebitda",),
        "years_in_operation": ("years in operation", "years in business"),
        "b2b_invoicing_pct": ("b2b invoicing share", "b2b share", "invoicing share"),
        "export_sales_pct": ("export sales share", "export share"),
        "has_tax_arrears": ("tax arrears",),
        "latest_financials_available": (
            "latest financial statements available",
            "financial statements available",
            "latest financial statements",
        ),
        "payment_delays_12m": ("payment delays",),
        "largest_customer_share_pct": (
            "largest customer share",
            "customer concentration",
            "largest customer concentration",
        ),
        "requested_product_interest": ("interested in", "requested product interest"),
    }

    FILTER_DEFINITIONS: dict[str, tuple[str, str, str]] = {
        "has_tax_arrears": ("yes", "have tax arrears", "customers with tax arrears"),
        "payment_delays_12m": (
            "repeated",
            "have repeated payment delays",
            "customers with repeated payment delays",
        ),
        "latest_financials_available": (
            "no",
            "are missing latest financial statements",
            "customers missing latest financial statements",
        ),
    }

    def answer(
        self,
        question: str,
        dataframe: pd.DataFrame | None,
        dataset_file_name: str | None,
    ) -> StructuredQueryResult:
        if dataframe is None or dataset_file_name is None:
            return StructuredQueryResult(
                answer="No structured dataset is available for this question.",
                sources_used=[],
                support_level="low",
                limitations="A CSV file must be present and readable before structured questions can be answered.",
                matched_customer_name=None,
                matched_field_name=None,
            )

        field_name = self._match_field(question)
        customer_match = self._match_customer(question, dataframe)

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
                matched_field_name=field_name,
            )

        if customer_match["status"] == "single":
            if field_name is None:
                return StructuredQueryResult(
                    answer="I found the customer, but I could not map the requested field to a supported CSV column.",
                    sources_used=[f"{dataset_file_name} | row: {customer_match['matches'][0]}"],
                    support_level="low",
                    limitations="This step supports only explicitly mapped business fields.",
                    matched_customer_name=customer_match["matches"][0],
                    matched_field_name=None,
                )

            row = dataframe.loc[dataframe["customer_name"] == customer_match["matches"][0]].iloc[0]
            return self._build_fact_result(
                row=row,
                field_name=field_name,
                dataset_file_name=dataset_file_name,
            )

        filter_result = self._try_filter_question(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            field_name=field_name,
        )
        if filter_result is not None:
            return filter_result

        comparison_result = self._try_comparison_question(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            field_name=field_name,
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
                matched_field_name=field_name,
            )

        return StructuredQueryResult(
            answer="I could not map this structured question to a supported deterministic query pattern.",
            sources_used=[dataset_file_name],
            support_level="low",
            limitations="This step supports specific customer facts, a few comparisons, and a few simple filters only.",
            matched_customer_name=None,
            matched_field_name=field_name,
        )

    def _build_fact_result(
        self,
        row: pd.Series,
        field_name: str,
        dataset_file_name: str,
    ) -> StructuredQueryResult:
        customer_name = str(row["customer_name"])
        value = row[field_name]
        label = self._field_label(field_name)

        if pd.isna(value):
            return StructuredQueryResult(
                answer=f"{customer_name} does not have a value for {label} in the current dataset.",
                sources_used=[f"{dataset_file_name} | row: {customer_name}"],
                support_level="low",
                limitations="The dataset row exists, but this field is empty.",
                matched_customer_name=customer_name,
                matched_field_name=field_name,
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
            matched_field_name=field_name,
        )

    def _try_comparison_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        field_name: str | None,
    ) -> StructuredQueryResult | None:
        normalized = question.lower()
        if field_name is None:
            return None
        if field_name not in {
            "latest_revenue_eur",
            "ebitda_eur",
            "equity_ratio_pct",
            "largest_customer_share_pct",
        }:
            return None

        if any(term in normalized for term in ("highest", "largest", "max", "maximum")):
            ranked = dataframe.sort_values(by=field_name, ascending=False)
            descriptor = "highest"
        elif any(term in normalized for term in ("lowest", "smallest", "min", "minimum")):
            ranked = dataframe.sort_values(by=field_name, ascending=True)
            descriptor = "lowest"
        else:
            return None

        if ranked.empty:
            return None

        row = ranked.iloc[0]
        customer_name = str(row["customer_name"])
        formatted_value = self._format_value(field_name, row[field_name])
        label = self._field_label(field_name)

        return StructuredQueryResult(
            answer=f"{customer_name} has the {descriptor} {label}: {formatted_value}.",
            sources_used=[f"{dataset_file_name} | sorted by: {field_name} | row: {customer_name}"],
            support_level="high",
            limitations="This answer is based only on the current CSV values and does not include document context.",
            matched_customer_name=customer_name,
            matched_field_name=field_name,
        )

    def _try_filter_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        dataset_file_name: str,
        field_name: str | None,
    ) -> StructuredQueryResult | None:
        normalized = question.lower()

        if field_name == "requested_product_interest":
            product_name = self._extract_product_name(question, dataframe)
            if product_name is None:
                return StructuredQueryResult(
                    answer="I could not determine which product interest to filter on.",
                    sources_used=[dataset_file_name],
                    support_level="low",
                    limitations="Ask using a product name that appears in the dataset.",
                    matched_customer_name=None,
                    matched_field_name=field_name,
                )

            matches = dataframe[
                dataframe["requested_product_interest"]
                .astype(str)
                .str.lower()
                .eq(product_name.lower())
            ]
            return self._build_filter_result(
                matches=matches,
                dataset_file_name=dataset_file_name,
                filter_label=f"requested product interest = {product_name}",
                field_name=field_name,
            )

        for column_name, (expected_value, answer_phrase, _) in self.FILTER_DEFINITIONS.items():
            if column_name != field_name and answer_phrase not in normalized:
                continue
            if answer_phrase in normalized or (
                field_name == column_name and any(token in normalized for token in ("which customer", "which customers", "who"))
            ):
                matches = dataframe[
                    dataframe[column_name].astype(str).str.lower().eq(expected_value)
                ]
                return self._build_filter_result(
                    matches=matches,
                    dataset_file_name=dataset_file_name,
                    filter_label=answer_phrase,
                    field_name=column_name,
                )

        return None

    def _build_filter_result(
        self,
        matches: pd.DataFrame,
        dataset_file_name: str,
        filter_label: str,
        field_name: str,
    ) -> StructuredQueryResult:
        if matches.empty:
            return StructuredQueryResult(
                answer=f"No customers matched the filter: {filter_label}.",
                sources_used=[f"{dataset_file_name} | filter: {field_name}"],
                support_level="high",
                limitations="This result reflects only the current CSV rows.",
                matched_customer_name=None,
                matched_field_name=field_name,
            )

        customer_names = matches["customer_name"].astype(str).tolist()
        return StructuredQueryResult(
            answer=f"Matching customers: {', '.join(customer_names)}.",
            sources_used=[f"{dataset_file_name} | filter: {field_name}"],
            support_level="high",
            limitations="This result reflects only the current CSV rows and does not include policy interpretation.",
            matched_customer_name=None,
            matched_field_name=field_name,
        )

    def _match_field(self, question: str) -> str | None:
        normalized = question.lower()
        best_field: str | None = None
        best_length = -1

        for field_name, aliases in self.FIELD_ALIASES.items():
            for alias in aliases:
                if alias in normalized and len(alias) > best_length:
                    best_field = field_name
                    best_length = len(alias)

        return best_field

    def _match_customer(self, question: str, dataframe: pd.DataFrame) -> dict[str, str | list[str]]:
        question_normalized = self._normalize_text(question)
        if not question_normalized:
            return {"status": "none", "matches": []}

        customer_names = dataframe["customer_name"].astype(str).tolist()

        exact_matches = [
            name for name in customer_names
            if name.lower() in question.lower()
        ]
        if len(exact_matches) == 1:
            return {"status": "single", "matches": exact_matches}
        if len(exact_matches) > 1:
            return {"status": "ambiguous", "matches": exact_matches}

        normalized_matches = [
            name for name in customer_names
            if self._normalize_text(name) in question_normalized
        ]
        if len(normalized_matches) == 1:
            return {"status": "single", "matches": normalized_matches}
        if len(normalized_matches) > 1:
            return {"status": "ambiguous", "matches": normalized_matches}

        partial_matches = []
        for name in customer_names:
            tokens = [token for token in self._normalize_text(name).split() if len(token) > 2]
            if tokens and all(token in question_normalized.split() for token in tokens[:2]):
                partial_matches.append(name)

        if len(partial_matches) == 1:
            return {"status": "single", "matches": partial_matches}
        if len(partial_matches) > 1:
            return {"status": "ambiguous", "matches": partial_matches}

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
        }
        return labels[field_name]

    def _format_value(self, field_name: str, value: object) -> str:
        if field_name in {
            "latest_revenue_eur",
            "ebitda_eur",
        }:
            return f"EUR {float(value):,.0f}"
        if field_name in {
            "ebitda_margin_pct",
            "equity_ratio_pct",
            "b2b_invoicing_pct",
            "export_sales_pct",
            "largest_customer_share_pct",
        }:
            return f"{float(value):.1f}%"
        if field_name in {"years_in_operation"}:
            return str(int(value))
        return str(value)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _looks_like_customer_fact_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            phrase in normalized
            for phrase in ("what is", "does ", "how many", "'s ")
        )
