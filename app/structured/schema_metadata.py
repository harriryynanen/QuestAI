from models import StructuredDatasetName


STRUCTURED_DATASET_SCHEMAS: dict[StructuredDatasetName, dict[str, object]] = {
    "customer_portfolio": {
        "description": (
            "Customer portfolio facts with one row per customer. Use for financial metrics, "
            "portfolio attributes, and requested product interest."
        ),
        "fields": {
            "latest_revenue_eur": "latest revenue or turnover",
            "ebitda_eur": "EBITDA amount",
            "ebitda_margin_pct": "EBITDA margin percent",
            "equity_ratio_pct": "equity ratio percent",
            "debt_to_ebitda": "debt to EBITDA",
            "years_in_operation": "years in operation",
            "b2b_invoicing_pct": "B2B invoicing share",
            "export_sales_pct": "export sales share",
            "has_tax_arrears": "tax arrears flag",
            "latest_financials_available": "latest financial statements availability",
            "payment_delays_12m": "payment delays over the last 12 months",
            "largest_customer_share_pct": "largest customer share or concentration",
            "requested_product_interest": "requested product interest",
        },
    },
    "advisory_case_pipeline": {
        "description": (
            "Advisory case pipeline facts with one row per case/customer. Use for case ownership, "
            "case status, support level, escalation, next action, and requested product."
        ),
        "fields": {
            "case_id": "case identifier",
            "advisory_owner": "advisory owner or person responsible for the case",
            "requested_product": "requested product in the case",
            "preliminary_status": "case preliminary status such as open",
            "support_level": "case support level",
            "missing_information_flags": "missing information flags",
            "escalation_flag": "escalation flag",
            "next_action": "next action for the case",
        },
    },
}

SUPPORTED_SEMANTIC_FIELD_NAMES: tuple[str, ...] = tuple(
    field_name
    for dataset in STRUCTURED_DATASET_SCHEMAS.values()
    for field_name in dataset["fields"].keys()
)

FIELD_TO_DATASET: dict[str, StructuredDatasetName] = {
    field_name: dataset_name
    for dataset_name, dataset in STRUCTURED_DATASET_SCHEMAS.items()
    for field_name in dataset["fields"].keys()
}


def infer_structured_dataset_from_field(
    field_name: str | None,
) -> StructuredDatasetName | None:
    if field_name is None:
        return None
    return FIELD_TO_DATASET.get(field_name)


def build_structured_schema_prompt_text() -> str:
    sections: list[str] = []
    for dataset_name, dataset in STRUCTURED_DATASET_SCHEMAS.items():
        field_text = ", ".join(
            f"{field_name} ({description})"
            for field_name, description in dataset["fields"].items()
        )
        sections.append(
            f"- {dataset_name}: {dataset['description']} Fields: {field_text}."
        )
    return "Available structured datasets:\n" + "\n".join(sections)
