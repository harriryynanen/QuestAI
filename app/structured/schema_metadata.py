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

STRUCTURED_FIELD_LABELS: dict[str, str] = {
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

STRUCTURED_FIELD_ALIASES: dict[StructuredDatasetName, dict[str, tuple[str, ...]]] = {
    "customer_portfolio": {
        "latest_revenue_eur": ("turnover", "revenue", "liikevaihto"),
        "ebitda_eur": ("ebitda", "kayttokate"),
        "ebitda_margin_pct": ("ebitda margin", "kayttokateprosentti"),
        "equity_ratio_pct": ("equity ratio", "omavaraisuusaste"),
        "debt_to_ebitda": (
            "debt to ebitda",
            "velka suhteessa ebitdaan",
            "velka suhteessa kayttokatteeseen",
        ),
        "years_in_operation": (
            "years in operation",
            "years in business",
            "toimintavuodet",
            "kuinka monta vuotta",
        ),
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
        "requested_product_interest": (
            "interested in",
            "requested product interest",
            "kiinnostunut tuotteesta",
        ),
    },
    "advisory_case_pipeline": {
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


def get_structured_field_label(field_name: str) -> str:
    return STRUCTURED_FIELD_LABELS[field_name]


def get_structured_field_aliases(
    dataset_name: StructuredDatasetName,
) -> dict[str, tuple[str, ...]]:
    return STRUCTURED_FIELD_ALIASES[dataset_name]


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
