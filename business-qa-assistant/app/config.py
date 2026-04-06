from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppConfig:
    app_title: str = "Business Q&A Assistant"
    app_description: str = (
        "Ask a question about a fictional business banking advisory scenario. "
        "This demo currently uses simple routing and placeholder answers."
    )
    retrieval_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "policy",
            "guideline",
            "eligibility",
            "criteria",
            "instruction",
            "product",
        )
    )
    structured_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "turnover",
            "equity ratio",
            "ebitda",
            "highest",
            "lowest",
            "customer",
            "segment",
        )
    )


DEFAULT_CONFIG = AppConfig()
