from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    docs_path: Path
    structured_data_path: Path
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = AppConfig(
    project_root=PROJECT_ROOT,
    docs_path=PROJECT_ROOT / "data" / "docs",
    structured_data_path=PROJECT_ROOT / "data" / "structured",
)
