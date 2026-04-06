from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    docs_path: Path
    structured_data_path: Path
    markdown_chunk_max_characters: int = 400
    retrieval_top_k: int = 3
    app_title: str = "Business Q&A Assistant"
    app_description: str = (
        "Ask a question about a fictional business banking advisory scenario. "
        "This demo currently uses markdown retrieval and deterministic CSV querying."
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
            "revenue",
            "equity ratio",
            "ebitda",
            "debt to ebitda",
            "years in operation",
            "financial statements",
            "tax arrears",
            "payment delays",
            "largest customer share",
            "customer concentration",
            "highest",
            "largest",
            "missing",
            "lowest",
            "customer",
            "segment",
            "interested in",
        )
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = AppConfig(
    project_root=PROJECT_ROOT,
    docs_path=PROJECT_ROOT / "data" / "docs",
    structured_data_path=PROJECT_ROOT / "data" / "structured",
)
