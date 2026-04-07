import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    docs_path: Path
    structured_data_path: Path
    markdown_chunk_max_characters: int = 400
    pdf_min_text_characters: int = 40
    retrieval_top_k: int = 3
    retrieval_context_max_characters: int = 2400
    llm_enabled_for_retrieval: bool = True
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    app_title: str = "QuestAI"
    app_description: str = (
        "Ask a question about a fictional business banking advisory scenario. "
        "This demo uses grounded document retrieval, deterministic CSV querying, and cautious evidence synthesis."
    )
    example_questions: tuple[str, ...] = (
        "What does the policy say about tax arrears?",
        "What is Harbor Foods Demo Oy's equity ratio?",
        "Which customer has the highest turnover?",
        "Which customers are missing latest financial statements?",
        "Based on the policy and customer data, does Harbor Foods Demo Oy look aligned with InvoiceBridge Demo?",
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


def _get_streamlit_secret(key: str) -> Any | None:
    try:
        streamlit = import_module("streamlit")
    except Exception:
        return None

    try:
        return streamlit.secrets.get(key)
    except Exception:
        return None


def get_runtime_setting(key: str, default: str | None = None) -> str | None:
    secret_value = _get_streamlit_secret(key)
    if secret_value is not None:
        return str(secret_value)

    env_value = os.getenv(key)
    if env_value is not None:
        return env_value

    return default


def get_runtime_bool(key: str, default: bool) -> bool:
    secret_value = _get_streamlit_secret(key)
    if isinstance(secret_value, bool):
        return secret_value
    if secret_value is not None:
        return str(secret_value).lower() == "true"

    env_value = os.getenv(key)
    if env_value is not None:
        return env_value.lower() == "true"

    return default


def build_app_config() -> AppConfig:
    return AppConfig(
        project_root=PROJECT_ROOT,
        docs_path=PROJECT_ROOT / "data" / "docs",
        structured_data_path=PROJECT_ROOT / "data" / "structured",
        llm_enabled_for_retrieval=get_runtime_bool("ENABLE_LLM_FOR_RETRIEVAL", True),
        openai_api_key=get_runtime_setting("OPENAI_API_KEY"),
        openai_model=get_runtime_setting("OPENAI_MODEL", "gpt-5.4-mini") or "gpt-5.4-mini",
    )


DEFAULT_CONFIG = build_app_config()
