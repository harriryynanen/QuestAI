from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Route = Literal["retrieval", "structured", "combined"]
SupportLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class AnswerResponse:
    answer: str
    sources_used: list[str]
    support_level: SupportLevel
    limitations: str
    route: Route


@dataclass(frozen=True)
class DocumentInfo:
    file_name: str
    path: Path
    extension: str
    size_bytes: int


@dataclass(frozen=True)
class StructuredDataInfo:
    dataset_found: bool
    file_name: str | None
    row_count: int | None
    column_names: list[str]
