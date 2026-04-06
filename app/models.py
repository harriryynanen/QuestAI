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
    retrieved_chunks: list["RetrievedChunk"]
    matched_customer_name: str | None = None
    matched_field_name: str | None = None


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


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    path: Path
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    file_name: str
    text: str
    section_heading: str | None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: int
    matched_terms: list[str]


@dataclass(frozen=True)
class StructuredQueryResult:
    answer: str
    sources_used: list[str]
    support_level: SupportLevel
    limitations: str
    matched_customer_name: str | None
    matched_field_name: str | None
