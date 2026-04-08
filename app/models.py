from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Route = Literal["retrieval", "structured", "combined", "unknown"]
SupportLevel = Literal["low", "medium", "high"]
StructuredDatasetName = Literal["customer_portfolio", "advisory_case_pipeline"]
SynthesisStatus = Literal["success", "missing_api_key", "disabled", "api_error", "invalid_response"]
RoutingMethod = Literal["llm", "rules", "safe_fallback"]
RoutingStatus = Literal["success", "missing_api_key", "disabled", "api_error", "invalid_response"]
PlanningMethod = Literal["llm", "heuristic_fallback"]
PlanningStatus = Literal["success", "missing_api_key", "disabled", "api_error", "invalid_response"]
Operation = Literal[
    "fact",
    "filter",
    "comparison",
    "count",
    "list",
    "exists",
    "policy_lookup",
    "product_guidance",
    "preliminary_assessment",
    "unknown",
]
AssessmentBucket = Literal["broadly_aligned", "caution", "not_enough_information"]


@dataclass(frozen=True)
class AnswerResponse:
    answer: str
    sources_used: list[str]
    support_level: SupportLevel
    limitations: str
    route: Route
    retrieved_chunks: list["RetrievedChunk"]
    follow_up_questions: list[str]
    matched_customer_name: str | None = None
    matched_customer_names: list[str] | None = None
    matched_field_name: str | None = None
    matched_field_value: str | None = None
    structured_dataset: StructuredDatasetName | None = None
    synthesis_method: str = "deterministic"
    synthesis_status: SynthesisStatus | None = None
    synthesis_status_message: str | None = None
    routing_method: RoutingMethod = "rules"
    routing_confidence: SupportLevel = "low"
    routing_reason: str | None = None
    planning_method: PlanningMethod | None = None
    planning_reason: str | None = None


@dataclass(frozen=True)
class DocumentInfo:
    file_name: str
    path: Path
    extension: str
    size_bytes: int
    source_type: str | None = None


@dataclass(frozen=True)
class StructuredDataInfo:
    dataset_found: bool
    file_name: str | None
    row_count: int | None
    column_names: list[str]
    dataset_name: StructuredDatasetName | None = None


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    path: Path
    text: str
    source_type: str = "markdown"


@dataclass(frozen=True)
class DocumentLoadIssue:
    file_name: str
    path: Path
    reason: str
    source_type: str


@dataclass(frozen=True)
class DocumentLoadResult:
    documents: list["DocumentRecord"]
    issues: list[DocumentLoadIssue]


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
    score: float
    matched_terms: list[str]
    match_summary: str


@dataclass(frozen=True)
class StructuredQueryResult:
    answer: str
    sources_used: list[str]
    support_level: SupportLevel
    limitations: str
    matched_customer_name: str | None
    matched_customer_names: list[str] | None
    matched_field_name: str | None
    matched_field_value: str | None = None
    planning_method: PlanningMethod = "heuristic_fallback"
    planning_reason: str | None = None


@dataclass(frozen=True)
class RetrievalSynthesisResult:
    answer: str
    support_level: SupportLevel
    limitations: str
    synthesis_method: str
    status: SynthesisStatus
    failure_reason: str | None = None


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    confidence: SupportLevel
    reason: str
    method: RoutingMethod


@dataclass(frozen=True)
class IntentClassificationResult:
    route: Route
    confidence: SupportLevel
    reason: str
    method: RoutingMethod
    status: RoutingStatus
    failure_reason: str | None = None


@dataclass(frozen=True)
class SemanticQueryPlan:
    route: Route
    operation: Operation
    customer_name: str | None
    field_name: str | None
    product_name: str | None
    document_topic: str | None
    comparison_direction: str | None
    filter_value: str | None
    needs_documents: bool
    needs_structured_data: bool
    confidence: SupportLevel
    reason: str
    method: PlanningMethod
    structured_dataset: StructuredDatasetName | None = None


@dataclass(frozen=True)
class SemanticPlanningResult:
    plan: SemanticQueryPlan
    status: PlanningStatus
    failure_reason: str | None = None


@dataclass(frozen=True)
class CombinedEvidence:
    summary: str
    sources_used: list[str]
    missing_information: list[str]


@dataclass(frozen=True)
class CustomerAssessment:
    customer_name: str
    bucket: AssessmentBucket
    reason: str
    sources_used: list[str]
