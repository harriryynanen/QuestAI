import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config import build_app_config
from models import RetrievalSynthesisResult, SemanticPlanningResult, SemanticQueryPlan
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.answer_service import AnswerService
from services.router import RuleBasedRouter
from structured.customer_data import CustomerDataLoader
from structured.query_engine import StructuredQueryEngine


class StaticPlanner:
    def __init__(self, planning_result: SemanticPlanningResult) -> None:
        self.planning_result = planning_result

    def plan(
        self,
        question: str,
        conversation_context: str | None = None,
    ) -> SemanticPlanningResult:
        return self.planning_result


class StubOpenAIClient:
    def __init__(
        self,
        retrieval_result: RetrievalSynthesisResult | None = None,
        combined_result: RetrievalSynthesisResult | None = None,
    ) -> None:
        self.retrieval_result = retrieval_result or RetrievalSynthesisResult(
            answer="Fallback retrieval",
            support_level="medium",
            limitations="Fallback retrieval limitations.",
            synthesis_method="fallback",
            status="disabled",
            failure_reason="LLM disabled in tests.",
        )
        self.combined_result = combined_result or RetrievalSynthesisResult(
            answer="Fallback combined",
            support_level="medium",
            limitations="Fallback combined limitations.",
            synthesis_method="fallback",
            status="disabled",
            failure_reason="LLM disabled in tests.",
        )

    def synthesize_retrieval_answer(self, question: str, retrieved_chunks: list) -> RetrievalSynthesisResult:
        return self.retrieval_result

    def synthesize_combined_answer(self, question: str, evidence, document_evidence: list[str]) -> RetrievalSynthesisResult:
        return self.combined_result


def make_plan(
    *,
    route: str,
    operation: str,
    customer_name: str | None = None,
    field_name: str | None = None,
    product_name: str | None = None,
    document_topic: str | None = None,
    comparison_direction: str | None = None,
    filter_value: str | None = None,
    needs_documents: bool = False,
    needs_structured_data: bool = False,
    confidence: str = "high",
    reason: str = "Test plan",
    method: str = "llm",
) -> SemanticQueryPlan:
    return SemanticQueryPlan(
        route=route,
        operation=operation,
        customer_name=customer_name,
        field_name=field_name,
        product_name=product_name,
        document_topic=document_topic,
        comparison_direction=comparison_direction,
        filter_value=filter_value,
        needs_documents=needs_documents,
        needs_structured_data=needs_structured_data,
        confidence=confidence,
        reason=reason,
        method=method,
    )


def make_planning_result(
    plan: SemanticQueryPlan,
    *,
    status: str = "success",
    failure_reason: str | None = None,
) -> SemanticPlanningResult:
    return SemanticPlanningResult(
        plan=plan,
        status=status,
        failure_reason=failure_reason,
    )


@pytest.fixture
def plan_factory():
    return make_plan


@pytest.fixture
def planning_result_factory():
    return make_planning_result


@pytest.fixture
def app_config():
    return build_app_config()


@pytest.fixture
def customer_data_loader(app_config):
    return CustomerDataLoader(app_config.structured_data_path)


@pytest.fixture
def dataframe(customer_data_loader):
    return customer_data_loader.get_dataframe()


@pytest.fixture
def dataset_file_name(customer_data_loader):
    return customer_data_loader.get_dataset_file_name()


@pytest.fixture
def structured_query_engine():
    return StructuredQueryEngine()


@pytest.fixture
def markdown_documents(app_config):
    store = DocumentStore(app_config.docs_path)
    return [
        document
        for document in store.load_retrieval_bundle().documents
        if document.source_type == "markdown"
    ]


@pytest.fixture
def chunks(app_config, markdown_documents):
    return MarkdownChunker(
        max_characters=app_config.markdown_chunk_max_characters
    ).chunk_documents(markdown_documents)


@pytest.fixture
def answer_service_factory(app_config):
    def factory(
        *,
        planning_result: SemanticPlanningResult,
        retrieval_result: RetrievalSynthesisResult | None = None,
        combined_result: RetrievalSynthesisResult | None = None,
    ) -> AnswerService:
        router = RuleBasedRouter(
            retrieval_keywords=app_config.retrieval_keywords,
            structured_keywords=app_config.structured_keywords,
        )
        document_store = DocumentStore(app_config.docs_path)
        customer_data_loader = CustomerDataLoader(app_config.structured_data_path)
        chunker = MarkdownChunker(max_characters=app_config.markdown_chunk_max_characters)
        retriever = KeywordRetriever(top_k=app_config.retrieval_top_k)
        structured_query_engine = StructuredQueryEngine()
        llm_client = StubOpenAIClient(
            retrieval_result=retrieval_result,
            combined_result=combined_result,
        )

        return AnswerService(
            router=router,
            document_store=document_store,
            customer_data_loader=customer_data_loader,
            chunker=chunker,
            retriever=retriever,
            structured_query_engine=structured_query_engine,
            llm_client=llm_client,
            structured_query_planner=StaticPlanner(planning_result),
            retrieval_context_max_characters=app_config.retrieval_context_max_characters,
        )

    return factory
