"""Shared service construction for UI and lightweight API entrypoints."""

from __future__ import annotations

from app.config import AppConfig
from app.llm.client import LLMClient
from app.retrieval.chunker import MarkdownChunker
from app.retrieval.document_store import DocumentStore
from app.retrieval.retriever import KeywordRetriever
from app.services.answer_service import AnswerService
from app.services.router import RuleBasedRouter
from app.structured.customer_data import CustomerDataLoader
from app.structured.planner import StructuredQueryPlanner
from app.structured.query_engine import StructuredQueryEngine


def build_app_services(
    config: AppConfig,
) -> tuple[
    AnswerService,
    DocumentStore,
    CustomerDataLoader,
    MarkdownChunker,
    LLMClient,
]:
    """Construct the current QuestAI service graph used by local entrypoints."""
    router = RuleBasedRouter(
        retrieval_keywords=config.retrieval_keywords,
        structured_keywords=config.structured_keywords,
    )
    document_store = DocumentStore(
        docs_path=config.docs_path,
        pdf_min_text_characters=config.pdf_min_text_characters,
    )
    chunker = MarkdownChunker(max_characters=config.markdown_chunk_max_characters)
    retriever = KeywordRetriever(top_k=config.retrieval_top_k)
    llm_client = _build_llm_client(config)
    customer_data_loader = CustomerDataLoader(
        structured_data_path=config.structured_data_path
    )
    structured_query_engine = StructuredQueryEngine()
    structured_query_planner = StructuredQueryPlanner(llm_client)
    answer_service = AnswerService(
        router=router,
        document_store=document_store,
        customer_data_loader=customer_data_loader,
        chunker=chunker,
        retriever=retriever,
        structured_query_engine=structured_query_engine,
        llm_client=llm_client,
        structured_query_planner=structured_query_planner,
        retrieval_context_max_characters=config.retrieval_context_max_characters,
    )
    return answer_service, document_store, customer_data_loader, chunker, llm_client


def _build_llm_client(config: AppConfig) -> LLMClient:
    """Construct the configured LLM client for planning and synthesis."""
    if config.llm_provider != "openai":
        raise ValueError(
            f"Unsupported LLM provider: {config.llm_provider}. Only 'openai' is currently supported."
        )

    from app.llm.openai_client import OpenAIAppClient

    OpenAIAppClient.ensure_dependency_available()
    return OpenAIAppClient(
        model=config.openai_model,
        enabled=config.llm_enabled_for_retrieval,
        api_key=config.openai_api_key,
    )
