"""Streamlit entrypoint for the QuestAI chat UI."""

from __future__ import annotations

import streamlit as st

from config import build_app_config
from llm.openai_client import OpenAIAppClient
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.answer_service import AnswerService
from services.router import RuleBasedRouter
from structured.customer_data import CustomerDataLoader
from structured.planner import StructuredQueryPlanner
from structured.query_engine import StructuredQueryEngine
from ui.components import (
    build_dynamic_prompt_chips,
    render_composer,
    render_conversation,
    render_header,
    render_status_note,
)
from ui.styles import inject_styles


QUESTION_INPUT_KEY = "question_input"
CONVERSATION_KEY = "conversation_history"


def _ensure_session_state() -> None:
    """Initialize the chat session state used by the UI."""
    if QUESTION_INPUT_KEY not in st.session_state:
        st.session_state[QUESTION_INPUT_KEY] = ""
    if CONVERSATION_KEY not in st.session_state:
        st.session_state[CONVERSATION_KEY] = []


def _set_question(question: str) -> None:
    """Populate the composer with a suggested prompt."""
    st.session_state[QUESTION_INPUT_KEY] = question


def _clear_conversation() -> None:
    """Reset the current conversation and composer draft."""
    st.session_state[CONVERSATION_KEY] = []
    st.session_state[QUESTION_INPUT_KEY] = ""


def _submit_question(answer_service: AnswerService) -> None:
    """Submit the current question through the existing answer flow."""
    question = st.session_state.get(QUESTION_INPUT_KEY, "").strip()
    if not question:
        st.warning("Please enter a question before submitting.")
        return

    conversation_history = st.session_state[CONVERSATION_KEY]
    response = answer_service.answer_question(
        question=question,
        conversation_turns=conversation_history,
    )
    conversation_history.append({"question": question, "response": response})
    st.session_state[QUESTION_INPUT_KEY] = ""


def _build_answer_service(
    config,
) -> tuple[
    AnswerService,
    DocumentStore,
    CustomerDataLoader,
    MarkdownChunker,
    OpenAIAppClient,
]:
    """Construct the current application services used by the UI."""
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
    llm_client = OpenAIAppClient(
        model=config.openai_model,
        enabled=config.llm_enabled_for_retrieval,
        api_key=config.openai_api_key,
    )
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


def _build_status_line(
    config,
    llm_client: OpenAIAppClient,
    retrieval_documents: list,
    chunks: list,
    dataset_info,
) -> str:
    """Build the compact runtime status line shown in the header."""
    markdown_count = sum(
        1 for document in retrieval_documents if document.source_type == "markdown"
    )
    text_count = sum(1 for document in retrieval_documents if document.source_type == "text")
    pdf_count = sum(1 for document in retrieval_documents if document.source_type == "pdf")
    retrieval_status = "OpenAI available" if llm_client.is_available() else "Fallback mode"

    return (
        f"Retrieval synthesis: {retrieval_status} | "
        f"Model: {config.openai_model} | "
        f"Markdown docs: {markdown_count} | "
        f"Text docs: {text_count} | "
        f"PDF docs: {pdf_count} | "
        f"Chunks: {len(chunks)} | "
        f"CSV: {dataset_info.file_name if dataset_info.dataset_found and dataset_info.file_name else 'Not loaded'}"
    )


def run_app() -> None:
    """Run the Streamlit QuestAI application."""
    config = build_app_config()
    (
        answer_service,
        document_store,
        customer_data_loader,
        chunker,
        llm_client,
    ) = _build_answer_service(config)

    retrieval_bundle = document_store.load_retrieval_bundle()
    retrieval_documents = retrieval_bundle.documents
    document_load_issues = retrieval_bundle.issues
    chunks = chunker.chunk_documents(retrieval_documents)
    dataset_info = customer_data_loader.get_data_info()
    skipped_pdf_issues = [
        issue for issue in document_load_issues if issue.source_type == "pdf"
    ]

    st.set_page_config(page_title="QuestAI", layout="centered")
    _ensure_session_state()
    inject_styles()

    conversation_history = st.session_state[CONVERSATION_KEY]
    synthesis_status_code, synthesis_status_message = llm_client.get_status()
    status_line = _build_status_line(
        config=config,
        llm_client=llm_client,
        retrieval_documents=retrieval_documents,
        chunks=chunks,
        dataset_info=dataset_info,
    )

    render_header(status_line=status_line, clear_callback=_clear_conversation)

    if synthesis_status_code != "success":
        render_status_note(synthesis_status_message)
    if skipped_pdf_issues:
        render_status_note(
            "Skipped PDF files: "
            + "; ".join(f"{issue.file_name} ({issue.reason})" for issue in skipped_pdf_issues)
        )

    render_conversation(conversation_history)
    prompt_chips = build_dynamic_prompt_chips(
        conversation_history=conversation_history,
        example_questions=config.example_questions,
    )
    render_composer(
        prompt_chips=prompt_chips,
        answer_service=answer_service,
        question_input_key=QUESTION_INPUT_KEY,
        set_question_callback=_set_question,
        submit_callback=_submit_question,
    )
