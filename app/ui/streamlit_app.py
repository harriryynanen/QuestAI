import html

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


QUESTION_INPUT_KEY = "question_input"
CONVERSATION_KEY = "conversation_history"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .qa-status-card {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            background: #fafbfc;
            margin-bottom: 1rem;
        }
        .qa-composer-card {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 16px;
            padding: 0.85rem 0.9rem 0.55rem 0.9rem;
            background: #ffffff;
            margin-bottom: 1rem;
        }
        .qa-user-card {
            border-radius: 14px;
            background: #f3f6fb;
            border: 1px solid #dde5f0;
            padding: 0.9rem 1rem;
            margin: 0.25rem 0 0.6rem 0;
        }
        .qa-user-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #5a6272;
            margin-bottom: 0.45rem;
        }
        .qa-user-text {
            color: #1f2937;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .qa-answer-card {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 16px;
            padding: 1.1rem 1rem 1rem 1rem;
            background: linear-gradient(180deg, rgba(248, 249, 252, 0.92), rgba(255, 255, 255, 0.98));
            margin: 0.2rem 0 0.65rem 0;
        }
        .qa-answer-title {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #5a6272;
            margin-bottom: 0.55rem;
        }
        .qa-answer-text {
            font-size: 1.04rem;
            line-height: 1.72;
            color: #1f2430;
            white-space: pre-wrap;
        }
        .qa-badge-row {
            margin: 0.2rem 0 0.35rem 0;
        }
        .qa-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #eef2f7;
            border: 1px solid #d7deea;
            color: #334155;
            font-size: 0.78rem;
            margin: 0 0.35rem 0.35rem 0;
        }
        .qa-citations {
            margin-top: 0.6rem;
            color: #5b6474;
            font-size: 0.86rem;
        }
        .qa-citation {
            display: inline-block;
            margin-right: 0.35rem;
            padding: 0.06rem 0.34rem;
            border-radius: 6px;
            background: #f3f5f8;
            border: 1px solid #e2e7ef;
            color: #374151;
            font-weight: 600;
        }
        .qa-source-summary {
            margin-top: 0.65rem;
            color: #4b5563;
            font-size: 0.9rem;
        }
        .qa-source-line {
            margin: 0.18rem 0;
        }
        .qa-turn-spacer {
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_session_state() -> None:
    if QUESTION_INPUT_KEY not in st.session_state:
        st.session_state[QUESTION_INPUT_KEY] = ""
    if CONVERSATION_KEY not in st.session_state:
        st.session_state[CONVERSATION_KEY] = []


def _set_question(question: str) -> None:
    st.session_state[QUESTION_INPUT_KEY] = question


def _clear_conversation() -> None:
    st.session_state[CONVERSATION_KEY] = []
    st.session_state[QUESTION_INPUT_KEY] = ""


def _submit_question(answer_service: AnswerService) -> None:
    question = st.session_state.get(QUESTION_INPUT_KEY, "").strip()
    if not question:
        st.warning("Please enter a question before submitting.")
        return

    response = answer_service.answer_question(question)
    st.session_state[CONVERSATION_KEY].append(
        {
            "question": question,
            "response": response,
        }
    )
    st.session_state[QUESTION_INPUT_KEY] = ""


def _format_source_label(source: str) -> str:
    if " | row:" in source:
        return source
    return source.replace(" -- ", " - ")


def _render_badges(response) -> None:
    badges = [
        f"Route: {response.route.title()}",
        f"Support: {response.support_level.title()}",
        f"Synthesis: {response.synthesis_method.replace('_', ' ').title()}",
    ]
    badge_html = "".join(f"<span class='qa-badge'>{badge}</span>" for badge in badges)
    st.markdown(f"<div class='qa-badge-row'>{badge_html}</div>", unsafe_allow_html=True)


def _render_user_message(question: str) -> None:
    st.markdown(
        (
            "<div class='qa-user-card'>"
            "<div class='qa-user-label'>You</div>"
            f"<div class='qa-user-text'>{html.escape(question)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_answer_card(response) -> None:
    citation_markers = " ".join(
        f"<span class='qa-citation'>[{index}]</span>"
        for index, _ in enumerate(response.sources_used, start=1)
    )
    source_preview = response.sources_used[:2]
    preview_html = "".join(
        f"<div class='qa-source-line'><strong>[{index}]</strong> {html.escape(_format_source_label(source))}</div>"
        for index, source in enumerate(source_preview, start=1)
    )
    st.markdown(
        (
            "<div class='qa-answer-card'>"
            "<div class='qa-answer-title'>Answer</div>"
            f"<div class='qa-answer-text'>{html.escape(response.answer)}</div>"
            f"<div class='qa-citations'>{citation_markers}</div>"
            f"<div class='qa-source-summary'>{preview_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_answer_details(response) -> None:
    with st.expander("Why this answer", expanded=False):
        st.caption(
            f"Route: {response.route} | "
            f"Support level: {response.support_level.title()} | "
            f"Synthesis: {response.synthesis_method.replace('_', ' ').title()}"
        )
        if response.routing_reason:
            st.caption(
                f"Routing: {response.route} ({response.routing_method}, {response.routing_confidence} confidence) - "
                f"{response.routing_reason}"
            )
        if response.planning_reason:
            st.caption(
                f"Structured planning: {response.planning_method} - {response.planning_reason}"
            )
        if response.synthesis_status_message:
            st.caption(response.synthesis_status_message)

        st.markdown("**Sources Used**")
        if response.sources_used:
            for index, source in enumerate(response.sources_used, start=1):
                st.write(f"[{index}] {_format_source_label(source)}")
        else:
            st.write("No sources were used for this answer.")

        if response.matched_customer_name or response.matched_field_name:
            st.markdown("**Structured Match**")
            if response.matched_customer_name:
                st.write(f"Matched customer: {response.matched_customer_name}")
            if response.matched_field_name:
                st.write(f"Matched field: {response.matched_field_name}")

        if response.retrieved_chunks:
            st.markdown("**Retrieved Evidence**")
            for item in response.retrieved_chunks:
                heading = item.chunk.section_heading or "No heading"
                st.write(
                    f"- {item.chunk.file_name} -- {heading} | {item.chunk.chunk_id} | score={item.score:.1f}"
                )
                st.caption(item.match_summary)
                st.caption(item.chunk.text)

        st.markdown("**Limitations**")
        st.write(response.limitations)


def _render_conversation(conversation_history: list[dict[str, object]]) -> None:
    for index, turn in enumerate(conversation_history):
        question = turn["question"]
        response = turn["response"]

        with st.chat_message("user"):
            _render_user_message(question)

        with st.chat_message("assistant"):
            _render_answer_card(response)
            _render_badges(response)
            _render_answer_details(response)

        if index < len(conversation_history) - 1:
            st.markdown("<div class='qa-turn-spacer'></div>", unsafe_allow_html=True)


def run_app() -> None:
    config = build_app_config()
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
    retrieval_bundle = document_store.load_retrieval_bundle()
    retrieval_documents = retrieval_bundle.documents
    document_load_issues = retrieval_bundle.issues
    chunks = chunker.chunk_documents(retrieval_documents)
    dataset_info = customer_data_loader.get_data_info()
    markdown_count = sum(1 for document in retrieval_documents if document.source_type == "markdown")
    text_count = sum(1 for document in retrieval_documents if document.source_type == "text")
    pdf_count = sum(1 for document in retrieval_documents if document.source_type == "pdf")
    skipped_pdf_issues = [issue for issue in document_load_issues if issue.source_type == "pdf"]

    st.set_page_config(page_title=config.app_title, layout="centered")
    _ensure_session_state()
    _inject_styles()

    st.title(config.app_title)
    st.write(config.app_description)

    synthesis_status_code, synthesis_status_message = llm_client.get_status()
    retrieval_status = "OpenAI available" if llm_client.is_available() else "Fallback mode"
    with st.container():
        st.markdown("<div class='qa-status-card'>", unsafe_allow_html=True)
        st.caption(
            f"Retrieval synthesis: {retrieval_status} | "
            f"Model: {config.openai_model} | "
            f"Markdown docs: {markdown_count} | "
            f"Text docs: {text_count} | "
            f"PDF docs: {pdf_count} | "
            f"Chunks: {len(chunks)} | "
            f"CSV: {dataset_info.file_name if dataset_info.dataset_found and dataset_info.file_name else 'Not loaded'}"
        )
        if synthesis_status_code != "success":
            st.caption(synthesis_status_message)
        if skipped_pdf_issues:
            st.caption(
                "Skipped PDF files: "
                + "; ".join(f"{issue.file_name} ({issue.reason})" for issue in skipped_pdf_issues)
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='qa-composer-card'>", unsafe_allow_html=True)
        st.subheader("Ask QuestAI")
        st.text_area(
            "Your question",
            key=QUESTION_INPUT_KEY,
            height=120,
            placeholder=(
                "Ask a policy question, a customer data question, or a cautious combined question.\n\n"
                "For example: Based on the policy and customer data, does Maple Works Mock Ltd "
                "appear to meet the FlexLine Demo criteria?"
            ),
            label_visibility="collapsed",
        )
        submit_col, clear_col = st.columns([1, 1])
        with submit_col:
            st.button(
                "Send",
                type="primary",
                use_container_width=True,
                on_click=_submit_question,
                args=(answer_service,),
            )
        with clear_col:
            st.button(
                "Clear conversation",
                use_container_width=True,
                on_click=_clear_conversation,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Example prompts", expanded=False):
        for index, example_question in enumerate(config.example_questions):
            st.button(
                example_question,
                key=f"example-question-{index}",
                use_container_width=True,
                on_click=_set_question,
                args=(example_question,),
            )

    conversation_history = st.session_state[CONVERSATION_KEY]
    if conversation_history:
        _render_conversation(conversation_history)
    else:
        st.info("Ask a question to start the conversation.")
