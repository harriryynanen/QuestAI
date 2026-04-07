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
LAST_RESPONSE_KEY = "last_response"


def _set_question(question: str) -> None:
    st.session_state[QUESTION_INPUT_KEY] = question


def _submit_question(answer_service: AnswerService) -> None:
    question = st.session_state.get(QUESTION_INPUT_KEY, "").strip()
    if not question:
        st.warning("Please enter a question before submitting.")
        return

    st.session_state[LAST_RESPONSE_KEY] = answer_service.answer_question(question)


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
    documents = document_store.list_documents()
    retrieval_documents = document_store.load_retrieval_documents()
    document_load_issues = document_store.get_document_load_issues()
    chunks = chunker.chunk_documents(retrieval_documents)
    dataset_info = customer_data_loader.get_data_info()
    markdown_count = sum(1 for document in retrieval_documents if document.source_type == "markdown")
    pdf_count = sum(1 for document in retrieval_documents if document.source_type == "pdf")
    skipped_pdf_issues = [issue for issue in document_load_issues if issue.source_type == "pdf"]

    st.set_page_config(page_title=config.app_title, layout="centered")
    if QUESTION_INPUT_KEY not in st.session_state:
        st.session_state[QUESTION_INPUT_KEY] = ""

    st.title(config.app_title)
    st.write(config.app_description)

    synthesis_status_code, synthesis_status_message = llm_client.get_status()
    retrieval_status = "OpenAI available" if llm_client.is_available() else "Fallback mode"
    with st.container():
        st.subheader("Status")
        st.caption(
            f"Retrieval synthesis: {retrieval_status} | "
            f"Model: {config.openai_model} | "
            f"Markdown docs: {markdown_count} | "
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

    with st.container():
        st.subheader("Ask A Question")
        st.text_input(
            "Business question",
            key=QUESTION_INPUT_KEY,
            placeholder="e.g. What does the policy say about tax arrears?",
        )
        st.button("Submit", type="primary", on_click=_submit_question, args=(answer_service,))

    with st.container():
        st.subheader("Example Questions")
        for index, example_question in enumerate(config.example_questions):
            st.button(
                example_question,
                key=f"example-question-{index}",
                use_container_width=True,
                on_click=_set_question,
                args=(example_question,),
            )

    response = st.session_state.get(LAST_RESPONSE_KEY)
    if response is None:
        return

    with st.container():
        st.subheader("Answer")
        st.write(response.answer)

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

        st.subheader("Sources Used")
        if response.sources_used:
            for source in response.sources_used:
                st.write(f"- {source}")
        else:
            st.write("No sources were used for this answer.")

        if response.matched_customer_name or response.matched_field_name:
            with st.expander("Structured Match", expanded=False):
                if response.matched_customer_name:
                    st.write(f"Matched customer: {response.matched_customer_name}")
                if response.matched_field_name:
                    st.write(f"Matched field: {response.matched_field_name}")

        if response.retrieved_chunks:
            with st.expander("Retrieved Evidence", expanded=False):
                for item in response.retrieved_chunks:
                    heading = item.chunk.section_heading or "No heading"
                    st.write(
                        f"- {item.chunk.file_name} -- {heading} | {item.chunk.chunk_id} | score={item.score:.1f}"
                    )
                    st.caption(item.match_summary)
                    st.caption(item.chunk.text)

        st.subheader("Limitations")
        st.write(response.limitations)

    if response.follow_up_questions:
        with st.container():
            st.subheader("Suggested Next Questions")
            for index, follow_up_question in enumerate(response.follow_up_questions):
                st.button(
                    follow_up_question,
                    key=f"follow-up-question-{index}",
                    use_container_width=True,
                    on_click=_set_question,
                    args=(follow_up_question,),
                )
