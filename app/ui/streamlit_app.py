import streamlit as st

from config import build_app_config
from llm.openai_client import OpenAIRetrievalSynthesizer
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.answer_service import AnswerService
from services.router import SimpleRouter
from structured.customer_data import CustomerDataLoader
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
    router = SimpleRouter(
        retrieval_keywords=config.retrieval_keywords,
        structured_keywords=config.structured_keywords,
    )
    document_store = DocumentStore(docs_path=config.docs_path)
    chunker = MarkdownChunker(max_characters=config.markdown_chunk_max_characters)
    retriever = KeywordRetriever(top_k=config.retrieval_top_k)
    retrieval_synthesizer = OpenAIRetrievalSynthesizer(
        model=config.openai_model,
        enabled=config.llm_enabled_for_retrieval,
        api_key=config.openai_api_key,
    )
    customer_data_loader = CustomerDataLoader(
        structured_data_path=config.structured_data_path
    )
    structured_query_engine = StructuredQueryEngine()
    answer_service = AnswerService(
        router=router,
        document_store=document_store,
        customer_data_loader=customer_data_loader,
        chunker=chunker,
        retriever=retriever,
        structured_query_engine=structured_query_engine,
        retrieval_synthesizer=retrieval_synthesizer,
        retrieval_context_max_characters=config.retrieval_context_max_characters,
    )
    documents = document_store.list_documents()
    markdown_documents = document_store.load_markdown_documents()
    chunks = chunker.chunk_documents(markdown_documents)
    dataset_info = customer_data_loader.get_data_info()

    st.set_page_config(page_title=config.app_title, layout="centered")
    if QUESTION_INPUT_KEY not in st.session_state:
        st.session_state[QUESTION_INPUT_KEY] = ""

    st.title(config.app_title)
    st.write(config.app_description)

    retrieval_status = "OpenAI available" if retrieval_synthesizer.is_available() else "Fallback mode"
    with st.container():
        st.subheader("Status")
        st.caption(
            f"Retrieval synthesis: {retrieval_status} | "
            f"Model: {config.openai_model} | "
            f"Markdown docs: {len(markdown_documents)} | "
            f"Chunks: {len(chunks)} | "
            f"CSV: {dataset_info.file_name if dataset_info.dataset_found and dataset_info.file_name else 'Not loaded'}"
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

        st.subheader("Route")
        st.code(response.route)

        st.subheader("Support Level")
        st.write(response.support_level.title())

        st.subheader("Synthesis")
        st.write(response.synthesis_method.replace("_", " ").title())

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
            with st.expander("Retrieved Chunks", expanded=False):
                for item in response.retrieved_chunks:
                    heading = item.chunk.section_heading or "No heading"
                    st.write(
                        f"- {item.chunk.file_name} | {heading} | score={item.score} | terms={', '.join(item.matched_terms)}"
                    )
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
