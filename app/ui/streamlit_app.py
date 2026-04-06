import streamlit as st

from config import DEFAULT_CONFIG
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.answer_service import AnswerService
from services.router import SimpleRouter
from structured.customer_data import CustomerDataLoader
from structured.query_engine import StructuredQueryEngine


def run_app() -> None:
    config = DEFAULT_CONFIG
    router = SimpleRouter(
        retrieval_keywords=config.retrieval_keywords,
        structured_keywords=config.structured_keywords,
    )
    document_store = DocumentStore(docs_path=config.docs_path)
    chunker = MarkdownChunker(max_characters=config.markdown_chunk_max_characters)
    retriever = KeywordRetriever(top_k=config.retrieval_top_k)
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
    )
    documents = document_store.list_documents()
    markdown_documents = document_store.load_markdown_documents()
    chunks = chunker.chunk_documents(markdown_documents)
    dataset_info = customer_data_loader.get_data_info()

    st.set_page_config(page_title=config.app_title, layout="centered")
    st.title(config.app_title)
    st.write(config.app_description)

    st.subheader("Data Status")
    st.write(f"Documents folder: `{config.docs_path}`")
    if documents:
        st.write("Detected document files:")
        for document in documents:
            st.write(
                f"- {document.file_name} ({document.extension}, {document.size_bytes} bytes)"
            )
    else:
        st.info(
            "No supported documents found in data/docs/. Supported types: .md, .txt, .pdf."
        )

    st.write(f"Loaded markdown documents: {len(markdown_documents)}")
    st.write(f"Generated markdown chunks: {len(chunks)}")

    st.write(f"Structured data folder: `{config.structured_data_path}`")
    if dataset_info.dataset_found and dataset_info.file_name is not None:
        st.write(f"Detected CSV dataset: {dataset_info.file_name}")
        st.write(f"Row count: {dataset_info.row_count}")
        st.write("Columns:")
        if dataset_info.column_names:
            for column_name in dataset_info.column_names:
                st.write(f"- {column_name}")
        else:
            st.write("No columns detected.")
    else:
        st.info("No CSV dataset found in data/structured/. Supported type: .csv.")

    question = st.text_input("Business question", placeholder="e.g. What are the eligibility criteria for the product?")
    submitted = st.button("Submit", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
            return

        response = answer_service.answer_question(question)

        st.subheader("Answer")
        st.write(response.answer)

        st.subheader("Route")
        st.code(response.route)

        st.subheader("Support Level")
        st.write(response.support_level.title())

        st.subheader("Sources Used")
        if response.sources_used:
            for source in response.sources_used:
                st.write(f"- {source}")
        else:
            st.write("No sources were used for this answer.")

        if response.matched_customer_name or response.matched_field_name:
            st.subheader("Structured Match")
            if response.matched_customer_name:
                st.write(f"Matched customer: {response.matched_customer_name}")
            if response.matched_field_name:
                st.write(f"Matched field: {response.matched_field_name}")

        if response.retrieved_chunks:
            st.subheader("Retrieved Chunks")
            for item in response.retrieved_chunks:
                heading = item.chunk.section_heading or "No heading"
                st.write(
                    f"- {item.chunk.file_name} | {heading} | score={item.score} | terms={', '.join(item.matched_terms)}"
                )
                st.caption(item.chunk.text)

        st.subheader("Limitations")
        st.write(response.limitations)
