import streamlit as st

from config import DEFAULT_CONFIG
from services.answer_service import AnswerService
from services.router import SimpleRouter


def run_app() -> None:
    config = DEFAULT_CONFIG
    router = SimpleRouter(
        retrieval_keywords=config.retrieval_keywords,
        structured_keywords=config.structured_keywords,
    )
    answer_service = AnswerService(router=router)

    st.set_page_config(page_title=config.app_title, layout="centered")
    st.title(config.app_title)
    st.write(config.app_description)

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
            st.write("No sources used in this placeholder slice.")

        st.subheader("Limitations")
        st.write(response.limitations)
