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
        :root {
            --qa-bg: #f6f7fb;
            --qa-panel: #ffffff;
            --qa-border: #dfe3ea;
            --qa-text: #162033;
            --qa-muted: #6d7890;
            --qa-accent: #0f8b8d;
            --qa-chip: #eef2f8;
            --qa-shadow: 0 8px 30px rgba(18, 31, 53, 0.08);
            --qa-content-width: 900px;
            --qa-header-height: 116px;
            --qa-composer-height: 168px;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
            background: var(--qa-bg);
        }

        .block-container {
            max-width: var(--qa-content-width);
            padding-top: calc(var(--qa-header-height) + 1.4rem);
            padding-bottom: calc(var(--qa-composer-height) + 1.5rem);
        }

        #qa-header-anchor + div {
            position: fixed;
            inset: 0 0 auto 0;
            z-index: 20;
            background: rgba(246, 247, 251, 0.96);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(223, 227, 234, 0.85);
        }

        #qa-composer-anchor + div {
            position: fixed;
            inset: auto 0 0 0;
            z-index: 20;
            background: linear-gradient(to top, rgba(246, 247, 251, 0.98), rgba(246, 247, 251, 0.92));
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(223, 227, 234, 0.85);
            padding: 0.55rem 0 0.8rem 0;
        }

        #qa-header-anchor + div > div,
        #qa-composer-anchor + div > div {
            width: min(calc(100vw - 24px), var(--qa-content-width));
            margin: 0 auto;
        }

        .qa-header-shell {
            min-height: var(--qa-header-height);
            display: flex;
            align-items: center;
        }

        .qa-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            width: 100%;
        }

        .qa-brand-wrap {
            min-width: 0;
        }

        .qa-title {
            margin: 0 0 0.25rem 0;
            color: var(--qa-text);
            font-size: 1.95rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.05;
        }

        .qa-status-line {
            color: var(--qa-muted);
            font-size: 0.83rem;
            line-height: 1.45;
        }

        .qa-status-note {
            margin: 0.25rem 0 0 0;
        }

        .qa-chat-region {
            padding-top: 1.8rem;
            padding-bottom: 0.5rem;
        }

        .qa-chat-shell {
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
        }

        .qa-user-row {
            display: flex;
            justify-content: flex-end;
            width: 100%;
        }

        .qa-user-message {
            display: flex;
            align-items: flex-end;
            justify-content: flex-end;
            gap: 0.65rem;
            width: 100%;
        }

        .qa-user-bubble {
            max-width: 78%;
            background: #e9edf4;
            border: 1px solid #dbe1e8;
            border-radius: 18px 18px 8px 18px;
            padding: 0.9rem 1rem;
            color: var(--qa-text);
            font-size: 0.96rem;
            line-height: 1.5;
            white-space: pre-wrap;
            box-shadow: 0 3px 12px rgba(18, 31, 53, 0.05);
        }

        .qa-assistant-row {
            display: flex;
            justify-content: flex-start;
            width: 100%;
        }

        .qa-assistant-message {
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
            width: 100%;
        }

        .qa-avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 0.76rem;
            font-weight: 700;
        }

        .qa-avatar-user {
            background: #dfe4eb;
            color: #3f4957;
            border: 1px solid #cfd7e2;
        }

        .qa-avatar-assistant {
            background: #e8eef7;
            color: #1f3b5b;
            border: 1px solid #d3dfef;
        }

        .qa-assistant-main {
            width: min(100%, 840px);
            min-width: 0;
        }

        .qa-answer-card {
            background: var(--qa-panel);
            border: 1px solid var(--qa-border);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: var(--qa-shadow);
        }

        .qa-answer-label {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: var(--qa-muted);
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .qa-answer-text {
            color: var(--qa-text);
            font-size: 1rem;
            line-height: 1.6;
            white-space: pre-wrap;
            margin-bottom: 0.85rem;
        }

        .qa-citations {
            margin-bottom: 0.45rem;
        }

        .qa-citation {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 28px;
            height: 28px;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            padding: 0 0.32rem;
            border: 1px solid var(--qa-border);
            border-radius: 8px;
            background: #f8fafc;
            color: #4e5a72;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .qa-source-preview {
            color: var(--qa-muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-bottom: 0.75rem;
        }

        .qa-source-line {
            margin: 0.12rem 0;
        }

        .qa-badge-row {
            margin-top: 0.15rem;
        }

        .qa-badge {
            display: inline-block;
            margin: 0 0.35rem 0.35rem 0;
            padding: 0.38rem 0.65rem;
            border-radius: 999px;
            background: var(--qa-chip);
            border: 1px solid #d7dfeb;
            color: #53617d;
            font-size: 0.75rem;
            line-height: 1;
        }

        .qa-composer-shell {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .qa-input-wrap {
            background: transparent;
            border: 0;
            padding: 0;
            box-shadow: none;
        }

        .qa-composer-help {
            color: var(--qa-muted);
            font-size: 0.78rem;
            line-height: 1.4;
            padding-top: 0.2rem;
        }

        .qa-chip-note {
            color: var(--qa-muted);
            font-size: 0.75rem;
            margin-bottom: 0.22rem;
        }

        .qa-prompt-row {
            margin-bottom: 0.15rem;
        }

        div[data-testid="stTextArea"] {
            margin-bottom: 0;
        }

        div[data-testid="stTextArea"] > div {
            background: var(--qa-panel) !important;
            border: 1px solid var(--qa-border) !important;
            border-radius: 18px !important;
            box-shadow: var(--qa-shadow) !important;
            padding: 0.7rem !important;
        }

        div[data-testid="stTextArea"] textarea {
            border: 0 !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
            color: var(--qa-text) !important;
            min-height: 54px !important;
            max-height: 124px !important;
            padding: 0 !important;
            font-size: 0.94rem !important;
            line-height: 1.5 !important;
            resize: none !important;
        }

        div[data-testid="stButton"] > button {
            border-radius: 999px;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            width: 44px !important;
            min-width: 44px !important;
            height: 44px !important;
            padding: 0 !important;
            border-radius: 999px !important;
            font-size: 1.2rem !important;
            background: var(--qa-accent) !important;
            border: 0 !important;
        }

        .qa-clear-wrap div[data-testid="stButton"] > button {
            border: 1px solid var(--qa-border) !important;
            background: var(--qa-panel) !important;
            color: var(--qa-text) !important;
            border-radius: 12px !important;
            padding: 0.68rem 1rem !important;
            min-height: 44px !important;
            box-shadow: 0 2px 8px rgba(18, 31, 53, 0.04);
        }

        .qa-prompts-wrap div[data-testid="stButton"] > button {
            border: 1px solid var(--qa-border) !important;
            background: #ffffff !important;
            color: #44506a !important;
            border-radius: 999px !important;
            padding: 0.22rem 0.6rem !important;
            font-size: 0.72rem !important;
            line-height: 1.2 !important;
            white-space: normal !important;
            text-align: center !important;
            min-height: 30px !important;
            box-shadow: none !important;
        }

        @media (max-width: 768px) {
            :root {
                --qa-header-height: 136px;
                --qa-composer-height: 210px;
            }

            .block-container {
                padding-top: calc(var(--qa-header-height) + 1.2rem);
                padding-bottom: calc(var(--qa-composer-height) + 1.2rem);
            }

            .qa-header-row {
                flex-direction: column;
                align-items: flex-start;
            }

            .qa-clear-wrap {
                align-self: flex-end;
                width: 100%;
                max-width: 120px;
            }

            .qa-user-bubble {
                max-width: 92%;
            }

            .qa-assistant-main {
                width: 100%;
            }
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

    conversation_history = st.session_state[CONVERSATION_KEY]
    response = answer_service.answer_question(
        question=question,
        conversation_turns=conversation_history,
    )
    conversation_history.append({"question": question, "response": response})
    st.session_state[QUESTION_INPUT_KEY] = ""


def _format_source_label(source: str) -> str:
    if " | row:" in source:
        return source
    return source.replace(" -- ", " - ")


def _deduplicate_display_sources(sources_used: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for source in sources_used:
        display_source = _format_source_label(source)
        if " | row:" in source:
            file_name, _, rest = source.partition(" | row: ")
            row_name = rest.split(" | ", maxsplit=1)[0].strip()
            display_source = f"{file_name} - {row_name} row"
        elif " | sorted by:" in source:
            file_name, _, rest = source.partition(" | sorted by: ")
            row_name = ""
            if " | row: " in rest:
                row_name = rest.split(" | row: ", maxsplit=1)[1].strip()
            display_source = f"{file_name} - {row_name} row" if row_name else file_name
        elif " | filter:" in source:
            file_name = source.split(" | filter:", maxsplit=1)[0].strip()
            display_source = f"{file_name} - filtered rows"
        elif " | exists check:" in source:
            file_name = source.split(" | exists check:", maxsplit=1)[0].strip()
            display_source = f"{file_name} - existence check"
        elif " | rows counted" in source:
            file_name = source.split(" | rows counted", maxsplit=1)[0].strip()
            display_source = f"{file_name} - row count"
        elif " | listed rows" in source:
            file_name = source.split(" | listed rows", maxsplit=1)[0].strip()
            display_source = f"{file_name} - listed rows"

        if display_source not in seen:
            seen.add(display_source)
            deduped.append(display_source)

    return deduped


def _render_badges(response) -> None:
    badges = [
        f"Route: {response.route.title()}",
        f"Support: {response.support_level.title()}",
        f"Synthesis: {response.synthesis_method.replace('_', ' ').title()}",
    ]
    badge_html = "".join(
        f"<span class='qa-badge'>{html.escape(badge)}</span>" for badge in badges
    )
    st.markdown(f"<div class='qa-badge-row'>{badge_html}</div>", unsafe_allow_html=True)


def _render_user_message(question: str) -> None:
    st.markdown(
        (
            "<div class='qa-user-row'>"
            "<div class='qa-user-message'>"
            f"<div class='qa-user-bubble'>{html.escape(question)}</div>"
            "<div class='qa-avatar qa-avatar-user'>You</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_answer_card(response) -> None:
    display_sources = _deduplicate_display_sources(response.sources_used)
    citation_markers = " ".join(
        f"<span class='qa-citation'>[{index}]</span>"
        for index, _ in enumerate(display_sources, start=1)
    )
    source_preview = display_sources[:3]
    preview_html = "".join(
        (
            f"<div class='qa-source-line'><strong>[{index}]</strong> "
            f"{html.escape(source)}</div>"
        )
        for index, source in enumerate(source_preview, start=1)
    )
    st.markdown(
        (
            "<div class='qa-answer-card'>"
            "<div class='qa-answer-label'>Answer</div>"
            f"<div class='qa-answer-text'>{html.escape(response.answer)}</div>"
            f"<div class='qa-citations'>{citation_markers}</div>"
            f"<div class='qa-source-preview'>{preview_html}</div>"
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
                f"Routing: {response.route} ({response.routing_method}, "
                f"{response.routing_confidence} confidence) - {response.routing_reason}"
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
                    f"- {item.chunk.file_name} -- {heading} | "
                    f"{item.chunk.chunk_id} | score={item.score:.1f}"
                )
                st.caption(item.match_summary)
                st.caption(item.chunk.text)

        st.markdown("**Limitations**")
        st.write(response.limitations)


def _render_assistant_message(response) -> None:
    st.markdown(
        (
            "<div class='qa-assistant-row'>"
            "<div class='qa-assistant-message'>"
            "<div class='qa-avatar qa-avatar-assistant'>Q</div>"
            "<div class='qa-assistant-main'>"
        ),
        unsafe_allow_html=True,
    )
    _render_answer_card(response)
    _render_badges(response)
    _render_answer_details(response)
    st.markdown("</div></div></div>", unsafe_allow_html=True)


def _extract_topic_prompts(question: str) -> list[str]:
    normalized = question.lower()
    prompt_map = [
        ("tax arrears", "What does the policy say about tax arrears?"),
        ("payment delays", "What does the policy say about payment delays?"),
        ("financial statements", "Which customers are missing latest financial statements?"),
        ("equity ratio", "What is Harbor Foods Demo Oy's equity ratio?"),
        ("turnover", "Which customer has the highest turnover?"),
        ("flexline", "What are the criteria for FlexLine Demo?"),
        ("invoicebridge", "How does the guide describe InvoiceBridge Demo?"),
    ]

    prompts: list[str] = []
    for pattern, prompt in prompt_map:
        if pattern in normalized and prompt not in prompts:
            prompts.append(prompt)
    return prompts


def _build_dynamic_prompt_chips(conversation_history: list[dict[str, object]], config) -> list[str]:
    if not conversation_history:
        return list(config.example_questions[:3])

    last_turn = conversation_history[-1]
    last_question = str(last_turn["question"])
    last_response = last_turn["response"]

    suggestions: list[str] = []
    suggestions.extend(_extract_topic_prompts(last_question))
    suggestions.extend(last_response.follow_up_questions)

    if last_response.route == "retrieval":
        suggestions.extend(
            [
                "What about payment delays?",
                "What guidance is given about missing financial statements?",
                "How does the guide describe this product?",
            ]
        )
    elif last_response.route == "structured":
        customer_name = last_response.matched_customer_name or "Harbor Foods Demo Oy"
        suggestions.extend(
            [
                f"What is the debt to EBITDA of {customer_name}?",
                f"Does {customer_name} have tax arrears?",
                "Which customer has the highest turnover?",
            ]
        )
    elif last_response.route == "combined":
        customer_name = last_response.matched_customer_name or "Harbor Foods Demo Oy"
        suggestions.extend(
            [
                "What does the policy say about payment delays?",
                f"What is the equity ratio of {customer_name}?",
                "What if turnover is weaker?",
            ]
        )
    else:
        suggestions.extend(config.example_questions)

    deduped: list[str] = []
    for suggestion in suggestions:
        cleaned = suggestion.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)

    return deduped[:3]


def _render_header(status_line: str) -> None:
    st.markdown("<div id='qa-header-anchor'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='qa-header-shell'>", unsafe_allow_html=True)
        header_cols = st.columns([12, 2], gap="large")
        with header_cols[0]:
            st.markdown(
                (
                    "<div class='qa-header-row'>"
                    "<div class='qa-brand-wrap'>"
                    "<div class='qa-title'>QuestAI</div>"
                    f"<div class='qa-status-line'>{html.escape(status_line)}</div>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown("<div class='qa-clear-wrap'>", unsafe_allow_html=True)
            st.button(
                "Clear",
                key="clear-conversation-button",
                help="Start a fresh conversation",
                use_container_width=True,
                on_click=_clear_conversation,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_conversation(conversation_history: list[dict[str, object]]) -> None:
    st.markdown("<div class='qa-chat-region'><div class='qa-chat-shell'>", unsafe_allow_html=True)
    for turn in conversation_history:
        _render_user_message(str(turn["question"]))
        _render_assistant_message(turn["response"])
    st.markdown("</div></div>", unsafe_allow_html=True)


def _render_composer(prompt_chips: list[str], answer_service: AnswerService) -> None:
    st.markdown("<div id='qa-composer-anchor'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='qa-composer-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='qa-input-wrap'>", unsafe_allow_html=True)
        st.text_area(
            "Message QuestAI",
            key=QUESTION_INPUT_KEY,
            height=72,
            placeholder="Ask about policy guidance, customer data, or a cautious combined question.",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if prompt_chips:
            st.markdown("<div class='qa-chip-note'>Suggested prompts</div>", unsafe_allow_html=True)
            ratios = [max(2, min(len(prompt) // 10 + 1, 6)) for prompt in prompt_chips]
            prompt_cols = st.columns(ratios, gap="small")
            for index, prompt in enumerate(prompt_chips):
                with prompt_cols[index]:
                    st.markdown("<div class='qa-prompts-wrap qa-prompt-row'>", unsafe_allow_html=True)
                    st.button(
                        prompt,
                        key=f"prompt-chip-{index}",
                        use_container_width=False,
                        on_click=_set_question,
                        args=(prompt,),
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        composer_cols = st.columns([12, 1], gap="small")
        with composer_cols[0]:
            st.markdown(
                "<div class='qa-composer-help'>Recent conversation context is passed in lightly for follow-up questions.</div>",
                unsafe_allow_html=True,
            )
        with composer_cols[1]:
            st.button(
                ">",
                key="send-question",
                type="primary",
                use_container_width=True,
                on_click=_submit_question,
                args=(answer_service,),
            )
        st.markdown("</div>", unsafe_allow_html=True)


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
    markdown_count = sum(
        1 for document in retrieval_documents if document.source_type == "markdown"
    )
    text_count = sum(1 for document in retrieval_documents if document.source_type == "text")
    pdf_count = sum(1 for document in retrieval_documents if document.source_type == "pdf")
    skipped_pdf_issues = [
        issue for issue in document_load_issues if issue.source_type == "pdf"
    ]

    st.set_page_config(page_title="QuestAI", layout="centered")
    _ensure_session_state()
    _inject_styles()

    conversation_history = st.session_state[CONVERSATION_KEY]
    synthesis_status_code, synthesis_status_message = llm_client.get_status()
    retrieval_status = "OpenAI available" if llm_client.is_available() else "Fallback mode"

    status_line = (
        f"Retrieval synthesis: {retrieval_status} | "
        f"Model: {config.openai_model} | "
        f"Markdown docs: {markdown_count} | "
        f"Text docs: {text_count} | "
        f"PDF docs: {pdf_count} | "
        f"Chunks: {len(chunks)} | "
        f"CSV: {dataset_info.file_name if dataset_info.dataset_found and dataset_info.file_name else 'Not loaded'}"
    )
    _render_header(status_line)

    if synthesis_status_code != "success":
        st.markdown("<div class='qa-status-note'>", unsafe_allow_html=True)
        st.caption(synthesis_status_message)
        st.markdown("</div>", unsafe_allow_html=True)
    if skipped_pdf_issues:
        st.markdown("<div class='qa-status-note'>", unsafe_allow_html=True)
        st.caption(
            "Skipped PDF files: "
            + "; ".join(f"{issue.file_name} ({issue.reason})" for issue in skipped_pdf_issues)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    _render_conversation(conversation_history)
    prompt_chips = _build_dynamic_prompt_chips(conversation_history, config)
    _render_composer(prompt_chips, answer_service)
