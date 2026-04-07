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
            --bg: #f6f7fb;
            --panel: rgba(255,255,255,0.95);
            --panel-solid: #ffffff;
            --border: #dfe3ea;
            --text: #162033;
            --muted: #6d7890;
            --accent: #0f8b8d;
            --chip: #eef2f8;
            --header-height: 104px;
            --composer-height: 150px;
            --content-width: 760px;
            --radius: 18px;
            --shadow: 0 8px 30px rgba(18, 31, 53, 0.08);
            --shadow-header: 0 4px 20px rgba(18, 31, 53, 0.04);
            --shadow-footer: 0 -8px 30px rgba(18, 31, 53, 0.05);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        }

        /* Pad the main content area so it doesn't hide behind floating elements */
        .block-container {
            max-width: var(--content-width) !important;
            padding-top: calc(var(--header-height) + 2rem) !important;
            padding-bottom: calc(var(--composer-height) + 2.5rem) !important;
        }

        /* ----------------------------------- */
        /* FULL-WIDTH FLOATING HEADER          */
        /* ----------------------------------- */
        div[data-testid="stVerticalBlock"]:has(> div.element-container #qa-header-anchor) {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw !important;
            height: var(--header-height);
            background: var(--panel);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            z-index: 99999;
            box-shadow: var(--shadow-header);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 0;
        }
        div[data-testid="stVerticalBlock"]:has(> div.element-container #qa-header-anchor) > div {
            max-width: var(--content-width) !important;
            width: 100%;
            margin: 0 auto;
            padding: 0 1rem;
        }

        /* Header typography */
        .brand-wrap { min-width: 0; }
        .brand {
            font-size: 20px;
            font-weight: 700;
            margin: 0 0 4px 0;
            letter-spacing: -0.02em;
            color: var(--text);
            line-height: 1.2;
        }
        .meta {
            font-size: 13px;
            color: var(--muted);
            line-height: 1.35;
        }

        /* Clear button styling */
        .header-clear-wrap div[data-testid="stButton"] > button {
            border: 1px solid var(--border) !important;
            background: var(--panel-solid) !important;
            color: var(--text) !important;
            border-radius: 12px !important;
            padding: 11px 18px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(18,31,53,0.04) !important;
            min-height: auto !important;
            height: auto !important;
            float: right;
        }

        /* ----------------------------------- */
        /* CHAT MESSAGE STYLES                 */
        /* ----------------------------------- */
        .chat-inner {
            display: flex;
            flex-direction: column;
            gap: 20px;
            width: 100%;
        }
        .message-row {
            display: flex;
            width: 100%;
        }
        .message-row.user {
            justify-content: flex-end;
            margin-bottom: 8px;
        }
        .message-bubble {
            max-width: 80%;
            border-radius: var(--radius);
            padding: 14px 18px;
            font-size: 15px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .message-row.user .message-bubble {
            background: #e9edf4;
            color: var(--text);
            border-top-right-radius: 6px;
        }

        /* Bot Answer Card */
        .answer-card {
            background: var(--panel-solid);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: var(--shadow);
            width: 100%;
            margin-bottom: 12px;
        }
        .answer-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 12px;
            text-transform: uppercase;
        }
        .answer-main {
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 16px;
            color: var(--text);
            white-space: pre-wrap;
        }

        /* Bot Citations & Chips */
        .citation-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 28px;
            height: 28px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #f8fafc;
            color: #4e5a72;
            font-size: 13px;
            margin-right: 6px;
            margin-bottom: 12px;
            font-weight: 500;
        }
        .source-line {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.4;
            margin-bottom: 6px;
        }
        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
            margin-top: 6px;
        }
        .chip {
            border-radius: 999px;
            background: var(--chip);
            border: 1px solid #d7dfeb;
            color: #53617d;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 500;
        }

        /* Expander override */
        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            background: #fafbfd !important;
            box-shadow: none !important;
        }
        [data-testid="stExpander"] summary {
            padding: 12px 16px !important;
        }
        [data-testid="stExpander"] summary p {
            font-weight: 600 !important;
            color: var(--text) !important;
            font-size: 14px !important;
        }

        /* ----------------------------------- */
        /* FULL-WIDTH FLOATING COMPOSER        */
        /* ----------------------------------- */
        div[data-testid="stVerticalBlock"]:has(> div.element-container #qa-composer-anchor) {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100vw !important;
            background: linear-gradient(to top, rgba(246,247,251,1) 75%, rgba(246,247,251,0.9));
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(223,227,234,0.85);
            z-index: 99999;
            box-shadow: var(--shadow-footer);
            display: flex;
            justify-content: center;
            padding: 16px 0 24px 0;
        }
        div[data-testid="stVerticalBlock"]:has(> div.element-container #qa-composer-anchor) > div {
            max-width: var(--content-width) !important;
            width: 100%;
            margin: 0 auto;
            padding: 0 1rem;
        }

        /* Text Area / Input styling */
        div[data-testid="stTextArea"] > div {
            background: var(--panel-solid) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 4px 12px rgba(18, 31, 53, 0.04) !important;
            padding: 10px 14px !important;
        }
        div[data-testid="stTextArea"] textarea {
            border: 0 !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
            color: var(--text) !important;
            min-height: 52px !important;
            max-height: 120px !important;
            padding: 0 !important;
            font-size: 15px !important;
            line-height: 1.5 !important;
        }

        /* Composer Action Area */
        .composer-bottom-layout {
            margin-top: 8px;
            display: flex;
            align-items: center;
        }
        
        /* Suggestion Buttons */
        .suggestion-col div[data-testid="stButton"] > button {
            border: 1px solid var(--border) !important;
            background: #ffffff !important;
            border-radius: 999px !important;
            padding: 8px 14px !important;
            font-size: 13px !important;
            color: #44506a !important;
            white-space: nowrap !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
            min-height: auto !important;
            line-height: 1.2 !important;
        }
        .suggestion-col div[data-testid="stButton"] > button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        /* Send Button */
        .send-btn-wrap div[data-testid="stButton"] > button {
            width: 48px !important;
            min-width: 48px !important;
            height: 48px !important;
            border-radius: 999px !important;
            border: 0 !important;
            background: var(--accent) !important;
            color: #fff !important;
            font-size: 24px !important;
            cursor: pointer !important;
            padding: 0 !important;
            box-shadow: 0 4px 12px rgba(15, 139, 141, 0.25) !important;
            float: right;
            transition: transform 0.1s ease;
        }
        .send-btn-wrap div[data-testid="stButton"] > button:active {
            transform: scale(0.95);
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
        f"<span class='chip'>{html.escape(badge)}</span>" for badge in badges
    )
    st.markdown(f"<div class='chips'>{badge_html}</div>", unsafe_allow_html=True)


def _render_user_message(question: str) -> None:
    st.markdown(
        f"<div class='message-row user'><div class='message-bubble'>{html.escape(question)}</div></div>",
        unsafe_allow_html=True,
    )


def _render_answer_card(response) -> None:
    display_sources = _deduplicate_display_sources(response.sources_used)
    
    st.markdown("<div class='answer-label'>ANSWER</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-main'>{html.escape(response.answer)}</div>", unsafe_allow_html=True)

    if display_sources:
        citation_markers = "".join(
            f"<span class='citation-badge'>[{index}]</span>"
            for index, _ in enumerate(display_sources, start=1)
        )
        st.markdown(f"<div>{citation_markers}</div>", unsafe_allow_html=True)

        source_preview = display_sources[:3]
        preview_html = "".join(
            f"<div class='source-line'>[{index}] {html.escape(source)}</div>"
            for index, source in enumerate(source_preview, start=1)
        )
        st.markdown(f"<div class='source-container'>{preview_html}</div>", unsafe_allow_html=True)


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
    st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
    _render_answer_card(response)
    _render_badges(response)
    _render_answer_details(response)
    st.markdown("</div>", unsafe_allow_html=True)


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
    with st.container():
        st.markdown("<span id='qa-header-anchor'></span>", unsafe_allow_html=True)
        header_cols = st.columns([8, 2], gap="large")
        with header_cols[0]:
            st.markdown(
                (
                    "<div class='brand-wrap'>"
                    "<h1 class='brand'>QuestAI</h1>"
                    f"<div class='meta'>{html.escape(status_line)}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown("<div class='header-clear-wrap'>", unsafe_allow_html=True)
            st.button(
                "Clear",
                key="clear-conversation-button",
                help="Start a fresh conversation",
                use_container_width=True,
                on_click=_clear_conversation,
            )
            st.markdown("</div>", unsafe_allow_html=True)


def _render_conversation(conversation_history: list[dict[str, object]]) -> None:
    st.markdown("<div class='chat-inner'>", unsafe_allow_html=True)
    for turn in conversation_history:
        _render_user_message(str(turn["question"]))
        _render_assistant_message(turn["response"])
    st.markdown("</div>", unsafe_allow_html=True)


def _render_composer(prompt_chips: list[str], answer_service: AnswerService) -> None:
    with st.container():
        st.markdown("<span id='qa-composer-anchor'></span>", unsafe_allow_html=True)
        
        st.text_area(
            "Message QuestAI",
            key=QUESTION_INPUT_KEY,
            height=60,
            placeholder="Ask about policy guidance, customer data, or a cautious combined question.",
            label_visibility="collapsed",
        )

        composer_cols = st.columns([10, 2], gap="small", vertical_alignment="center")
        with composer_cols[0]:
            if prompt_chips:
                chip_col_count = len(prompt_chips)
                if chip_col_count > 0:
                    chip_cols = st.columns(chip_col_count, gap="small")
                    for index, prompt in enumerate(prompt_chips):
                        with chip_cols[index]:
                            st.markdown("<div class='suggestion-col'>", unsafe_allow_html=True)
                            st.button(
                                prompt,
                                key=f"prompt-chip-{index}",
                                use_container_width=True,
                                on_click=_set_question,
                                args=(prompt,),
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

        with composer_cols[1]:
            st.markdown("<div class='send-btn-wrap'>", unsafe_allow_html=True)
            st.button(
                "›",
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
        st.markdown("<div style='margin-bottom: 10px;'>", unsafe_allow_html=True)
        st.caption(synthesis_status_message)
        st.markdown("</div>", unsafe_allow_html=True)
    if skipped_pdf_issues:
        st.markdown("<div style='margin-bottom: 10px;'>", unsafe_allow_html=True)
        st.caption(
            "Skipped PDF files: "
            + "; ".join(f"{issue.file_name} ({issue.reason})" for issue in skipped_pdf_issues)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    _render_conversation(conversation_history)
    prompt_chips = _build_dynamic_prompt_chips(conversation_history, config)
    _render_composer(prompt_chips, answer_service)