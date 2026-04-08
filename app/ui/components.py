"""Rendering helpers for the QuestAI Streamlit UI."""

from __future__ import annotations

import html
from typing import Any, Callable

import streamlit as st

from app.models import AnswerResponse
from app.services.answer_service import AnswerService
from app.ui.formatters import build_visible_source_references, format_source_reference


def render_header(status_line: str, clear_callback: Callable[[], None]) -> None:
    """Render the fixed header with status and clear action."""
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
                on_click=clear_callback,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_status_note(message: str) -> None:
    """Render a muted status note below the header."""
    st.markdown("<div class='qa-status-note'>", unsafe_allow_html=True)
    st.caption(message)
    st.markdown("</div>", unsafe_allow_html=True)


def render_conversation(conversation_history: list[dict[str, Any]]) -> None:
    """Render the full conversation history in order."""
    st.markdown(
        "<div class='qa-chat-region'><div class='qa-conversation-shell'>",
        unsafe_allow_html=True,
    )
    for turn in conversation_history:
        render_user_message(str(turn["question"]))
        render_assistant_message(turn["response"])
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_user_message(question: str) -> None:
    """Render a right-aligned user message bubble."""
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


def render_assistant_message(response: AnswerResponse) -> None:
    """Render a left-aligned assistant response card."""
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


def render_composer(
    prompt_chips: list[str],
    answer_service: AnswerService,
    question_input_key: str,
    set_question_callback: Callable[[str], None],
    submit_callback: Callable[[AnswerService], None],
) -> None:
    """Render the fixed bottom composer and suggested prompt chips."""
    st.markdown("<div id='qa-composer-anchor'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='qa-composer-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='qa-input-wrap'>", unsafe_allow_html=True)
        st.text_area(
            "Message QuestAI",
            key=question_input_key,
            height=72,
            placeholder="Ask about policy guidance, customer data, or a cautious combined question.",
            label_visibility="collapsed",
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
                on_click=submit_callback,
                args=(answer_service,),
            )

        if prompt_chips:
            st.markdown("<div class='qa-chip-note'>Suggested prompts</div>", unsafe_allow_html=True)
            prompt_cols = st.columns(len(prompt_chips), gap="small")
            for index, prompt in enumerate(prompt_chips):
                with prompt_cols[index]:
                    st.markdown("<div class='qa-prompt-chip'>", unsafe_allow_html=True)
                    st.button(
                        prompt,
                        key=f"prompt-chip-{index}",
                        use_container_width=True,
                        on_click=set_question_callback,
                        args=(prompt,),
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def build_dynamic_prompt_chips(
    conversation_history: list[dict[str, Any]],
    example_questions: list[str],
) -> list[str]:
    """Build up to three context-aware prompt suggestions for the composer."""
    if not conversation_history:
        return _build_starter_prompt_chips(example_questions)

    last_turn = conversation_history[-1]
    last_question = str(last_turn["question"])
    last_response: AnswerResponse = last_turn["response"]

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
        suggestions.extend(example_questions)

    deduped: list[str] = []
    for suggestion in suggestions:
        cleaned = suggestion.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)

    return deduped[:3]


def _build_starter_prompt_chips(example_questions: list[str]) -> list[str]:
    """Pick a small balanced starter set with retrieval, structured, and combined examples."""
    categorized = {
        "retrieval": None,
        "structured": None,
        "combined": None,
    }

    for question in example_questions:
        category = _categorize_example_question(question)
        if category and categorized[category] is None:
            categorized[category] = question

    starter_chips = [
        categorized["retrieval"],
        categorized["structured"],
        categorized["combined"],
    ]
    return [chip for chip in starter_chips if chip]


def _categorize_example_question(question: str) -> str | None:
    normalized = question.lower()
    if "based on" in normalized and "data" in normalized:
        return "combined"
    if any(term in normalized for term in ("policy", "guide", "criteria")):
        return "retrieval"
    return "structured"


def _render_answer_card(response: AnswerResponse) -> None:
    """Render the main answer card with compact citation markers."""
    visible_sources = build_visible_source_references(response.sources_used)
    citation_markers = " ".join(
        f"<span class='qa-citation'>[{index}]</span>"
        for index, _ in enumerate(visible_sources, start=1)
    )
    st.markdown(
        (
            "<div class='qa-answer-card'>"
            "<div class='qa-answer-label'>Answer</div>"
            f"<div class='qa-answer-text'>{html.escape(response.answer)}</div>"
            f"<div class='qa-citations'>{citation_markers}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_badges(response: AnswerResponse) -> None:
    """Render the compact route/support/synthesis badges."""
    badges = [
        f"Route: {response.route.title()}",
        f"Support: {response.support_level.title()}",
        f"Synthesis: {response.synthesis_method.replace('_', ' ').title()}",
    ]
    badge_html = "".join(
        f"<span class='qa-badge'>{html.escape(badge)}</span>" for badge in badges
    )
    st.markdown(f"<div class='qa-badge-row'>{badge_html}</div>", unsafe_allow_html=True)


def _render_answer_details(response: AnswerResponse) -> None:
    """Render the detailed provenance and reasoning expander."""
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
                st.write(f"[{index}] {format_source_reference(source)}")
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


def _extract_topic_prompts(question: str) -> list[str]:
    """Derive lightweight topic-based suggested prompts from the last question."""
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
