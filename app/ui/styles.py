"""Streamlit CSS injection for the QuestAI chat UI."""

from __future__ import annotations

import streamlit as st


_UI_STYLES = """
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
    --qa-composer-height: 156px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
    background: var(--qa-bg);
}

.block-container {
    max-width: var(--qa-content-width);
    padding-top: calc(var(--qa-header-height) - 5rem);
    padding-bottom: calc(var(--qa-composer-height) + 3rem);
}

div[data-testid="stVerticalBlock"] > div:has(> div > #qa-header-anchor) {
    position: fixed;
    inset: 0 0 auto 0;
    z-index: 1000;
    margin: 0 !important;
    padding-bottom: 0 !important;
    background: rgba(246, 247, 251, 0.96);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(223, 227, 234, 0.85);
    box-shadow: 0 8px 24px rgba(18, 31, 53, 0.05);
}

div[data-testid="stVerticalBlock"] > div:has(> div > #qa-composer-anchor) {
    position: fixed;
    inset: auto 0 0 0;
    z-index: 1000;
    margin: 0 !important;
    padding-bottom: 0 !important;
    background: linear-gradient(to top, rgba(246, 247, 251, 0.98), rgba(246, 247, 251, 0.94));
    backdrop-filter: blur(12px);
    border-top: 1px solid rgba(223, 227, 234, 0.85);
    box-shadow: 0 -8px 24px rgba(18, 31, 53, 0.06);
}

div[data-testid="stVerticalBlock"] > div:has(> div > #qa-header-anchor) > div,
div[data-testid="stVerticalBlock"] > div:has(> div > #qa-composer-anchor) > div {
    width: min(calc(100vw - 24px), var(--qa-content-width));
    margin: 0 auto;
}

.qa-header-shell {
    min-height: 0;
    display: flex;
    align-items: flex-start;
    padding: 0.75rem 0 0.65rem 0;
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
    padding-top: 0.4rem;
    padding-bottom: 0.75rem;
}

.qa-conversation-shell {
    display: flex;
    flex-direction: column;
    gap: 1.05rem;
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
    gap: 0.55rem;
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
    gap: 0.95rem;
    width: 100%;
    padding: 0.15rem 0;
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
    box-shadow: 0 3px 10px rgba(18, 31, 53, 0.05);
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
    margin-bottom: 0.75rem;
}

.qa-citations {
    margin-bottom: 0.2rem;
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

.qa-badge-row {
    margin-top: 0.2rem;
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
    padding: 0.6rem 0 0;
}

.qa-input-wrap {
    background: transparent;
    border: 0;
    padding: 0;
    box-shadow: none;
}

.qa-chip-note {
    color: var(--qa-muted);
    font-size: 0.76rem;
    margin-bottom: 0.25rem;
}

.qa-prompts-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: flex-start;
}

.qa-prompt-chip {
    display: inline-block;
}

.qa-composer-help {
    color: var(--qa-muted);
    font-size: 0.78rem;
    line-height: 1.4;
    padding-top: 0.15rem;
    margin-bottom: 0 !important;
}

div[data-testid="stTextArea"] {
    margin-bottom: 0;
}

div[data-testid="stTextArea"] > div,
div[data-baseweb="textarea"] {
    background: #ffffff !important;
    border: 1px solid var(--qa-border) !important;
    border-radius: 18px !important;
    box-shadow: var(--qa-shadow) !important;
    padding: 0.7rem !important;
}

div[data-testid="stTextArea"] textarea {
    border: 0 !important;
    outline: none !important;
    box-shadow: none !important;
    background: #ffffff !important;
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
    box-shadow: 0 10px 24px rgba(15, 139, 141, 0.2) !important;
}

.qa-clear-wrap div[data-testid="stButton"] > button {
    border: 1px solid var(--qa-border) !important;
    background: var(--qa-panel) !important;
    color: var(--qa-text) !important;
    border-radius: 12px !important;
    padding: 0.68rem 1rem !important;
    min-height: 44px !important;
    box-shadow: 0 4px 16px rgba(18, 31, 53, 0.06);
}

.qa-prompts-wrap div[data-testid="stButton"] > button {
    border: 1px solid var(--qa-border) !important;
    background: #ffffff !important;
    color: #44506a !important;
    border-radius: 999px !important;
    padding: 0.28rem 0.72rem !important;
    font-size: 12px !important;
    line-height: 1.25 !important;
    white-space: normal !important;
    text-align: center !important;
    min-height: 32px !important;
    margin: 0 !important;
    box-shadow: 0 2px 8px rgba(18, 31, 53, 0.04) !important;
}

@media (max-width: 768px) {
    :root {
        --qa-header-height: 136px;
        --qa-composer-height: 210px;
    }

    .block-container {
        padding-top: calc(var(--qa-header-height) - 5.2rem);
        padding-bottom: calc(var(--qa-composer-height) + 2.5rem);
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
"""


def inject_styles() -> None:
    """Inject the shared Streamlit UI stylesheet."""
    st.markdown(_UI_STYLES, unsafe_allow_html=True)
