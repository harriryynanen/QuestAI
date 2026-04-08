"""Lightweight HTTP API entrypoint for QuestAI."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from bootstrap import build_app_services
from config import build_app_config
from models import AnswerResponse, Route, SupportLevel
from services.answer_service import AnswerService


class ConversationTurnResponsePayload(BaseModel):
    answer: str
    sources_used: list[str] = Field(default_factory=list)
    support_level: SupportLevel
    limitations: str
    route: Route
    follow_up_questions: list[str] = Field(default_factory=list)
    matched_customer_name: str | None = None
    matched_customer_names: list[str] | None = None
    matched_field_name: str | None = None
    matched_field_value: str | None = None
    structured_dataset: str | None = None
    synthesis_method: str = "deterministic"
    synthesis_status: str | None = None
    synthesis_status_message: str | None = None
    routing_method: str = "rules"
    routing_confidence: str = "low"
    routing_reason: str | None = None
    planning_method: str | None = None
    planning_reason: str | None = None


class ConversationTurnPayload(BaseModel):
    question: str
    response: ConversationTurnResponsePayload


class AnswerRequest(BaseModel):
    question: str
    conversation_turns: list[ConversationTurnPayload] = Field(default_factory=list)


class AnswerResponsePayload(BaseModel):
    answer: str
    route: Route
    support_level: SupportLevel
    limitations: str
    sources_used: list[str]
    follow_up_questions: list[str]
    matched_customer_name: str | None = None
    matched_customer_names: list[str] | None = None
    matched_field_name: str | None = None
    matched_field_value: str | None = None
    structured_dataset: str | None = None
    routing_method: str
    routing_confidence: str
    routing_reason: str | None = None
    planning_method: str | None = None
    planning_reason: str | None = None
    synthesis_method: str
    synthesis_status: str | None = None
    synthesis_status_message: str | None = None


def _to_answer_response(payload: ConversationTurnResponsePayload) -> AnswerResponse:
    return AnswerResponse(
        answer=payload.answer,
        sources_used=payload.sources_used,
        support_level=payload.support_level,
        limitations=payload.limitations,
        route=payload.route,
        retrieved_chunks=[],
        follow_up_questions=payload.follow_up_questions,
        matched_customer_name=payload.matched_customer_name,
        matched_customer_names=payload.matched_customer_names,
        matched_field_name=payload.matched_field_name,
        matched_field_value=payload.matched_field_value,
        structured_dataset=payload.structured_dataset,
        synthesis_method=payload.synthesis_method,
        synthesis_status=payload.synthesis_status,
        synthesis_status_message=payload.synthesis_status_message,
        routing_method=payload.routing_method,
        routing_confidence=payload.routing_confidence,
        routing_reason=payload.routing_reason,
        planning_method=payload.planning_method,
        planning_reason=payload.planning_reason,
    )


def _to_payload(response: AnswerResponse) -> AnswerResponsePayload:
    return AnswerResponsePayload(
        answer=response.answer,
        route=response.route,
        support_level=response.support_level,
        limitations=response.limitations,
        sources_used=response.sources_used,
        follow_up_questions=response.follow_up_questions,
        matched_customer_name=response.matched_customer_name,
        matched_customer_names=response.matched_customer_names,
        matched_field_name=response.matched_field_name,
        matched_field_value=response.matched_field_value,
        structured_dataset=response.structured_dataset,
        routing_method=response.routing_method,
        routing_confidence=response.routing_confidence,
        routing_reason=response.routing_reason,
        planning_method=response.planning_method,
        planning_reason=response.planning_reason,
        synthesis_method=response.synthesis_method,
        synthesis_status=response.synthesis_status,
        synthesis_status_message=response.synthesis_status_message,
    )


def create_api_app(answer_service: AnswerService | None = None) -> FastAPI:
    app = FastAPI(title="QuestAI API", version="0.1.0")

    if answer_service is None:
        config = build_app_config()
        answer_service, _, _, _, _ = build_app_services(config)
    app.state.answer_service = answer_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/answer", response_model=AnswerResponsePayload)
    def answer(request: AnswerRequest) -> AnswerResponsePayload:
        conversation_turns = [
            {"question": turn.question, "response": _to_answer_response(turn.response)}
            for turn in request.conversation_turns
        ]
        response = app.state.answer_service.answer_question(
            question=request.question,
            conversation_turns=conversation_turns,
        )
        return _to_payload(response)

    return app


app = create_api_app()
