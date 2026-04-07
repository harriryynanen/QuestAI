import json

from llm.openai_client import OpenAIAppClient
from models import RoutingDecision
from services.router import RuleBasedRouter, is_confident_routing_decision


def test_semantic_planning_parses_valid_json(monkeypatch):
    llm_client = OpenAIAppClient(
        model="gpt-5.4-mini",
        enabled=True,
        api_key="test-key",
    )

    payload = {
        "route": "structured",
        "operation": "fact",
        "customer_name": "Harbor Foods Demo Oy",
        "field_name": "equity_ratio_pct",
        "product_name": None,
        "document_topic": None,
        "comparison_direction": None,
        "filter_value": None,
        "needs_documents": False,
        "needs_structured_data": True,
        "confidence": "high",
        "reason": "The question asks for one customer field from the CSV.",
    }

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            return type("Response", (), {"output_text": json.dumps(payload)})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(llm_client, "_get_client", lambda: FakeClient())

    result = llm_client.plan_question("What is Harbor Foods Demo Oy's equity ratio?")

    assert result.status == "success"
    assert result.plan.route == "structured"
    assert result.plan.operation == "fact"
    assert result.plan.customer_name == "Harbor Foods Demo Oy"
    assert result.plan.field_name == "equity_ratio_pct"
    assert result.plan.needs_structured_data is True
    assert result.plan.needs_documents is False


def test_rule_router_still_routes_clear_retrieval_question(app_config):
    router = RuleBasedRouter(
        retrieval_keywords=app_config.retrieval_keywords,
        structured_keywords=app_config.structured_keywords,
    )

    decision = router.classify("What does the policy say about tax arrears?")

    assert decision.route == "retrieval"
    assert is_confident_routing_decision(decision)


def test_safe_fallback_is_used_when_semantic_and_rules_are_unclear(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Question is too underspecified.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    routing_decision = service._route_question(
        "Tell me something useful about this case",
        planning_result,
    )

    assert routing_decision == RoutingDecision(
        route="unknown",
        confidence="low",
        reason=(
            "The question could not be routed confidently. "
            "Try asking either a document question or a structured customer-data question."
        ),
        method="safe_fallback",
    )


def test_unclear_question_returns_safe_unknown_response(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Question is too underspecified.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question("Tell me something useful about this case")

    assert response.route == "unknown"
    assert response.support_level == "low"
    assert response.limitations
    assert "could not route" in response.answer.lower() or "try asking" in response.limitations.lower()
