import json

from app.llm.openai_client import OpenAIAppClient
from app.llm.prompts import build_semantic_plan_messages
from app.models import RoutingDecision
from app.services.router import RuleBasedRouter, is_confident_routing_decision


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
        "structured_dataset": "customer_portfolio",
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
    assert result.plan.structured_dataset == "customer_portfolio"


def test_semantic_planning_parses_advisory_case_fields(monkeypatch):
    llm_client = OpenAIAppClient(
        model="gpt-5.4-mini",
        enabled=True,
        api_key="test-key",
    )

    payload = {
        "route": "structured",
        "operation": "fact",
        "customer_name": "Harbor Foods Demo Oy",
        "field_name": "advisory_owner",
        "product_name": None,
        "document_topic": None,
        "comparison_direction": None,
        "filter_value": None,
        "needs_documents": False,
        "needs_structured_data": True,
        "confidence": "high",
        "reason": "The question asks for the person responsible for one advisory case.",
        "structured_dataset": "advisory_case_pipeline",
    }

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            return type("Response", (), {"output_text": json.dumps(payload)})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(llm_client, "_get_client", lambda: FakeClient())

    result = llm_client.plan_question("Who is responsible for Harbor Foods Demo Oy's case?")

    assert result.status == "success"
    assert result.plan.route == "structured"
    assert result.plan.operation == "fact"
    assert result.plan.customer_name == "Harbor Foods Demo Oy"
    assert result.plan.field_name == "advisory_owner"
    assert result.plan.structured_dataset == "advisory_case_pipeline"


def test_semantic_plan_prompt_includes_structured_dataset_schema():
    messages = build_semantic_plan_messages("Who is responsible for Harbor Foods Demo Oy's case?")
    system_message = messages[0]["content"]

    assert "Available structured datasets:" in system_message
    assert "customer_portfolio" in system_message
    assert "advisory_case_pipeline" in system_message
    assert "advisory_owner" in system_message
    assert "next_action" in system_message


def test_planner_context_includes_previous_structured_dataset_and_field_value(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="structured",
            operation="fact",
            customer_name="Riverstone Demo Logistics Ltd",
            field_name="advisory_owner",
            needs_structured_data=True,
            confidence="high",
            reason="Planner matched an advisory owner lookup.",
            structured_dataset="advisory_case_pipeline",
        )
    )
    service = answer_service_factory(planning_result=planning_result)
    response = service.answer_question("Who is the advisory owner of Riverstone Demo Logistics Ltd?")

    context = service._build_planner_context(
        [
            {
                "question": "Who is the advisory owner of Riverstone Demo Logistics Ltd?",
                "response": response,
            }
        ]
    )

    assert context is not None
    assert "Structured dataset: advisory_case_pipeline" in context
    assert "Matched field: advisory_owner" in context
    assert "Matched field value: Mika Salonen" in context
    assert "Matched customer: Riverstone Demo Logistics Ltd" in context


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
