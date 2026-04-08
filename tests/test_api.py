from fastapi.testclient import TestClient

from app.api import create_api_app


def test_health_endpoint_returns_success(answer_service_factory, plan_factory, planning_result_factory):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Health endpoint does not use planning.",
        )
    )
    app = create_api_app(answer_service=answer_service_factory(planning_result=planning_result))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_endpoint_returns_valid_response_shape(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="retrieval",
            operation="policy_lookup",
            needs_documents=True,
            confidence="high",
            reason="Planner matched a document question.",
        )
    )
    app = create_api_app(answer_service=answer_service_factory(planning_result=planning_result))
    client = TestClient(app)

    response = client.post("/answer", json={"question": "What does the policy say about tax arrears?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["route"] == "retrieval"
    assert isinstance(payload["sources_used"], list)
    assert isinstance(payload["follow_up_questions"], list)
    assert "support_level" in payload
    assert "limitations" in payload
    assert "routing_method" in payload


def test_answer_endpoint_handles_structured_question(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="structured",
            operation="fact",
            customer_name="Harbor Foods Demo Oy",
            field_name="equity_ratio_pct",
            needs_structured_data=True,
            confidence="high",
            reason="Planner matched a structured fact query.",
            structured_dataset="customer_portfolio",
        )
    )
    app = create_api_app(answer_service=answer_service_factory(planning_result=planning_result))
    client = TestClient(app)

    response = client.post("/answer", json={"question": "What is Harbor Foods Demo Oy's equity ratio?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "structured"
    assert payload["matched_customer_name"] == "Harbor Foods Demo Oy"
    assert payload["matched_field_name"] == "equity_ratio_pct"
    assert payload["structured_dataset"] == "customer_portfolio"
    assert "%" in payload["answer"]
