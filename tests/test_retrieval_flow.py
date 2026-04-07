from models import RetrievalSynthesisResult


def test_retrieval_flow_returns_grounded_sources(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="retrieval",
            operation="policy_lookup",
            document_topic="tax arrears",
            needs_documents=True,
            confidence="high",
            reason="The question asks for document guidance.",
        )
    )
    retrieval_result = RetrievalSynthesisResult(
        answer="The policy treats unresolved tax arrears as a negative screening condition.",
        support_level="high",
        limitations="This answer is grounded only in retrieved markdown passages.",
        synthesis_method="llm",
        status="success",
        failure_reason=None,
    )
    service = answer_service_factory(
        planning_result=planning_result,
        retrieval_result=retrieval_result,
    )

    response = service.answer_question("What does the policy say about tax arrears?")

    assert response.route == "retrieval"
    assert response.synthesis_method == "llm"
    assert response.retrieved_chunks
    assert response.sources_used
    assert any(source.endswith("General exclusions") or "tax arrears" in source.lower() for source in response.sources_used)
    assert response.limitations


def test_retrieval_flow_falls_back_safely_when_llm_is_unavailable(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="retrieval",
            operation="policy_lookup",
            document_topic="payment delays",
            needs_documents=True,
            confidence="high",
            reason="The question asks for document guidance.",
        )
    )
    fallback_result = RetrievalSynthesisResult(
        answer="",
        support_level="low",
        limitations="",
        synthesis_method="fallback",
        status="missing_api_key",
        failure_reason="LLM synthesis unavailable: missing API key.",
    )
    service = answer_service_factory(
        planning_result=planning_result,
        retrieval_result=fallback_result,
    )

    response = service.answer_question("What does the policy say about payment delays?")

    assert response.route == "retrieval"
    assert response.synthesis_method == "fallback"
    assert response.synthesis_status == "missing_api_key"
    assert response.synthesis_status_message == "LLM synthesis unavailable: missing API key."
    assert response.sources_used
