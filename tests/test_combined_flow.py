from models import RetrievalSynthesisResult


def test_combined_flow_assembles_document_and_structured_sources(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="combined",
            operation="preliminary_assessment",
            customer_name="Maple Works Mock Ltd",
            product_name="FlexLine Demo",
            document_topic="FlexLine Demo criteria",
            needs_documents=True,
            needs_structured_data=True,
            confidence="high",
            reason="The question asks for policy-plus-customer assessment.",
        )
    )
    combined_result = RetrievalSynthesisResult(
        answer=(
            "Based on the available policy and customer evidence, Maple Works Mock Ltd appears only partially "
            "supported for FlexLine Demo. Missing financial statements limit confidence."
        ),
        support_level="high",
        limitations="This is preliminary decision support only, not a final approval decision.",
        synthesis_method="llm",
        status="success",
        failure_reason=None,
    )
    service = answer_service_factory(
        planning_result=planning_result,
        combined_result=combined_result,
    )

    response = service.answer_question(
        "Based on the policy and customer data, does Maple Works Mock Ltd appear to meet the FlexLine Demo criteria?"
    )

    assert response.route == "combined"
    assert response.sources_used
    assert any(".md --" in source for source in response.sources_used)
    assert any(".csv | row:" in source for source in response.sources_used)
    assert response.limitations
    assert "approval" in response.limitations.lower()
    assert "approved" not in response.answer.lower()
    assert response.support_level in {"medium", "high"}


def test_combined_support_is_capped_when_structured_evidence_is_missing(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="combined",
            operation="preliminary_assessment",
            customer_name=None,
            product_name="AssetGrow Demo",
            document_topic="AssetGrow Demo criteria",
            needs_documents=True,
            needs_structured_data=True,
            confidence="high",
            reason="The question asks for combined assessment but no customer is identified.",
        )
    )
    combined_result = RetrievalSynthesisResult(
        answer="The available policy suggests caution, but the customer evidence is incomplete.",
        support_level="high",
        limitations="This is preliminary decision support only.",
        synthesis_method="llm",
        status="success",
        failure_reason=None,
    )
    service = answer_service_factory(
        planning_result=planning_result,
        combined_result=combined_result,
    )

    response = service.answer_question(
        "Based on the policy and customer data, is this customer a fit for AssetGrow Demo?"
    )

    assert response.route == "combined"
    assert response.support_level == "low"
    assert response.limitations
