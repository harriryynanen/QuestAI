from models import AnswerResponse, RetrievalSynthesisResult


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
    assert any(source.endswith("Page 1") or source.endswith("Page 2") or ".md --" in source for source in response.sources_used)
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


def test_group_combined_assessment_uses_referenced_customer_scope(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Use scoped follow-up fallback in tests.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    first_response = service.answer_question("Which customers have tax arrears?")
    grouped_response = service.answer_question(
        "Which of those customers appear broadly aligned with AssetGrow Demo?",
        conversation_turns=[{"question": "Which customers have tax arrears?", "response": first_response}],
    )

    assert grouped_response.route == "combined"
    assert "Broadly aligned" in grouped_response.answer or "Caution / mixed signals" in grouped_response.answer
    assert "Bright Forge Test Ltd" in grouped_response.answer
    assert "eligible" not in grouped_response.answer.lower()
    assert "approved" not in grouped_response.answer.lower()
    assert any(".csv | row:" in source for source in grouped_response.sources_used)


def test_group_combined_assessment_reports_unresolved_scope_clearly(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Use scoped follow-up fallback in tests.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question(
        "Which of those customers appear broadly aligned with AssetGrow Demo?"
    )

    assert response.route in {"combined", "unknown"}
    assert response.support_level == "low"
    assert "could not determine" in response.answer.lower() or "identify the customer set" in response.limitations.lower()


def test_grouped_combined_follow_up_reuses_previous_product_and_scope(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Use scoped follow-up fallback in tests.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    first_response = service.answer_question("Which customers have tax arrears?")
    grouped_response = service.answer_question(
        "Which of those customers appear broadly aligned with FlexLine Demo?",
        conversation_turns=[{"question": "Which customers have tax arrears?", "response": first_response}],
    )
    follow_up_response = service.answer_question(
        "Which companies are not",
        conversation_turns=[
            {"question": "Which customers have tax arrears?", "response": first_response},
            {
                "question": "Which of those customers appear broadly aligned with FlexLine Demo?",
                "response": grouped_response,
            },
        ],
    )

    assert grouped_response.route == "combined"
    assert follow_up_response.route == "combined"
    assert "FlexLine Demo" in follow_up_response.answer
    assert "companies that show caution" in follow_up_response.answer.lower() or "not enough information" in follow_up_response.answer.lower()
    assert "could not route" not in follow_up_response.answer.lower()


def test_dataset_wide_combined_follow_up_can_infer_scope_from_previous_question(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Use scoped follow-up fallback in tests.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    previous_response = AnswerResponse(
        answer="Preliminary grouped view for FlexLine Demo: Broadly aligned based on available evidence.",
        sources_used=["demo_customer_portfolio.csv"],
        support_level="medium",
        limitations="Preliminary view only.",
        route="combined",
        retrieved_chunks=[],
        follow_up_questions=[],
        matched_customer_name=None,
        matched_customer_names=None,
        matched_field_name=None,
        synthesis_method="deterministic",
        synthesis_status=None,
        synthesis_status_message=None,
        routing_method="rules",
        routing_confidence="medium",
        routing_reason="Test previous grouped combined turn.",
        planning_method="heuristic_fallback",
        planning_reason="Test previous grouped combined turn.",
    )

    follow_up_response = service.answer_question(
        "Which companies are not",
        conversation_turns=[
            {
                "question": "Are all of the companies in the data eligible for FlexLine Demo?",
                "response": previous_response,
            }
        ],
    )

    assert follow_up_response.route == "combined"
    assert "FlexLine Demo" in follow_up_response.answer
    assert "could not route" not in follow_up_response.answer.lower()
    assert follow_up_response.support_level in {"medium", "high"}
