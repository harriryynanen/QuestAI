from pathlib import Path

from app.models import DocumentLoadResult, DocumentRecord, RetrievalSynthesisResult


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


def test_retrieval_flow_can_cite_pdf_sources(
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
        answer="The PDF guidance highlights unresolved tax arrears as an escalation condition.",
        support_level="high",
        limitations="This answer is grounded only in retrieved document passages.",
        synthesis_method="llm",
        status="success",
        failure_reason=None,
    )
    service = answer_service_factory(
        planning_result=planning_result,
        retrieval_result=retrieval_result,
    )
    service.document_store.load_retrieval_bundle = lambda: DocumentLoadResult(
        documents=[
            DocumentRecord(
                document_id="policy_pdf",
                file_name="policy.pdf",
                path=Path("policy.pdf"),
                text="## Page 1\nUnresolved tax arrears should be escalated.",
                source_type="pdf",
            )
        ],
        issues=[],
    )

    response = service.answer_question("What does the policy say about tax arrears?")

    assert response.route == "retrieval"
    assert response.sources_used
    assert any("policy.pdf" in source for source in response.sources_used)


def test_retrieval_flow_can_cite_text_sources(
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
        answer="The text guidance notes that unresolved tax arrears should be escalated.",
        support_level="high",
        limitations="This answer is grounded only in retrieved document passages.",
        synthesis_method="llm",
        status="success",
        failure_reason=None,
    )
    service = answer_service_factory(
        planning_result=planning_result,
        retrieval_result=retrieval_result,
    )
    service.document_store.load_retrieval_bundle = lambda: DocumentLoadResult(
        documents=[
            DocumentRecord(
                document_id="policy_text",
                file_name="policy.txt",
                path=Path("policy.txt"),
                text="Unresolved tax arrears should be escalated.",
                source_type="text",
            )
        ],
        issues=[],
    )

    response = service.answer_question("What does the policy say about tax arrears?")

    assert response.route == "retrieval"
    assert response.sources_used
    assert any("policy.txt" in source for source in response.sources_used)


def test_answer_service_rejects_overly_long_question_before_planning(
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
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question("A" * 2001)

    assert response.route == "unknown"
    assert response.routing_method == "safe_fallback"
    assert "too long" in response.answer.lower()
    assert "lightweight input hygiene" in response.limitations.lower()


def test_answer_service_normalizes_small_control_character_noise_without_breaking_valid_question(
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

    response = service.answer_question("What\x00 does the policy say about tax arrears?\x07")

    assert response.route == "retrieval"
    assert response.answer == retrieval_result.answer
