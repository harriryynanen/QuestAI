import pytest


def test_advisory_case_escalation_filter_is_supported(
    structured_query_engine,
    advisory_dataframe,
    advisory_dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="filter",
        field_name="escalation_flag",
        filter_value="yes",
        needs_structured_data=True,
        structured_dataset="advisory_case_pipeline",
    )

    result = structured_query_engine.answer(
        question="Which cases have escalation flag?",
        dataframe=advisory_dataframe,
        dataset_file_name=advisory_dataset_file_name,
        plan=plan,
        dataset_name="advisory_case_pipeline",
    )

    assert result.support_level == "high"
    assert "Aurora Mock Services Oy" in result.answer
    assert "Maple Works Mock Ltd" in result.answer
    assert result.matched_field_name == "escalation_flag"


def test_advisory_case_owner_lookup_by_customer_name_is_supported(
    structured_query_engine,
    advisory_dataframe,
    advisory_dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="fact",
        customer_name="Harbor Foods Demo Oy",
        field_name="advisory_owner",
        needs_structured_data=True,
        structured_dataset="advisory_case_pipeline",
    )

    result = structured_query_engine.answer(
        question="Who owns Harbor Foods Demo Oy's case?",
        dataframe=advisory_dataframe,
        dataset_file_name=advisory_dataset_file_name,
        plan=plan,
        dataset_name="advisory_case_pipeline",
    )

    assert result.support_level == "high"
    assert "Mika Salonen" in result.answer
    assert result.matched_customer_name == "Harbor Foods Demo Oy"
    assert result.matched_field_name == "advisory_owner"


def test_advisory_case_support_level_filter_is_supported(
    structured_query_engine,
    advisory_dataframe,
    advisory_dataset_file_name,
):
    result = structured_query_engine.answer(
        question="Which cases have support level not_enough_information?",
        dataframe=advisory_dataframe,
        dataset_file_name=advisory_dataset_file_name,
        plan=None,
        dataset_name="advisory_case_pipeline",
    )

    assert result.support_level == "high"
    assert "Maple Works Mock Ltd" in result.answer
    assert result.matched_field_name == "support_level"


def test_advisory_case_open_product_count_is_supported(
    structured_query_engine,
    advisory_dataframe,
    advisory_dataset_file_name,
):
    result = structured_query_engine.answer(
        question="How many open InvoiceBridge Demo cases are there?",
        dataframe=advisory_dataframe,
        dataset_file_name=advisory_dataset_file_name,
        plan=None,
        dataset_name="advisory_case_pipeline",
    )

    assert result.support_level == "high"
    assert "3 open InvoiceBridge Demo cases" in result.answer


def test_advisory_case_next_action_lookup_is_supported(
    structured_query_engine,
    advisory_dataframe,
    advisory_dataset_file_name,
):
    result = structured_query_engine.answer(
        question="What is the next action for Maple Works Mock Ltd's case?",
        dataframe=advisory_dataframe,
        dataset_file_name=advisory_dataset_file_name,
        plan=None,
        dataset_name="advisory_case_pipeline",
    )

    assert result.support_level == "high"
    assert "Request latest financial statements" in result.answer
    assert result.matched_customer_name == "Maple Works Mock Ltd"
    assert result.matched_field_name == "next_action"


def test_answer_service_routes_advisory_case_questions_to_structured(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="unknown",
            operation="unknown",
            confidence="low",
            reason="Use fallback behavior in tests.",
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question("Which cases have escalation flag?")

    assert response.route == "structured"
    assert response.support_level == "high"
    assert "Aurora Mock Services Oy" in response.answer


def test_answer_service_can_infer_advisory_dataset_from_planned_field(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="structured",
            operation="fact",
            customer_name="Harbor Foods Demo Oy",
            field_name="advisory_owner",
            needs_structured_data=True,
            confidence="high",
            reason="Planner resolved the target field but omitted the dataset.",
            structured_dataset=None,
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question("Who is responsible for Harbor Foods Demo Oy's case?")

    assert response.route == "structured"
    assert response.support_level == "high"
    assert "Mika Salonen" in response.answer
    assert response.matched_field_name == "advisory_owner"


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        (
            "Who is the advisory owner of Harbor Foods Demo Oy?",
            {
                "operation": "fact",
                "customer_name": "Harbor Foods Demo Oy",
                "field_name": "advisory_owner",
                "structured_dataset": "advisory_case_pipeline",
            },
        ),
        (
            "Who is the advisory_owner of Harbor Foods Demo Oy?",
            {
                "operation": "fact",
                "customer_name": "Harbor Foods Demo Oy",
                "field_name": "advisory_owner",
                "structured_dataset": "advisory_case_pipeline",
            },
        ),
        (
            "advisory_owner of Harbor Foods Demo Oy",
            {
                "operation": "fact",
                "customer_name": "Harbor Foods Demo Oy",
                "field_name": "advisory_owner",
                "structured_dataset": "advisory_case_pipeline",
            },
        ),
        (
            "Who is responsible for Harbor Foods Demo Oy's case?",
            {
                "operation": "fact",
                "customer_name": "Harbor Foods Demo Oy",
                "field_name": "advisory_owner",
                "structured_dataset": "advisory_case_pipeline",
            },
        ),
        (
            "Who owns Harbor Foods Demo Oy's case?",
            {
                "operation": "fact",
                "customer_name": "Harbor Foods Demo Oy",
                "field_name": "advisory_owner",
                "structured_dataset": "advisory_case_pipeline",
            },
        ),
    ],
)
def test_answer_service_uses_semantic_plan_for_advisory_owner_variants(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
    question,
    plan,
):
    planning_result = planning_result_factory(
        plan_factory(
            route="structured",
            needs_structured_data=True,
            confidence="high",
            reason="Planner mapped the advisory case owner question to the case pipeline schema.",
            **plan,
        )
    )
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question(question)

    assert response.route == "structured"
    assert response.support_level == "high"
    assert "Mika Salonen" in response.answer
    assert response.matched_customer_name == "Harbor Foods Demo Oy"
    assert response.matched_field_name == "advisory_owner"
    assert response.sources_used == [
        "demo_advisory_case_pipeline.csv | row: Harbor Foods Demo Oy | column: advisory_owner"
    ]


@pytest.mark.parametrize(
    ("question", "plan_kwargs", "expected_snippet", "expected_field"),
    [
        (
            "What is the next action for Maple Works Mock Ltd's case?",
            {
                "route": "structured",
                "operation": "fact",
                "customer_name": "Maple Works Mock Ltd",
                "field_name": "next_action",
                "needs_structured_data": True,
                "structured_dataset": "advisory_case_pipeline",
                "confidence": "high",
                "reason": "Planner matched an advisory case next-action lookup.",
            },
            "Request latest financial statements",
            "next_action",
        ),
        (
            "Which cases have support level not_enough_information?",
            {
                "route": "structured",
                "operation": "filter",
                "field_name": "support_level",
                "filter_value": "not_enough_information",
                "needs_structured_data": True,
                "structured_dataset": "advisory_case_pipeline",
                "confidence": "high",
                "reason": "Planner matched an advisory case support-level filter.",
            },
            "Maple Works Mock Ltd",
            "support_level",
        ),
        (
            "How many open InvoiceBridge Demo cases are there?",
            {
                "route": "structured",
                "operation": "count",
                "field_name": "requested_product",
                "product_name": "InvoiceBridge Demo",
                "filter_value": "open",
                "needs_structured_data": True,
                "structured_dataset": "advisory_case_pipeline",
                "confidence": "high",
                "reason": "Planner matched an advisory case open-count request by product.",
            },
            "3 matching cases",
            "requested_product",
        ),
    ],
)
def test_answer_service_uses_semantic_plan_for_advisory_case_queries(
    answer_service_factory,
    plan_factory,
    planning_result_factory,
    question,
    plan_kwargs,
    expected_snippet,
    expected_field,
):
    planning_result = planning_result_factory(plan_factory(**plan_kwargs))
    service = answer_service_factory(planning_result=planning_result)

    response = service.answer_question(question)

    assert response.route == "structured"
    assert response.support_level == "high"
    assert expected_snippet in response.answer
    assert response.matched_field_name == expected_field
