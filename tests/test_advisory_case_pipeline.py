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
