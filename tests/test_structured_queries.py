from app.structured.query_engine import StructuredQueryEngine


def test_structured_fact_query_with_semantic_plan(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="fact",
        customer_name="Harbor Foods Demo Oy",
        field_name="equity_ratio_pct",
        needs_structured_data=True,
    )

    result = structured_query_engine.answer(
        question="What is Harbor Foods Demo Oy's equity ratio?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
    )

    assert result.support_level == "high"
    assert result.matched_customer_name == "Harbor Foods Demo Oy"
    assert result.matched_field_name == "equity_ratio_pct"
    assert dataset_file_name in result.sources_used[0]


def test_structured_fact_query_with_missing_planned_field_returns_safe_response(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="fact",
        customer_name="Harbor Foods Demo Oy",
        field_name="advisory_owner",
        needs_structured_data=True,
        structured_dataset="customer_portfolio",
    )

    result = structured_query_engine.answer(
        question="Who is responsible for Harbor Foods Demo Oy's case?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
        dataset_name="customer_portfolio",
    )

    assert result.support_level == "low"
    assert "planned field does not exist" in result.answer.lower()
    assert result.matched_customer_name == "Harbor Foods Demo Oy"
    assert result.matched_field_name == "advisory_owner"


def test_structured_filter_query_returns_matching_customers(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="filter",
        field_name="has_tax_arrears",
        filter_value="yes",
        needs_structured_data=True,
    )

    result = structured_query_engine.answer(
        question="Which customers have tax arrears?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
    )

    assert result.support_level == "high"
    assert "Bright Forge Test Ltd" in result.answer
    assert result.matched_field_name == "has_tax_arrears"


def test_structured_comparison_query_returns_expected_customer(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="comparison",
        field_name="latest_revenue_eur",
        comparison_direction="highest",
        needs_structured_data=True,
    )

    result = structured_query_engine.answer(
        question="Which customer has the highest turnover?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
    )

    assert result.support_level == "high"
    assert result.matched_customer_name == "Riverstone Demo Logistics Ltd"
    assert result.matched_field_name == "latest_revenue_eur"


def test_structured_count_query_is_supported(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="count",
        needs_structured_data=True,
    )

    result = structured_query_engine.answer(
        question="How many customers do I have?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
    )

    assert result.support_level == "high"
    assert "10 customers" in result.answer


def test_structured_list_query_is_supported(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="list",
        needs_structured_data=True,
    )

    result = structured_query_engine.answer(
        question="List the customers",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
    )

    assert result.support_level == "high"
    assert "Harbor Foods Demo Oy" in result.answer
    assert result.matched_customer_names is not None
    assert len(result.matched_customer_names) == len(dataframe.index)


def test_structured_exists_query_is_supported(
    structured_query_engine,
    dataframe,
    dataset_file_name,
    plan_factory,
):
    plan = plan_factory(
        route="structured",
        operation="exists",
        field_name="latest_financials_available",
        filter_value="no",
        needs_structured_data=True,
    )

    result = structured_query_engine.answer(
        question="Does any customer have missing financial statements?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=plan,
    )

    assert result.support_level == "high"
    assert result.answer.startswith("Yes.")
    assert "Maple Works Mock Ltd" in result.answer


def test_finnish_heuristic_fact_query_still_works(
    structured_query_engine,
    dataframe,
    dataset_file_name,
):
    result = structured_query_engine.answer(
        question="Mika on Harbor Foods Demo Oy:n omavaraisuusaste?",
        dataframe=dataframe,
        dataset_file_name=dataset_file_name,
        plan=None,
    )

    assert result.support_level in {"medium", "high"}
    assert result.matched_customer_name == "Harbor Foods Demo Oy"
    assert result.matched_field_name == "equity_ratio_pct"


def test_count_followed_by_name_those_customers_lists_current_dataset(
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

    first_response = service.answer_question("How many customers are there in the data?")
    follow_up_response = service.answer_question(
        "Name those customers",
        conversation_turns=[{"question": "How many customers are there in the data?", "response": first_response}],
    )

    assert first_response.route == "structured"
    assert "10 customers" in first_response.answer
    assert follow_up_response.route == "structured"
    assert "Customer names in scope:" in follow_up_response.answer
    assert "Harbor Foods Demo Oy" in follow_up_response.answer
    assert "10 customers" not in follow_up_response.answer


def test_filter_follow_up_who_are_they_lists_matching_customers(
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

    first_response = service.answer_question("Which customers have tax arrears?")
    follow_up_response = service.answer_question(
        "Who are they?",
        conversation_turns=[{"question": "Which customers have tax arrears?", "response": first_response}],
    )

    assert first_response.route == "structured"
    assert "Bright Forge Test Ltd" in first_response.answer
    assert follow_up_response.route == "structured"
    assert "Customer names in scope:" in follow_up_response.answer
    assert "Bright Forge Test Ltd" in follow_up_response.answer


def test_ambiguous_list_follow_up_without_reference_is_handled_safely(
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

    response = service.answer_question("Name those customers")

    assert response.route == "unknown" or response.route == "structured"
    assert response.support_level == "low"
    assert "could not determine" in response.answer.lower() or "could not route" in response.answer.lower()
