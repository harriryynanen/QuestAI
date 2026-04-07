from structured.query_engine import StructuredQueryEngine


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
