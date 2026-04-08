from models import AnswerResponse
from services.conversation_scope import ConversationScopeResolver


def test_conversation_scope_resolver_maps_structured_list_follow_up():
    resolver = ConversationScopeResolver()
    previous_response = AnswerResponse(
        answer="Matching customers: Bright Forge Test Ltd.",
        sources_used=["demo_customer_portfolio.csv | filter: has_tax_arrears"],
        support_level="high",
        limitations="This result reflects only the current CSV rows.",
        route="structured",
        retrieved_chunks=[],
        follow_up_questions=[],
        matched_customer_name=None,
        matched_customer_names=["Bright Forge Test Ltd"],
        matched_field_name="has_tax_arrears",
    )

    result = resolver.resolve_structured_follow_up(
        question="Who are they?",
        conversation_turns=[
            {"question": "Which customers have tax arrears?", "response": previous_response}
        ],
    )

    assert result is not None
    assert result.plan.route == "structured"
    assert result.plan.operation == "list"


def test_conversation_scope_resolver_maps_advisory_owner_reverse_lookup_follow_up():
    resolver = ConversationScopeResolver()
    previous_response = AnswerResponse(
        answer="Riverstone Demo Logistics Ltd's case is owned by Mika Salonen.",
        sources_used=["demo_advisory_case_pipeline.csv | row: Riverstone Demo Logistics Ltd | column: advisory_owner"],
        support_level="high",
        limitations="This answer is taken directly from the current advisory case CSV row.",
        route="structured",
        retrieved_chunks=[],
        follow_up_questions=[],
        matched_customer_name="Riverstone Demo Logistics Ltd",
        matched_customer_names=["Riverstone Demo Logistics Ltd"],
        matched_field_name="advisory_owner",
        matched_field_value="Mika Salonen",
        structured_dataset="advisory_case_pipeline",
    )

    result = resolver.resolve_structured_follow_up(
        question="What other companies does Mika have?",
        conversation_turns=[
            {
                "question": "Who is the advisory owner of Riverstone Demo Logistics Ltd?",
                "response": previous_response,
            }
        ],
    )

    assert result is not None
    assert result.plan.route == "structured"
    assert result.plan.operation == "filter"
    assert result.plan.field_name == "advisory_owner"
    assert result.plan.filter_value == "Mika Salonen"
    assert result.plan.structured_dataset == "advisory_case_pipeline"


def test_conversation_scope_resolver_reuses_previous_group_scope_and_product():
    resolver = ConversationScopeResolver()
    previous_response = AnswerResponse(
        answer="Preliminary grouped view for FlexLine Demo: Broadly aligned based on available evidence.",
        sources_used=["demo_customer_portfolio.csv"],
        support_level="medium",
        limitations="Preliminary view only.",
        route="combined",
        retrieved_chunks=[],
        follow_up_questions=[],
        matched_customer_name=None,
        matched_customer_names=["Bright Forge Test Ltd", "Maple Works Mock Ltd"],
        matched_field_name=None,
        structured_dataset="customer_portfolio",
    )

    result = resolver.resolve_group_combined_follow_up(
        question="Which companies are not",
        conversation_turns=[
            {
                "question": "Which of those customers appear broadly aligned with FlexLine Demo?",
                "response": previous_response,
            }
        ],
    )

    assert result is not None
    assert result.plan.route == "combined"
    assert result.plan.operation == "preliminary_assessment"
    assert result.plan.product_name == "FlexLine Demo"


def test_conversation_scope_resolver_resolves_customer_names_from_combined_scope(
    dataframe,
):
    resolver = ConversationScopeResolver()
    previous_response = AnswerResponse(
        answer="Preliminary grouped view for AssetGrow Demo: Caution / mixed signals.",
        sources_used=["demo_customer_portfolio.csv"],
        support_level="medium",
        limitations="Preliminary view only.",
        route="combined",
        retrieved_chunks=[],
        follow_up_questions=[],
        matched_customer_name=None,
        matched_customer_names=["Bright Forge Test Ltd", "Maple Works Mock Ltd"],
        matched_field_name=None,
        structured_dataset="customer_portfolio",
    )

    names = resolver.resolve_customer_names_from_context(
        question="Which of those customers appear broadly aligned with AssetGrow Demo?",
        conversation_turns=[
            {
                "question": "Which customers have tax arrears?",
                "response": previous_response,
            }
        ],
        dataframe=dataframe,
    )

    assert names == ["Bright Forge Test Ltd", "Maple Works Mock Ltd"]
