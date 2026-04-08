from app.ui.components import build_dynamic_prompt_chips


def test_starter_prompt_chips_include_retrieval_structured_and_combined_examples():
    example_questions = [
        "What does the policy say about tax arrears?",
        "What is Harbor Foods Demo Oy's equity ratio?",
        "Which customer has the highest turnover?",
        "Based on the product guide and customer data, does Harbor Foods Demo Oy appear broadly aligned with InvoiceBridge Demo?",
    ]

    chips = build_dynamic_prompt_chips([], example_questions)

    assert len(chips) == 3
    assert any("policy" in chip.lower() for chip in chips)
    assert any("equity ratio" in chip.lower() or "highest turnover" in chip.lower() for chip in chips)
    assert any("based on" in chip.lower() and "data" in chip.lower() for chip in chips)
