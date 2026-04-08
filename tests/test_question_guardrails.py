from app.services.question_guardrails import QuestionGuardrails


def test_question_guardrails_normalize_small_control_character_noise():
    result = QuestionGuardrails.check("What\x00 is\tHarbor Foods Demo Oy's equity ratio?\x07")

    assert result.rejection_reason is None
    assert result.was_normalized is True
    assert result.normalized_question == "What is Harbor Foods Demo Oy's equity ratio?"


def test_question_guardrails_reject_overly_long_input():
    result = QuestionGuardrails.check("A" * 2001)

    assert result.rejection_reason is not None
    assert "too long" in result.rejection_reason.lower()


def test_question_guardrails_reject_control_character_heavy_input():
    result = QuestionGuardrails.check(("\x00\x01\x02\x03" * 7) + "policy?")

    assert result.rejection_reason is not None
    assert "control characters" in result.rejection_reason.lower()
