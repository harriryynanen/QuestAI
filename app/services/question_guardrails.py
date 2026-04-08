from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionGuardrailResult:
    normalized_question: str
    was_normalized: bool
    rejection_reason: str | None = None


class QuestionGuardrails:
    """Lightweight hygiene checks for LLM-facing user questions."""

    MAX_QUESTION_LENGTH = 2000
    MAX_CONTROL_CHARACTERS = 24

    @classmethod
    def check(cls, question: str) -> QuestionGuardrailResult:
        if len(question) > cls.MAX_QUESTION_LENGTH:
            return QuestionGuardrailResult(
                normalized_question="",
                was_normalized=False,
                rejection_reason=(
                    "The question is too long for this demo. Please shorten it and ask one focused question at a time."
                ),
            )

        control_character_count = sum(
            1 for character in question if ord(character) < 32 and character not in ("\n", "\r", "\t")
        )
        normalized_question = "".join(
            character
            if ord(character) >= 32 or character in ("\n", "\r", "\t")
            else " "
            for character in question
        )
        normalized_question = " ".join(normalized_question.split())
        was_normalized = normalized_question != question.strip()

        if control_character_count > cls.MAX_CONTROL_CHARACTERS:
            return QuestionGuardrailResult(
                normalized_question="",
                was_normalized=was_normalized,
                rejection_reason=(
                    "The question contains too many malformed control characters for safe processing. "
                    "Please rephrase it as a normal business question."
                ),
            )

        if not normalized_question:
            return QuestionGuardrailResult(
                normalized_question="",
                was_normalized=was_normalized,
                rejection_reason="Please enter a readable business question.",
            )

        return QuestionGuardrailResult(
            normalized_question=normalized_question,
            was_normalized=was_normalized,
            rejection_reason=None,
        )
