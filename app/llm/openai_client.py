import json

from openai import OpenAI

from llm.prompts import build_retrieval_messages
from models import RetrievalSynthesisResult, RetrievedChunk


class OpenAIRetrievalSynthesizer:
    def __init__(
        self,
        model: str,
        enabled: bool = True,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.api_key = api_key
        self._client: OpenAI | None = None

    def synthesize_retrieval_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RetrievalSynthesisResult:
        if not self.enabled:
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="disabled",
                failure_reason="LLM synthesis disabled by configuration.",
            )
        if not self.api_key:
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="missing_api_key",
                failure_reason="LLM synthesis unavailable: missing API key.",
            )

        try:
            client = self._get_client()
            response = client.responses.create(
                model=self.model,
                input=build_retrieval_messages(question=question, retrieved_chunks=retrieved_chunks),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "retrieval_answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                                "support_level": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                                "limitations": {"type": "string"},
                            },
                            "required": ["answer", "support_level", "limitations"],
                            "additionalProperties": False,
                        },
                    }
                },
            )
            payload = json.loads(response.output_text)
        except Exception:
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="api_error",
                failure_reason="LLM synthesis unavailable: API error.",
            )

        if not all(key in payload for key in ("answer", "support_level", "limitations")):
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="invalid_response",
                failure_reason="LLM synthesis unavailable: invalid response format.",
            )

        return RetrievalSynthesisResult(
            answer=str(payload["answer"]),
            support_level=str(payload["support_level"]),
            limitations=str(payload["limitations"]),
            synthesis_method="llm",
            status="success",
            failure_reason=None,
        )

    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def get_status(self) -> tuple[str, str]:
        if not self.enabled:
            return "disabled", "LLM synthesis disabled by configuration."
        if not self.api_key:
            return "missing_api_key", "LLM synthesis unavailable: missing API key."
        return "success", "OpenAI retrieval synthesis available."

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client
