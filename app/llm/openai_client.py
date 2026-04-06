import json
import os

from openai import OpenAI

from llm.prompts import build_retrieval_messages
from models import RetrievalSynthesisResult, RetrievedChunk


class OpenAIRetrievalSynthesizer:
    def __init__(self, model: str, enabled: bool = True) -> None:
        self.model = model
        self.enabled = enabled
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client: OpenAI | None = None

    def synthesize_retrieval_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RetrievalSynthesisResult | None:
        if not self.enabled or not self.api_key:
            return None

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
            return None

        return RetrievalSynthesisResult(
            answer=str(payload["answer"]),
            support_level=str(payload["support_level"]),
            limitations=str(payload["limitations"]),
            synthesis_method="llm",
        )

    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client
