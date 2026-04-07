import json

from openai import OpenAI

from llm.prompts import (
    build_combined_answer_messages,
    build_retrieval_messages,
    build_semantic_plan_messages,
)
from models import (
    CombinedEvidence,
    RetrievalSynthesisResult,
    RetrievedChunk,
    SemanticPlanningResult,
    SemanticQueryPlan,
)


class OpenAIAppClient:
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

    def plan_question(
        self,
        question: str,
    ) -> SemanticPlanningResult:
        fallback_plan = SemanticQueryPlan(
            route="unknown",
            operation="unknown",
            customer_name=None,
            field_name=None,
            product_name=None,
            document_topic=None,
            comparison_direction=None,
            filter_value=None,
            needs_documents=False,
            needs_structured_data=False,
            confidence="low",
            reason="Semantic planning unavailable.",
            method="llm",
        )
        if not self.enabled:
            return SemanticPlanningResult(
                plan=fallback_plan,
                status="disabled",
                failure_reason="Semantic planning disabled by configuration.",
            )
        if not self.api_key:
            return SemanticPlanningResult(
                plan=fallback_plan,
                status="missing_api_key",
                failure_reason="Semantic planning unavailable: missing API key.",
            )

        try:
            client = self._get_client()
            response = client.responses.create(
                model=self.model,
                input=build_semantic_plan_messages(question=question),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "semantic_plan",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "route": {
                                    "type": "string",
                                    "enum": ["retrieval", "structured", "combined", "unknown"],
                                },
                                "operation": {
                                    "type": "string",
                                    "enum": [
                                        "fact",
                                        "filter",
                                        "comparison",
                                        "count",
                                        "exists",
                                        "policy_lookup",
                                        "product_guidance",
                                        "preliminary_assessment",
                                        "unknown",
                                    ],
                                },
                                "customer_name": {"type": ["string", "null"]},
                                "field_name": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "latest_revenue_eur",
                                        "ebitda_eur",
                                        "ebitda_margin_pct",
                                        "equity_ratio_pct",
                                        "debt_to_ebitda",
                                        "years_in_operation",
                                        "b2b_invoicing_pct",
                                        "export_sales_pct",
                                        "has_tax_arrears",
                                        "latest_financials_available",
                                        "payment_delays_12m",
                                        "largest_customer_share_pct",
                                        "requested_product_interest",
                                        None,
                                    ],
                                },
                                "product_name": {"type": ["string", "null"]},
                                "document_topic": {"type": ["string", "null"]},
                                "comparison_direction": {
                                    "type": ["string", "null"],
                                    "enum": ["highest", "lowest", None],
                                },
                                "filter_value": {"type": ["string", "null"]},
                                "needs_documents": {"type": "boolean"},
                                "needs_structured_data": {"type": "boolean"},
                                "confidence": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "route",
                                "operation",
                                "customer_name",
                                "field_name",
                                "product_name",
                                "document_topic",
                                "comparison_direction",
                                "filter_value",
                                "needs_documents",
                                "needs_structured_data",
                                "confidence",
                                "reason",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
            )
            payload = json.loads(response.output_text)
        except Exception:
            return SemanticPlanningResult(
                plan=fallback_plan,
                status="api_error",
                failure_reason="Semantic planning unavailable: API error.",
            )

        required_keys = {
            "route",
            "operation",
            "customer_name",
            "field_name",
            "product_name",
            "document_topic",
            "comparison_direction",
            "filter_value",
            "needs_documents",
            "needs_structured_data",
            "confidence",
            "reason",
        }
        if not required_keys.issubset(payload.keys()):
            return SemanticPlanningResult(
                plan=fallback_plan,
                status="invalid_response",
                failure_reason="Semantic planning unavailable: invalid response format.",
            )

        return SemanticPlanningResult(
            plan=SemanticQueryPlan(
                route=str(payload["route"]),
                operation=str(payload["operation"]),
                customer_name=payload["customer_name"],
                field_name=payload["field_name"],
                product_name=payload["product_name"],
                document_topic=payload["document_topic"],
                comparison_direction=payload["comparison_direction"],
                filter_value=payload["filter_value"],
                needs_documents=bool(payload["needs_documents"]),
                needs_structured_data=bool(payload["needs_structured_data"]),
                confidence=str(payload["confidence"]),
                reason=str(payload["reason"]),
                method="llm",
            ),
            status="success",
            failure_reason=None,
        )

    def synthesize_combined_answer(
        self,
        question: str,
        evidence: CombinedEvidence,
        document_evidence: list[str],
    ) -> RetrievalSynthesisResult:
        if not self.enabled:
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="disabled",
                failure_reason="Combined synthesis disabled by configuration.",
            )
        if not self.api_key:
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="missing_api_key",
                failure_reason="Combined synthesis unavailable: missing API key.",
            )

        try:
            client = self._get_client()
            response = client.responses.create(
                model=self.model,
                input=build_combined_answer_messages(
                    question=question,
                    document_evidence=document_evidence,
                    structured_evidence=evidence.summary,
                    missing_information=evidence.missing_information,
                ),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "combined_answer",
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
                failure_reason="Combined synthesis unavailable: API error.",
            )

        if not all(key in payload for key in ("answer", "support_level", "limitations")):
            return RetrievalSynthesisResult(
                answer="",
                support_level="low",
                limitations="",
                synthesis_method="fallback",
                status="invalid_response",
                failure_reason="Combined synthesis unavailable: invalid response format.",
            )

        return RetrievalSynthesisResult(
            answer=str(payload["answer"]),
            support_level=str(payload["support_level"]),
            limitations=str(payload["limitations"]),
            synthesis_method="llm",
            status="success",
            failure_reason=None,
        )
