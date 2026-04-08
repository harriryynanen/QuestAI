from llm.openai_client import OpenAIAppClient
from models import SemanticPlanningResult, SemanticQueryPlan


class StructuredQueryPlanner:
    def __init__(self, llm_client: OpenAIAppClient) -> None:
        self.llm_client = llm_client

    def plan(
        self,
        question: str,
        conversation_context: str | None = None,
    ) -> SemanticPlanningResult:
        return self.llm_client.plan_question(
            question=question,
            conversation_context=conversation_context,
        )

    @staticmethod
    def heuristic_fallback_plan(reason: str) -> SemanticQueryPlan:
        return SemanticQueryPlan(
            route="unknown",
            operation="unknown",
            field_name=None,
            customer_name=None,
            product_name=None,
            document_topic=None,
            comparison_direction=None,
            filter_value=None,
            needs_documents=False,
            needs_structured_data=True,
            reason=reason,
            confidence="low",
            method="heuristic_fallback",
            structured_dataset=None,
        )
