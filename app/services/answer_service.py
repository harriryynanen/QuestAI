from llm.openai_client import OpenAIRetrievalSynthesizer
from models import (
    AnswerResponse,
    CombinedEvidence,
    RetrievalSynthesisResult,
    RetrievedChunk,
    RoutingDecision,
    SemanticPlanningResult,
    SemanticQueryPlan,
)
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.router import RuleBasedRouter, is_confident_routing_decision
from structured.customer_data import CustomerDataLoader
from structured.planner import StructuredQueryPlanner
from structured.query_engine import StructuredQueryEngine


class AnswerService:
    def __init__(
        self,
        router: RuleBasedRouter,
        document_store: DocumentStore,
        customer_data_loader: CustomerDataLoader,
        chunker: MarkdownChunker,
        retriever: KeywordRetriever,
        structured_query_engine: StructuredQueryEngine,
        retrieval_synthesizer: OpenAIRetrievalSynthesizer,
        structured_query_planner: StructuredQueryPlanner,
        retrieval_context_max_characters: int,
    ) -> None:
        self.router = router
        self.document_store = document_store
        self.customer_data_loader = customer_data_loader
        self.chunker = chunker
        self.retriever = retriever
        self.structured_query_engine = structured_query_engine
        self.retrieval_synthesizer = retrieval_synthesizer
        self.structured_query_planner = structured_query_planner
        self.retrieval_context_max_characters = retrieval_context_max_characters

    def answer_question(self, question: str) -> AnswerResponse:
        planning_result = self.structured_query_planner.plan(question)
        routing_decision = self._route_question(question, planning_result)
        route = routing_decision.route
        dataset_info = self.customer_data_loader.get_data_info()
        dataframe = self.customer_data_loader.get_dataframe()
        dataset_file_name = self.customer_data_loader.get_dataset_file_name()
        markdown_documents = self.document_store.load_markdown_documents()
        chunks = self.chunker.chunk_documents(markdown_documents)
        semantic_plan = planning_result.plan if planning_result.status == "success" else None

        if route == "unknown":
            return self._build_unclear_routing_response(routing_decision)

        if route == "retrieval":
            retrieved_chunks = self.retriever.retrieve(question=question, chunks=chunks)
            return self._build_retrieval_response(question, retrieved_chunks, routing_decision)

        if route == "structured":
            structured_result = self.structured_query_engine.answer(
                question=question,
                dataframe=dataframe,
                dataset_file_name=dataset_file_name,
                plan=semantic_plan,
            )
            planning_reason = structured_result.planning_reason
            if planning_result.status != "success":
                planning_reason = (
                    f"{structured_result.planning_reason} Planner fallback reason: {planning_result.failure_reason}"
                    if structured_result.planning_reason
                    else planning_result.failure_reason
                )
            return AnswerResponse(
                answer=structured_result.answer,
                sources_used=structured_result.sources_used,
                support_level=structured_result.support_level,
                limitations=structured_result.limitations,
                route="structured",
                retrieved_chunks=[],
                follow_up_questions=self._get_follow_up_questions(
                    route="structured",
                    matched_customer_name=structured_result.matched_customer_name,
                    matched_field_name=structured_result.matched_field_name,
                ),
                matched_customer_name=structured_result.matched_customer_name,
                matched_field_name=structured_result.matched_field_name,
                synthesis_method="deterministic",
                synthesis_status=None,
                synthesis_status_message=None,
                routing_method=routing_decision.method,
                routing_confidence=routing_decision.confidence,
                routing_reason=routing_decision.reason,
                planning_method=structured_result.planning_method,
                planning_reason=planning_reason,
            )

        return self._build_combined_response(
            question=question,
            routing_decision=routing_decision,
            planning_result=planning_result,
            semantic_plan=semantic_plan,
            chunks=chunks,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            dataset_info_file_name=dataset_info.file_name if dataset_info.dataset_found else None,
        )

    def _build_retrieval_response(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        routing_decision: RoutingDecision,
    ) -> AnswerResponse:
        if not retrieved_chunks:
            return AnswerResponse(
                answer=(
                    "I did not find a clearly relevant markdown passage for this question in the current demo documents."
                ),
                sources_used=[],
                support_level="low",
                limitations=(
                    "This step uses simple keyword overlap on markdown chunks only. "
                    "A missing match does not prove the documents contain no relevant guidance."
                ),
                route="retrieval",
                retrieved_chunks=[],
                follow_up_questions=self._get_follow_up_questions(route="retrieval"),
                matched_customer_name=None,
                matched_field_name=None,
                synthesis_method="fallback",
                synthesis_status=None,
                synthesis_status_message="No relevant chunks found for retrieval synthesis.",
                routing_method=routing_decision.method,
                routing_confidence=routing_decision.confidence,
                routing_reason=routing_decision.reason,
                planning_method=None,
                planning_reason=None,
            )

        selected_chunks = self._select_retrieval_chunks(retrieved_chunks)
        llm_result = self.retrieval_synthesizer.synthesize_retrieval_answer(
            question=question,
            retrieved_chunks=selected_chunks,
        )
        if llm_result.status == "success":
            return AnswerResponse(
                answer=llm_result.answer,
                sources_used=self._build_chunk_sources(selected_chunks),
                support_level=llm_result.support_level,
                limitations=llm_result.limitations,
                route="retrieval",
                retrieved_chunks=selected_chunks,
                follow_up_questions=self._get_follow_up_questions(route="retrieval"),
                matched_customer_name=None,
                matched_field_name=None,
                synthesis_method=llm_result.synthesis_method,
                synthesis_status=llm_result.status,
                synthesis_status_message=None,
                routing_method=routing_decision.method,
                routing_confidence=routing_decision.confidence,
                routing_reason=routing_decision.reason,
                planning_method=None,
                planning_reason=None,
            )

        fallback_result = self._build_fallback_retrieval_result(retrieved_chunks)

        return AnswerResponse(
            answer=fallback_result.answer,
            sources_used=self._build_chunk_sources(retrieved_chunks),
            support_level=fallback_result.support_level,
            limitations=fallback_result.limitations,
            route="retrieval",
            retrieved_chunks=retrieved_chunks,
            follow_up_questions=self._get_follow_up_questions(route="retrieval"),
            matched_customer_name=None,
            matched_field_name=None,
            synthesis_method=fallback_result.synthesis_method,
            synthesis_status=llm_result.status,
            synthesis_status_message=llm_result.failure_reason,
            routing_method=routing_decision.method,
            routing_confidence=routing_decision.confidence,
            routing_reason=routing_decision.reason,
            planning_method=None,
            planning_reason=None,
        )

    def _shorten_text(self, text: str, max_length: int = 180) -> str:
        shortened = " ".join(text.split())
        if len(shortened) <= max_length:
            return shortened
        return f"{shortened[: max_length - 3].rstrip()}..."

    def _build_fallback_retrieval_result(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RetrievalSynthesisResult:
        top_score = retrieved_chunks[0].score
        support_level = "high" if top_score >= 3 else "medium" if top_score == 2 else "low"

        excerpts = []
        for item in retrieved_chunks:
            heading = item.chunk.section_heading or "No heading"
            excerpt = self._shorten_text(item.chunk.text)
            excerpts.append(f"- {item.chunk.file_name} | {heading}: {excerpt}")

        answer = "\n".join(
            [
                "Relevant markdown guidance was found in the demo documents.",
                "This answer is using a fallback summary based on retrieved chunks:",
                *excerpts,
            ]
        )

        limitations = (
            "This answer is based only on retrieved markdown chunks. "
            "LLM synthesis was unavailable, so the app returned a deterministic fallback summary instead."
        )

        return RetrievalSynthesisResult(
            answer=answer,
            support_level=support_level,
            limitations=limitations,
            synthesis_method="fallback",
            status="success",
            failure_reason=None,
        )

    def _build_chunk_sources(self, retrieved_chunks: list[RetrievedChunk]) -> list[str]:
        return [
            f"{item.chunk.file_name} -- {item.chunk.section_heading or 'No heading'}"
            for item in retrieved_chunks
        ]

    def _select_retrieval_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        top_score = retrieved_chunks[0].score
        quality_floor = max(top_score * 0.55, 4.0)

        total_characters = 0
        selected_chunks: list[RetrievedChunk] = []
        for item in retrieved_chunks:
            if item.score < quality_floor:
                continue
            chunk_length = len(item.chunk.text)
            if (
                selected_chunks
                and total_characters + chunk_length > self.retrieval_context_max_characters
            ):
                break
            selected_chunks.append(item)
            total_characters += chunk_length
            if len(selected_chunks) >= 3:
                break

        if selected_chunks:
            return selected_chunks

        return retrieved_chunks[:2]

    def _get_follow_up_questions(
        self,
        route: str,
        matched_customer_name: str | None = None,
        matched_field_name: str | None = None,
    ) -> list[str]:
        if route == "retrieval":
            return [
                "What does the policy say about payment delays?",
                "What are the basic screening criteria for FlexLine Demo?",
                "What guidance is given about missing financial statements?",
            ]

        if route == "structured":
            if matched_customer_name:
                return [
                    f"What is the debt to EBITDA of {matched_customer_name}?",
                    f"Does {matched_customer_name} have tax arrears?",
                    f"What product is {matched_customer_name} interested in?",
                ]
            if matched_field_name == "latest_revenue_eur":
                return [
                    "Which customer has the lowest turnover?",
                    "Which customer has the highest EBITDA?",
                    "Which customers have repeated payment delays?",
                ]
            return [
                "Which customer has the highest turnover?",
                "Which customers have tax arrears?",
                "Which customers are interested in AssetGrow Demo?",
            ]

        if route == "combined":
            return [
                "What does the policy say about tax arrears?",
                "Which customers have tax arrears?",
                "What is Harbor Foods Demo Oy's equity ratio?",
            ]

        return []

    def _route_question(
        self,
        question: str,
        planning_result: SemanticPlanningResult,
    ) -> RoutingDecision:
        semantic_plan = planning_result.plan
        if (
            planning_result.status == "success"
            and is_confident_routing_decision(
                RoutingDecision(
                    route=semantic_plan.route,
                    confidence=semantic_plan.confidence,
                    reason=semantic_plan.reason,
                    method="llm",
                )
            )
        ):
            return RoutingDecision(
                route=semantic_plan.route,
                confidence=semantic_plan.confidence,
                reason=semantic_plan.reason,
                method="llm",
            )

        rules_decision = self.router.classify(question)
        if is_confident_routing_decision(rules_decision):
            fallback_reason = rules_decision.reason
            if planning_result.status != "success":
                fallback_reason = (
                    f"{rules_decision.reason} Semantic planning fallback reason: "
                    f"{planning_result.failure_reason}"
                )
            elif semantic_plan.route == "unknown":
                fallback_reason = f"{rules_decision.reason} Semantic planning returned unknown intent."
            elif semantic_plan.confidence == "low":
                fallback_reason = (
                    f"{rules_decision.reason} Semantic planning confidence was too low "
                    f"for direct routing."
                )

            return RoutingDecision(
                route=rules_decision.route,
                confidence=rules_decision.confidence,
                reason=fallback_reason,
                method="rules",
            )

        return RoutingDecision(
            route="unknown",
            confidence="low",
            reason=(
                "The question could not be routed confidently. "
                "Try asking either a document question or a structured customer-data question."
            ),
            method="safe_fallback",
        )

    def _build_combined_response(
        self,
        question: str,
        routing_decision: RoutingDecision,
        planning_result: SemanticPlanningResult,
        semantic_plan: SemanticQueryPlan | None,
        chunks: list,
        dataframe,
        dataset_file_name: str | None,
        dataset_info_file_name: str | None,
    ) -> AnswerResponse:
        retrieval_query = self._build_combined_retrieval_query(question, semantic_plan)
        retrieved_chunks = self.retriever.retrieve(question=retrieval_query, chunks=chunks)
        selected_chunks = self._select_retrieval_chunks(retrieved_chunks) if retrieved_chunks else []

        structured_evidence = self.structured_query_engine.build_combined_evidence(
            question=question,
            dataframe=dataframe,
            dataset_file_name=dataset_file_name,
            plan=semantic_plan,
        )

        document_evidence = [
            f"{item.chunk.file_name} | {item.chunk.section_heading or 'No heading'} | {item.chunk.text}"
            for item in selected_chunks
        ]
        combined_sources = self._merge_sources(
            self._build_chunk_sources(selected_chunks),
            structured_evidence.sources_used,
        )

        if not combined_sources and dataset_info_file_name:
            combined_sources.append(dataset_info_file_name)

        if not selected_chunks:
            structured_summary = structured_evidence.summary or "No structured evidence could be assembled."
            return AnswerResponse(
                answer=(
                    "I could not find clearly relevant document evidence for this combined question. "
                    f"Structured evidence available: {structured_summary}"
                ),
                sources_used=combined_sources,
                support_level="low",
                limitations=(
                    "Combined answers require both relevant markdown guidance and structured customer evidence. "
                    "The document side was insufficient for a grounded combined synthesis."
                ),
                route="combined",
                retrieved_chunks=[],
                follow_up_questions=self._get_follow_up_questions(route="combined"),
                matched_customer_name=semantic_plan.customer_name if semantic_plan else None,
                matched_field_name=semantic_plan.field_name if semantic_plan else None,
                synthesis_method="fallback",
                synthesis_status=None,
                synthesis_status_message="Combined synthesis skipped: no relevant document evidence found.",
                routing_method=routing_decision.method,
                routing_confidence=routing_decision.confidence,
                routing_reason=routing_decision.reason,
                planning_method=semantic_plan.method if semantic_plan else None,
                planning_reason=self._build_planning_reason(planning_result),
            )

        llm_result = self.retrieval_synthesizer.synthesize_combined_answer(
            question=question,
            evidence=structured_evidence,
            document_evidence=document_evidence,
        )
        if llm_result.status == "success":
            combined_support_level = self._cap_combined_support_level(
                proposed=llm_result.support_level,
                selected_chunks=selected_chunks,
                structured_evidence=structured_evidence,
            )
            return AnswerResponse(
                answer=llm_result.answer,
                sources_used=combined_sources,
                support_level=combined_support_level,
                limitations=llm_result.limitations,
                route="combined",
                retrieved_chunks=selected_chunks,
                follow_up_questions=self._get_follow_up_questions(route="combined"),
                matched_customer_name=semantic_plan.customer_name if semantic_plan else None,
                matched_field_name=semantic_plan.field_name if semantic_plan else None,
                synthesis_method=llm_result.synthesis_method,
                synthesis_status=llm_result.status,
                synthesis_status_message=None,
                routing_method=routing_decision.method,
                routing_confidence=routing_decision.confidence,
                routing_reason=routing_decision.reason,
                planning_method=semantic_plan.method if semantic_plan else None,
                planning_reason=self._build_planning_reason(planning_result),
            )

        fallback_summary = self._build_fallback_combined_answer(
            selected_chunks=selected_chunks,
            structured_evidence=structured_evidence,
        )
        return AnswerResponse(
            answer=fallback_summary.answer,
            sources_used=combined_sources,
            support_level=fallback_summary.support_level,
            limitations=fallback_summary.limitations,
            route="combined",
            retrieved_chunks=selected_chunks,
            follow_up_questions=self._get_follow_up_questions(route="combined"),
            matched_customer_name=semantic_plan.customer_name if semantic_plan else None,
            matched_field_name=semantic_plan.field_name if semantic_plan else None,
            synthesis_method=fallback_summary.synthesis_method,
            synthesis_status=llm_result.status,
            synthesis_status_message=llm_result.failure_reason,
            routing_method=routing_decision.method,
            routing_confidence=routing_decision.confidence,
            routing_reason=routing_decision.reason,
            planning_method=semantic_plan.method if semantic_plan else None,
            planning_reason=self._build_planning_reason(planning_result),
        )

    def _build_combined_retrieval_query(
        self,
        question: str,
        semantic_plan: SemanticQueryPlan | None,
    ) -> str:
        if semantic_plan is None:
            return question

        query_parts = [question]
        if semantic_plan.product_name:
            query_parts.append(semantic_plan.product_name)
        if semantic_plan.document_topic:
            query_parts.append(semantic_plan.document_topic)
        if semantic_plan.operation == "preliminary_assessment":
            query_parts.append("criteria policy screening")

        return " ".join(part for part in query_parts if part).strip()

    def _build_planning_reason(self, planning_result: SemanticPlanningResult) -> str | None:
        if planning_result.status == "success":
            return planning_result.plan.reason
        return planning_result.failure_reason

    def _merge_sources(self, *source_lists: list[str]) -> list[str]:
        merged: list[str] = []
        for source_list in source_lists:
            for source in source_list:
                if source not in merged:
                    merged.append(source)
        return merged

    def _build_fallback_combined_answer(
        self,
        selected_chunks: list[RetrievedChunk],
        structured_evidence: CombinedEvidence,
    ) -> RetrievalSynthesisResult:
        document_lines = [
            f"- {item.chunk.file_name} | {item.chunk.section_heading or 'No heading'}: "
            f"{self._shorten_text(item.chunk.text)}"
            for item in selected_chunks
        ]
        missing_line = (
            f"Missing or uncertain points: {'; '.join(structured_evidence.missing_information)}"
            if structured_evidence.missing_information
            else "Missing or uncertain points: none explicitly flagged from the current evidence pack."
        )
        answer = "\n".join(
            [
                "Combined evidence was assembled from documents and structured customer data.",
                "Document evidence:",
                *document_lines,
                "Structured evidence:",
                structured_evidence.summary,
                missing_line,
            ]
        )
        limitations = (
            "This fallback combined answer is assembled deterministically from retrieved markdown passages "
            "and structured customer facts. LLM synthesis was unavailable, so the app did not produce a "
            "single natural-language combined interpretation."
        )
        support_level = self._cap_combined_support_level(
            proposed="medium",
            selected_chunks=selected_chunks,
            structured_evidence=structured_evidence,
        )
        return RetrievalSynthesisResult(
            answer=answer,
            support_level=support_level,
            limitations=limitations,
            synthesis_method="fallback",
            status="success",
            failure_reason=None,
        )

    def _cap_combined_support_level(
        self,
        proposed: str,
        selected_chunks: list[RetrievedChunk],
        structured_evidence: CombinedEvidence,
    ) -> str:
        has_structured_row_evidence = any(
            "| row:" in source for source in structured_evidence.sources_used
        )
        if not selected_chunks or not has_structured_row_evidence:
            return "low"
        if structured_evidence.missing_information and proposed == "high":
            return "medium"
        return proposed

    def _build_unclear_routing_response(
        self,
        routing_decision: RoutingDecision,
    ) -> AnswerResponse:
        return AnswerResponse(
            answer=(
                "I could not route this question confidently to the document path or the structured data path."
            ),
            sources_used=[],
            support_level="low",
            limitations=(
                "Try asking either a document-focused question such as 'What does the policy say about tax arrears?' "
                "or a structured-data question such as 'What is Harbor Foods Demo Oy's equity ratio?'"
            ),
            route="unknown",
            retrieved_chunks=[],
            follow_up_questions=[
                "What does the policy say about tax arrears?",
                "Which customers have tax arrears?",
                "What is Harbor Foods Demo Oy's equity ratio?",
            ],
            matched_customer_name=None,
            matched_field_name=None,
            synthesis_method="deterministic",
            synthesis_status=None,
            synthesis_status_message=None,
            routing_method=routing_decision.method,
            routing_confidence=routing_decision.confidence,
            routing_reason=routing_decision.reason,
            planning_method=None,
            planning_reason=None,
        )
