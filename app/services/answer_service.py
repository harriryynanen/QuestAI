from llm.openai_client import OpenAIRetrievalSynthesizer
from models import AnswerResponse, RetrievalSynthesisResult, RetrievedChunk
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.router import SimpleRouter
from structured.customer_data import CustomerDataLoader
from structured.query_engine import StructuredQueryEngine


class AnswerService:
    def __init__(
        self,
        router: SimpleRouter,
        document_store: DocumentStore,
        customer_data_loader: CustomerDataLoader,
        chunker: MarkdownChunker,
        retriever: KeywordRetriever,
        structured_query_engine: StructuredQueryEngine,
        retrieval_synthesizer: OpenAIRetrievalSynthesizer,
        retrieval_context_max_characters: int,
    ) -> None:
        self.router = router
        self.document_store = document_store
        self.customer_data_loader = customer_data_loader
        self.chunker = chunker
        self.retriever = retriever
        self.structured_query_engine = structured_query_engine
        self.retrieval_synthesizer = retrieval_synthesizer
        self.retrieval_context_max_characters = retrieval_context_max_characters

    def answer_question(self, question: str) -> AnswerResponse:
        route = self.router.classify(question)
        dataset_info = self.customer_data_loader.get_data_info()
        dataframe = self.customer_data_loader.get_dataframe()
        dataset_file_name = self.customer_data_loader.get_dataset_file_name()
        markdown_documents = self.document_store.load_markdown_documents()
        chunks = self.chunker.chunk_documents(markdown_documents)

        if route == "retrieval":
            retrieved_chunks = self.retriever.retrieve(question=question, chunks=chunks)
            return self._build_retrieval_response(question, retrieved_chunks)

        if route == "structured":
            structured_result = self.structured_query_engine.answer(
                question=question,
                dataframe=dataframe,
                dataset_file_name=dataset_file_name,
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
            )

        sources_used = []
        if dataset_info.dataset_found and dataset_info.file_name is not None:
            sources_used.append(dataset_info.file_name)
        if markdown_documents:
            sources_used.extend(document.file_name for document in markdown_documents)
        if not sources_used:
            sources_used.append(
                "No detected source files yet. Add markdown documents or a CSV dataset to the data folders."
            )

        return AnswerResponse(
            answer=(
                "This question appears to need both document guidance and structured data. "
                "Combined reasoning is not implemented yet in this demo."
            ),
            sources_used=sources_used,
            support_level="low",
            limitations=(
                "Markdown retrieval and deterministic CSV querying are available separately, "
                "but the system does not yet synthesize them into one grounded combined answer."
            ),
            route=route,
            retrieved_chunks=[],
            follow_up_questions=self._get_follow_up_questions(route="combined"),
            matched_customer_name=None,
            matched_field_name=None,
            synthesis_method="deterministic",
            synthesis_status=None,
            synthesis_status_message=None,
        )

    def _build_retrieval_response(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
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
