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
                matched_customer_name=structured_result.matched_customer_name,
                matched_field_name=structured_result.matched_field_name,
                synthesis_method="deterministic",
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
            matched_customer_name=None,
            matched_field_name=None,
            synthesis_method="deterministic",
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
                matched_customer_name=None,
                matched_field_name=None,
                synthesis_method="fallback",
            )

        selected_chunks = self._trim_retrieved_chunks(retrieved_chunks)
        llm_result = self.retrieval_synthesizer.synthesize_retrieval_answer(
            question=question,
            retrieved_chunks=selected_chunks,
        )
        if llm_result is not None:
            return AnswerResponse(
                answer=llm_result.answer,
                sources_used=self._build_chunk_sources(selected_chunks),
                support_level=llm_result.support_level,
                limitations=llm_result.limitations,
                route="retrieval",
                retrieved_chunks=selected_chunks,
                matched_customer_name=None,
                matched_field_name=None,
                synthesis_method=llm_result.synthesis_method,
            )

        fallback_result = self._build_fallback_retrieval_result(retrieved_chunks)

        return AnswerResponse(
            answer=fallback_result.answer,
            sources_used=self._build_chunk_sources(retrieved_chunks),
            support_level=fallback_result.support_level,
            limitations=fallback_result.limitations,
            route="retrieval",
            retrieved_chunks=retrieved_chunks,
            matched_customer_name=None,
            matched_field_name=None,
            synthesis_method=fallback_result.synthesis_method,
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
        )

    def _build_chunk_sources(self, retrieved_chunks: list[RetrievedChunk]) -> list[str]:
        return [
            f"{item.chunk.file_name} | {item.chunk.section_heading or 'No heading'} | {item.chunk.chunk_id}"
            for item in retrieved_chunks
        ]

    def _trim_retrieved_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        total_characters = 0
        selected_chunks: list[RetrievedChunk] = []
        for item in retrieved_chunks:
            chunk_length = len(item.chunk.text)
            if (
                selected_chunks
                and total_characters + chunk_length > self.retrieval_context_max_characters
            ):
                break
            selected_chunks.append(item)
            total_characters += chunk_length
        return selected_chunks
