from models import AnswerResponse, RetrievedChunk
from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore
from retrieval.retriever import KeywordRetriever
from services.router import SimpleRouter
from structured.customer_data import CustomerDataLoader


class AnswerService:
    def __init__(
        self,
        router: SimpleRouter,
        document_store: DocumentStore,
        customer_data_loader: CustomerDataLoader,
        chunker: MarkdownChunker,
        retriever: KeywordRetriever,
    ) -> None:
        self.router = router
        self.document_store = document_store
        self.customer_data_loader = customer_data_loader
        self.chunker = chunker
        self.retriever = retriever

    def answer_question(self, question: str) -> AnswerResponse:
        route = self.router.classify(question)
        dataset_info = self.customer_data_loader.get_data_info()
        markdown_documents = self.document_store.load_markdown_documents()
        chunks = self.chunker.chunk_documents(markdown_documents)

        if route == "retrieval":
            retrieved_chunks = self.retriever.retrieve(question=question, chunks=chunks)
            return self._build_retrieval_response(retrieved_chunks)

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
                "This is a placeholder answer for the initial demo slice. "
                f"The question was routed to the '{route}' path. "
                "Structured and combined answer logic is not implemented yet."
            ),
            sources_used=sources_used,
            support_level="low",
            limitations=(
                "Only markdown retrieval is partially implemented in this stage. "
                "Structured and combined routes still use placeholder behavior."
            ),
            route=route,
            retrieved_chunks=[],
        )

    def _build_retrieval_response(
        self,
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
            )

        top_score = retrieved_chunks[0].score
        support_level = "high" if top_score >= 3 else "medium" if top_score == 2 else "low"

        excerpts = []
        sources_used = []
        for item in retrieved_chunks:
            heading = item.chunk.section_heading or "No heading"
            excerpt = self._shorten_text(item.chunk.text)
            excerpts.append(
                f"- {item.chunk.file_name} | {heading}: {excerpt}"
            )
            sources_used.append(f"{item.chunk.file_name} | {heading}")

        answer = "\n".join(
            [
                "Relevant markdown guidance was found in the demo documents.",
                "These excerpts are source-grounded and should be treated as preliminary guidance only:",
                *excerpts,
            ]
        )

        return AnswerResponse(
            answer=answer,
            sources_used=sources_used,
            support_level=support_level,
            limitations=(
                "This is a lightweight retrieval step based on keyword overlap, not a final decision or full reasoning system."
            ),
            route="retrieval",
            retrieved_chunks=retrieved_chunks,
        )

    def _shorten_text(self, text: str, max_length: int = 180) -> str:
        shortened = " ".join(text.split())
        if len(shortened) <= max_length:
            return shortened
        return f"{shortened[: max_length - 3].rstrip()}..."
