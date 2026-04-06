from models import AnswerResponse
from retrieval.document_store import DocumentStore
from services.router import SimpleRouter
from structured.customer_data import CustomerDataLoader


class AnswerService:
    def __init__(
        self,
        router: SimpleRouter,
        document_store: DocumentStore,
        customer_data_loader: CustomerDataLoader,
    ) -> None:
        self.router = router
        self.document_store = document_store
        self.customer_data_loader = customer_data_loader

    def answer_question(self, question: str) -> AnswerResponse:
        route = self.router.classify(question)
        documents = self.document_store.list_documents()
        dataset_info = self.customer_data_loader.get_data_info()

        sources_used = [document.file_name for document in documents]
        if dataset_info.dataset_found and dataset_info.file_name is not None:
            sources_used.append(dataset_info.file_name)

        if not sources_used:
            sources_used = [
                "No detected source files yet. Add supported documents or a CSV dataset to the data folders."
            ]

        return AnswerResponse(
            answer=(
                "This is a placeholder answer for the initial demo slice. "
                f"The question was routed to the '{route}' path."
            ),
            sources_used=sources_used,
            support_level="low",
            limitations=(
                "This response is not based on document retrieval or structured data querying yet. "
                "It only demonstrates routing, file discovery, and response rendering."
            ),
            route=route,
        )
