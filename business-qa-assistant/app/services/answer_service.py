from models import AnswerResponse
from services.router import SimpleRouter


class AnswerService:
    def __init__(self, router: SimpleRouter) -> None:
        self.router = router

    def answer_question(self, question: str) -> AnswerResponse:
        route = self.router.classify(question)

        return AnswerResponse(
            answer=(
                "This is a placeholder answer for the initial demo slice. "
                f"The question was routed to the '{route}' path."
            ),
            sources_used=["No live sources yet; retrieval and data integrations are not implemented."],
            support_level="low",
            limitations=(
                "This response is not based on document retrieval or structured data yet. "
                "It only demonstrates routing and response rendering."
            ),
            route=route,
        )
