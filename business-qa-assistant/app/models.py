from dataclasses import dataclass
from typing import Literal


Route = Literal["retrieval", "structured", "combined"]
SupportLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class AnswerResponse:
    answer: str
    sources_used: list[str]
    support_level: SupportLevel
    limitations: str
    route: Route
