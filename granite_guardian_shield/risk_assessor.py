from abc import ABC, abstractmethod

from llama_stack.log import get_logger
from llama_stack.apis.inference import CompletionMessage, Message

from granite_guardian_shield.config import Risk
from granite_guardian_shield.inference import Inference
from granite_guardian_shield.models import RiskProbability
from granite_guardian_shield.constants import HallucinationRisk, SimpleRisk

logger = get_logger(name=__name__, category="safety")


class RiskAssessor(ABC):
    @abstractmethod
    async def run(self, messages: list[Message]) -> RiskProbability:
        """
        Evaluate the risk of a list of messages and return the risk probability.
        """
        ...


class RiskAssessorFactory:
    def __init__(self, ggi: Inference):
        self.ggi = ggi

    def create_assessor(self, risk: Risk) -> RiskAssessor:
        if risk.name in SimpleRisk.__members__:
            return SimpleRiskAssessor(risk, self.ggi)
        elif risk.name == HallucinationRisk.answer_relevance:
            return AnswerContextRelevanceRiskAssessor(risk, self.ggi)
        elif risk.name == HallucinationRisk.context_relevance:
            raise Exception("context_relevance assessor not yet implemented")
        elif risk.name == HallucinationRisk.function_call:
            raise Exception("function_call assessor not yet implemented")
        elif risk.name == HallucinationRisk.groundedness:
            raise Exception("groundedness assessor not yet implemented")
        else:
            logger.info(f"Custom risk definition detected: {risk.name}")
            if not risk.definition or not risk.definition.strip():
                raise ValueError(f"Missing risk definition for custom risk {risk.name}")
            return SimpleRiskAssessor(risk, self.ggi)


class AnswerContextRelevanceRiskAssessor(RiskAssessor):
    def __init__(self, risk: Risk, ggi: Inference):
        self.risk = risk
        self.ggi = ggi

    async def run(self, messages: list[Message]) -> RiskProbability:
        # Peek at last message
        msg = messages[-1]
        if isinstance(msg, CompletionMessage):
            pass
        else:
            raise RuntimeError("Improper message stream for answer context relevance evaulator")


class SimpleRiskAssessor(RiskAssessor):
    def __init__(self, risk: Risk, ggi: Inference):
        self.risk = risk
        self.ggi = ggi

    async def run(self, messages: list[Message]) -> RiskProbability:
        # Peek at the last message
        msg = messages[-1]

        return await self.ggi.run(self.risk, [msg])
