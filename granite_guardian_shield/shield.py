import asyncio
import logging
from typing import Any, Protocol

import httpx
from llama_stack.apis.inference import Message, UserMessage
from llama_stack.apis.safety import (RunShieldResponse, Safety,
                                     SafetyViolation, ViolationLevel)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from granite_guardian_shield.config import GraniteGuardianShieldConfig, Risk
from granite_guardian_shield.helpers import parse_output
from granite_guardian_shield.models import RiskProbability

logger = logging.getLogger(__name__)


class Evaluator(Protocol):
    async def evaluate(self, message: Message) -> RiskProbability:
        ...


# TODO class ContextRiskEvaluator(Evaluator)


class SimpleRiskEvaluator(Evaluator):
    def __init__(self, risk: Risk, openai_client: AsyncOpenAI, model: str):
        self.risk = risk
        self.openai_client = openai_client
        self.model = model

    async def evaluate(self, message: Message) -> RiskProbability:
        guardian_config = {"risk_name": self.risk.name}
        if self.risk.definition:
            guardian_config["risk_definition"] = self.risk.definition

        response: ChatCompletion = await self.openai_client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            messages=[message],
            extra_body={"chat_template_kwargs": {"guardian_config": guardian_config}},
        )

        return parse_output(response, self.risk)


class GraniteGuardianShield(Safety, ShieldsProtocolPrivate):

    def __init__(self, config: GraniteGuardianShieldConfig) -> None:
        self.config = config
        _api_key = self.config.api_key
        self._openai_client: AsyncOpenAI = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=_api_key.get_secret_value() if _api_key else None,
            http_client=httpx.AsyncClient(verify=self.config.verify_ssl),
        )

        # Create one RiskEvaluator per risk
        self._evaluators = [
            SimpleRiskEvaluator(risk, self._openai_client, self.config.model)
            for risk in self.config.risks
        ]

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = {},
    ) -> RunShieldResponse:
        message: UserMessage = messages[-1]

        tasks = [evaluator.evaluate(message) for evaluator in self._evaluators]
        verdicts = await asyncio.gather(*tasks)
        violation_metadatas = []

        for v in verdicts:
            if v.is_risky:
                violation_metadatas.append(v.model_dump_json())

        if violation_metadatas:
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=message.content,
                    violation_level=ViolationLevel.ERROR,
                    metadata={"metadata": violation_metadatas},
                )
            )

        return RunShieldResponse()
