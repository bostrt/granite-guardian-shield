import asyncio
import logging
from typing import Any

import httpx
from llama_stack.apis.inference import Message, UserMessage
from llama_stack.apis.safety import Safety, RunShieldResponse, SafetyViolation, ViolationLevel
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from granite_guardian_shield.config import GraniteGuardianShieldConfig, Risk
from granite_guardian_shield.helpers import parse_output
from granite_guardian_shield.models import RiskProbability

logger = logging.getLogger(__name__)


class GraniteGuardianShield(Safety, ShieldsProtocolPrivate):

    def __init__(self, config: GraniteGuardianShieldConfig) -> None:
        self.config = config
        _api_key = self.config.api_key
        self._openai_client: AsyncOpenAI = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=_api_key.get_secret_value() if _api_key else None,
            http_client=httpx.AsyncClient(verify=self.config.verify_ssl),
        )

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        pass

    async def _evaluate_risk(self, message: UserMessage, risk: Risk) -> RiskProbability:
        guardian_config = {"risk_name": risk.name}
        if risk.definition:
            guardian_config["risk_definition"] = risk.definition

        guardian_resp: ChatCompletion = await self._openai_client.chat.completions.create(
            model=self.config.model,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            messages=[message],
            extra_body={"chat_template_kwargs": {"guardian_config": guardian_config}},
        )

        return parse_output(guardian_resp)

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = {},
    ) -> RunShieldResponse:
        message: UserMessage = messages[-1]
        logger.debug(f"run_shield::{message.content}")

        tasks = [self._evaluate_risk(message, risk) for risk in self.config.risks]
        verdicts = await asyncio.gather(*tasks)
        metadatas = [v.model_dump() for v in verdicts]

        if any(v.is_risky for v in verdicts):
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=message.content,
                    violation_level=ViolationLevel.ERROR,
                    metadata={"metadata": metadatas},
                )
            )

        return RunShieldResponse()
