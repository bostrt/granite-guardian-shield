import logging
from typing import Any

import httpx
from llama_stack.apis.inference import Message, UserMessage
from llama_stack.apis.safety import Safety, RunShieldResponse, SafetyViolation, ViolationLevel
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from granite_guardian_shield.config import GraniteGuardianShieldConfig
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

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = {},
    ) -> RunShieldResponse:
        # shield = await self.shield_store.get_shield(shield_id)
        # if not shield:
        #     raise ValueError(f"Unknown shield {shield_id}")

        # TODO How to ensure this is UserMessage and not Tool/System/etc?
        message: UserMessage = messages[-1]
        logger.debug(f"run_shield::{message.content}")

        metadatas = []
        verdicts: list[RiskProbability] = []
        for risk in self.config.risks:
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

            verdict = parse_output(guardian_resp)
            verdicts.append(verdict)

            metadatas.append(verdict.model_dump())

        if any([v.is_risky for v in verdicts]):
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=message.content,
                    violation_level=ViolationLevel.ERROR,
                    metadata={"metadata": [metadatas]},
                )
            )
        else:
            # TODO Maybe still include results in response without error violationlevel
            return RunShieldResponse()
