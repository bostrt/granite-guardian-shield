import asyncio
from typing import Any

from llama_stack.apis.inference import (CompletionMessage, Message,
                                        ToolResponseMessage, UserMessage)
from llama_stack.apis.safety import (RunShieldResponse, Safety,
                                     SafetyViolation, ViolationLevel)
from llama_stack.apis.shields import Shield
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from granite_guardian_shield.config import Risk
from granite_guardian_shield.helpers import get_higher_violation_level
from granite_guardian_shield.inference import Inference
from granite_guardian_shield.risk_assessor import (RiskAssessor,
                                                   RiskAssessorFactory)

logger = get_logger(name=__name__, category="safety")


class GraniteGuardianShield(Safety, ShieldsProtocolPrivate):
    """
    Manages registration and running of Shields
    """
    def __init__(self, inference: Inference) -> None:
        self._shield_risk_map: dict[str, list[RiskAssessor]] = dict()
        self.inference = inference

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        """
        Called by Llama Stack per shield configuration. Validates the risk configuration and stores in
        internal shield/risk mapping.
        """
        if shield.params is None or "risks" not in shield.params or not shield.params.get("risks"):
            raise ValueError(f"No risks defined for {shield.shield_id}")
        else:
            risk_configs = shield.params.get("risks", [])
            assessor_factory = RiskAssessorFactory(self.inference)
            risks: list[RiskAssessor] = []
            for risk_config in risk_configs:
                risk = Risk.model_validate(risk_config)
                risks.append(assessor_factory.create_assessor(risk))
            self._shield_risk_map[shield.shield_id] = risks
        logger.info(f"Registered {shield.shield_id}")

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = {},
    ) -> RunShieldResponse:
        """
        Run a single Shield for the updated list of messages in a session. This may evaluate multiple risks and run
        multiple inferences depending on user's Shield configuration.
        """
        # Peek at the last message
        msg = messages[-1]

        if isinstance(msg, UserMessage):
            # Handle with input safety guardrails
            logger.debug(f"----------------------{msg}")
        elif isinstance(msg, CompletionMessage):
            # Handle with the output safety guardrails
            logger.debug(f"++++++++++++++++++++++ {msg}")
        elif isinstance(msg, ToolResponseMessage):
            logger.debug(f"===================== {msg}")
        else:
            logger.debug(f"run_shield::unknown message type::{msg.model_dump_json()}")
            return RunShieldResponse()

        tasks = [assessor.run(messages) for assessor in self._shield_risk_map[shield_id]]
        verdicts = await asyncio.gather(*tasks)
        violation_metadatas = []

        highest_violation_level = ViolationLevel.INFO
        for v in verdicts:
            if v.is_risky:
                violation_metadatas.append(v.model_dump())
                higher_violation_level = get_higher_violation_level(highest_violation_level, v.violation_level)
                highest_violation_level = higher_violation_level

        if violation_metadatas:
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=msg.content,
                    violation_level=highest_violation_level,
                    metadata={"metadata": violation_metadatas},
                )
            )

        return RunShieldResponse()
