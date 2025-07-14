from typing import Generator
from llama_stack.log import get_logger
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from llama_stack.apis.inference import Message
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)


from granite_guardian_shield.constants import RISK_NAME, RISK_DEFINITION
from granite_guardian_shield.config import Risk
from granite_guardian_shield.helpers import parse_output
from granite_guardian_shield.models import RiskProbability
from abc import ABC, abstractmethod

logger = get_logger(name=__name__, category="safety")

# NOTE
# NOTE This package should be deleted and replaced with Llama Stack inference when this issue is resolved
# NOTE https://github.com/meta-llama/llama-stack/issues/2720
# NOTE


class Inference(ABC):
    @abstractmethod
    async def run(self, risk: Risk, messages: list[Message]) -> RiskProbability:
        pass


class GraniteGuardianVLLMInference(Inference):
    def __init__(self, openai_client: AsyncOpenAI, model: str):
        self.openai_client = openai_client
        self.model = model

    def _convert_messages(
        self, messages: list[Message]
    ) -> Generator[ChatCompletionMessageParam, None, None]:
        for message in messages:
            if message.role == "user":
                yield ChatCompletionUserMessageParam(
                    content=str(message.content), role=message.role
                )
            elif message.role == "assistant":
                yield ChatCompletionAssistantMessageParam(
                    content=str(message.content), role=message.role
                )
            elif message.role == "tool":
                yield ChatCompletionToolMessageParam(
                    content=str(message.content),
                    role=message.role,
                    tool_call_id="ABCDEF1234",
                )
            else:
                logger.warning(f"Unknown role {message.role}")

    async def run(self, risk: Risk, messages: list[Message]) -> RiskProbability:
        guardian_config = {RISK_NAME: risk.name}
        if risk.definition:
            guardian_config[RISK_DEFINITION] = risk.definition

        openai_messages = self._convert_messages(messages)
        response: ChatCompletion = await self.openai_client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            # TODO This seems to be broke for the output checks like relevance. Do I need to summarize user inputs before checking relevance? Maybe just don't care right now?
            # TODO Make this handle System, User, or Context type messages
            messages=openai_messages,
            extra_body={"chat_template_kwargs": {"guardian_config": guardian_config}},
        )

        logger.debug(response)
        return parse_output(response, risk)
