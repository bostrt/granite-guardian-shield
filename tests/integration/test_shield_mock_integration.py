import pytest
from unittest.mock import AsyncMock, patch

from granite_guardian_shield.config import GraniteGuardianShieldConfig, Risk
from granite_guardian_shield.shield import GraniteGuardianShield
from llama_stack.apis.inference import UserMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob


@pytest.mark.asyncio
@patch("granite_guardian_shield.shield.AsyncOpenAI")
async def test_run_shield_integration_with_mocked_openai(mock_openai_cls):
    # Config with 2 risks
    config = GraniteGuardianShieldConfig(
        base_url="https://example.com",
        api_key=None,
        model="test-model",
        verify_ssl=False,
        risks=[
            Risk(name="toxicity", definition="Toxic language"),
            Risk(name="jailbreak"),
            Risk(name="violence")
        ]
    )

    # Build mock logprobs for "Yes"
    mock_logprobs = ChoiceLogprobs(
        content=[
            ChatCompletionTokenLogprob(
                token="Yes",
                logprob=0.0,
                top_logprobs=[
                    TopLogprob(token="Yes", logprob=0.0),
                    TopLogprob(token="No", logprob=-2.3),
                ]
            )
        ]
    )

    # Build mock OpenAI response
    mock_response = ChatCompletion(
        id="cmpl-123",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(content="Yes", role="assistant"),
                logprobs=mock_logprobs
            )
        ]
    )

    # Patch client
    mock_openai_instance = AsyncMock()
    mock_openai_instance.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_openai_instance

    # Shield under test
    shield = GraniteGuardianShield(config)
    user_msg = UserMessage(content="you are bad", role="user")

    result = await shield.run_shield("test-shield", [user_msg])

    # Validate result
    assert result.violation is not None
    assert result.violation.violation_level.value == "error"
    assert result.violation.user_message == "you are bad"
    assert isinstance(result.violation.metadata["metadata"], list)
    assert len(result.violation.metadata["metadata"]) == 3
