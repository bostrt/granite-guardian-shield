from unittest.mock import AsyncMock, patch

import pytest
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.safety import ViolationLevel

from granite_guardian_shield.config import GraniteGuardianShieldConfig, Risk
from granite_guardian_shield.models import RiskProbability
from granite_guardian_shield.shield import GraniteGuardianShield


@pytest.mark.asyncio
@patch("granite_guardian_shield.shield.parse_output")
@patch("granite_guardian_shield.shield.AsyncOpenAI")
async def test_run_shield_with_violation(mock_openai_cls, mock_parse_output):
    # Proper Risk object
    mock_config = GraniteGuardianShieldConfig(
        api_key="fake",
        base_url="https://example.com",
        model="test",
        verify_ssl=False,
        risks=[
            Risk(name="toxicity", definition="Prompt contains toxic language.")
        ]
    )

    # Return a risky verdict
    mock_verdict = RiskProbability(
        is_risky=True,
        safe_confidence=0.03,
        risky_confidence=0.97,
        risk_name="toxicity",
    )
    mock_parse_output.return_value = mock_verdict

    # Mock OpenAI client
    mock_openai_instance = AsyncMock()
    mock_openai_cls.return_value = mock_openai_instance
    mock_openai_instance.chat.completions.create.return_value = AsyncMock()

    shield = GraniteGuardianShield(mock_config)
    user_msg = UserMessage(content="You suck", role="user")

    result = await shield.run_shield("shield-id", [user_msg])

    assert result.violation is not None
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert "metadata" in result.violation.metadata


@pytest.mark.asyncio
@patch("granite_guardian_shield.shield.parse_output")
@patch("granite_guardian_shield.shield.AsyncOpenAI")
async def test_run_shield_no_violation(mock_openai_cls, mock_parse_output):
    mock_config = GraniteGuardianShieldConfig(
        api_key="fake",
        base_url="https://example.com",
        model="test",
        verify_ssl=False,
        risks=[
            Risk(name="toxicity", definition="Prompt contains toxic language.")
        ]
    )

    mock_verdict = RiskProbability(
        is_risky=False,
        safe_confidence=0.95,
        risky_confidence=0.05,
        risk_name="toxicity",
    )
    mock_parse_output.return_value = mock_verdict

    mock_openai_instance = AsyncMock()
    mock_openai_cls.return_value = mock_openai_instance
    mock_openai_instance.chat.completions.create.return_value = AsyncMock()

    shield = GraniteGuardianShield(mock_config)
    user_msg = UserMessage(content="Have a nice day!", role="user")

    result = await shield.run_shield("shield-id", [user_msg])

    assert result.violation is None
