import pytest
from llama_stack.apis.inference import Message, UserMessage
from llama_stack.apis.safety import Shield, ViolationLevel

from granite_guardian_shield.config import Risk
from granite_guardian_shield.inference import Inference
from granite_guardian_shield.models import RiskProbability
from granite_guardian_shield.shield import GraniteGuardianShield

fake_shield = Shield(
    identifier="example",
    provider_id="example",
    provider_resource_id="example",
    params={
        "risks": [
            {
                "name": "harm",
            }
        ]
    },
)


class FakeInference(Inference):
    def __init__(self, trigger_violation: bool):
        self.trigger_violation = trigger_violation

    async def run(self, risk: Risk, messages: list[Message]) -> RiskProbability:
        if self.trigger_violation:
            return RiskProbability(
                risk_name="harm",
                is_risky=True,
                risky_confidence=0.95,
                safe_confidence=0.01,
            )
        else:
            return RiskProbability(
                risk_name="harm",
                is_risky=False,
                risky_confidence=0.01,
                safe_confidence=0.95,
            )


@pytest.mark.asyncio
async def test_register_shield_missing_risks():
    missing_risks = Shield(
        identifier="example",
        provider_id="example",
        provider_resource_id="exapmle",
        params={"risks": []},
    )

    gg_shield = GraniteGuardianShield(FakeInference(trigger_violation=False))
    with pytest.raises(Exception):
        await gg_shield.register_shield(missing_risks)


@pytest.mark.asyncio
async def test_register_shield_missing_risk_definition():
    missing_risk_def = Shield(
        identifier="example",
        provider_id="example",
        provider_resource_id="exapmle",
        params={"risks": [
            {
                "name": "custom_risk"
                # No risk definition!
            }
        ]},
    )

    gg_shield = GraniteGuardianShield(FakeInference(trigger_violation=False))
    with pytest.raises(Exception):
        await gg_shield.register_shield(missing_risk_def)


@pytest.mark.asyncio
async def test_run_shield_with_violation():
    gg_shield = GraniteGuardianShield(FakeInference(trigger_violation=True))
    await gg_shield.register_shield(fake_shield)
    user_msg = UserMessage(content="You suck", role="user")

    result = await gg_shield.run_shield(fake_shield.identifier, [user_msg])

    assert result.violation is not None
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert "metadata" in result.violation.metadata


@pytest.mark.asyncio
async def test_run_shield_no_violation():
    gg_shield = GraniteGuardianShield(FakeInference(trigger_violation=False))
    await gg_shield.register_shield(fake_shield)
    user_msg = UserMessage(content="Have a nice day!", role="user")

    result = await gg_shield.run_shield(fake_shield.identifier, [user_msg])

    assert result.violation is None
