from pydantic import BaseModel, ConfigDict, Field, SecretStr

from granite_guardian_shield.constants import SimpleRisk, RISK_NAME, RISK_DEFINITION


class Risk(BaseModel):
    # TODO Support groundedness, relevance, answer_relevance, and function_call
    # TODO Support response safety
    name: str = Field(
        default=SimpleRisk.harm,
        description="A Granite Guardian risk name. This may be a custom risk name or one of the predefined risks.",
        serialization_alias=RISK_NAME,
    )
    definition: str | None = Field(
        default=None,
        description="Optional definition for a custom Granite Guardian risk",
        examples=[
            "User message contains personal information or sensitive personal information that is included as a part of a prompt.",
        ],
        serialization_alias=RISK_DEFINITION,
    )

    model_config = ConfigDict(serialize_by_alias=True)


class GraniteGuardianShieldConfig(BaseModel):
    base_url: str = Field(
        description="OpenAI Compatible endpoint serving Granite Guardian"
    )
    api_key: SecretStr | None = Field(
        description="Optional API Key for the base_url",
        default=None,
    )
    verify_ssl: bool = True
    model: str = Field(
        description="The Granite Guardian model name",
        default="granite-guardian-3-2-8b"
    )
    risks: list[Risk] = Field(
        default=[Risk()],
        description="List of risks to run on each user input",
    )
    risky_threshold: float | None = Field(
        description="The percentage confidence threshold when to consider a user input risky. Defaults to None which uses Granite Guardian model's default behavior.",
        default=None,
    )
