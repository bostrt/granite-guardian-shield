from pydantic import BaseModel, Field, SecretStr


class GraniteGuardianShieldConfig(BaseModel):
    base_url: str = Field(
        description="OpenAI Compatible API serving Granite Guardian endpoint"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API Key"
    )
    verify_ssl: bool = True
    model: str = "granite-guardian-3-1-8b"

    # TODO risky_threshold: float
