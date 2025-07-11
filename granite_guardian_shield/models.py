from pydantic import BaseModel, Field, field_validator


class RiskProbability(BaseModel):
    """
    Model representing the risk output from Granite Guardian.
    """

    risk_name: str = Field(
        description="The risk name associated with this risk probability"
    )
    risk_definition: str | None = Field(
        description="Custom risk definition", default=None
    )
    is_risky: bool = Field(
        description="Granite Guardian default determination if input is risky"
    )
    safe_confidence: float | None = Field(
        ge=0.0, le=1.0, description="Probability that prompt is safe, if available"
    )
    risky_confidence: float | None = Field(
        ge=0.0, le=1.0, description="Probability that prompt is risky, if available"
    )

    @field_validator("safe_confidence")
    @classmethod
    def round_safe_confidence(cls, v: float | None) -> float | None:
        if not v:
            return v
        return round(v, 4)

    @field_validator("risky_confidence")
    @classmethod
    def round_risky_confidnce(cls, v: float | None) -> float | None:
        if not v:
            return v
        return round(v, 4)
