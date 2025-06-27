from pydantic import BaseModel, Field, field_validator


class RiskProbability(BaseModel):
    """
    Model representing the risk output from Granite Guardian.
    """

    risk_name: str = Field(description="The risk name associated with this risk probability")
    risk_definition: str | None = Field(description="Custom risk definition", default=None)
    is_risky: bool = Field(description="Granite Guardian hard yes/no answer")
    safe_confidence: float = Field(
        ge=0.0, le=1.0, description="Probability that prompt is safe"
    )
    risky_confidence: float = Field(
        ge=0.0, le=1.0, description="Probability that prompt is risky"
    )

    @field_validator('safe_confidence')
    @classmethod
    def round_safe_confidence(cls, v: float) -> float:
        return round(v, 4)

    @field_validator('risky_confidence')
    @classmethod
    def round_risky_confidnce(cls, v: float) -> float:
        return round(v, 4)
