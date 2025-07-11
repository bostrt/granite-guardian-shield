from enum import StrEnum


RISK_NAME = "risk_name"
RISK_DEFINITION = "risk_definition"


class SimpleRisk(StrEnum):
    """
    Enumeration of simple Granite Guardian risks that check user inputs or assistant response.
    """
    harm = "harm"
    social_bias = "social_bias"
    profanity = "profanity"
    sexual_cntent = "sexual_content"
    unethical_behavior = "unethical_behavior"
    violence = "violence"
    harm_engagement = "harm_engagement"
    evasivness = "evasiveness"


class HallucinationRisk(StrEnum):
    """
    Enumeration of Granite Guardian risks that check assistant response in relation to some other input.
    """
    context_relevance = "context_relevance"
    groundedness = "groundedness"
    answer_relevance = "answer_relevance"
    function_call = "function_call"
