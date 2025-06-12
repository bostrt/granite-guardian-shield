import math
from typing import Tuple

from openai.types.chat.chat_completion import ChatCompletion, ChoiceLogprobs

from granite_guardian_shield.models import RiskProbability


def _softmax2(log_a: float, log_b: float) -> Tuple[float, float]:
    """
    Convert two log-probabilities into plain probabilities that add to 1.
    log_a , log_b = Natural-log values that represent the models raw confidence
    in two mutually-exclusive outcomes a and b.

    Since we only use torch for the two value softmax, we can remove the large torch dependency.

    Recreating:
    torch.softmax(
        torch.tensor([math.log(safe_prob), math.log(risky_prob)]), dim=0
    )

    Granite Guardian gives us log-probs for each answer token.
    We sum those logs separately for “safe” (a / No) and “risky” (b / Yes) and then call
    this function to turn the pair into clean, comparable percentages.
    e.g. P(a) + P(b) = 1.0

    """
    m = max(log_a, log_b)
    exp_a = math.exp(log_a - m)
    exp_b = math.exp(log_b - m)
    denom = exp_a + exp_b
    return exp_a / denom, exp_b / denom


def get_probabilities(
    logprobs: ChoiceLogprobs,
    safe_token: str = "No",
    risky_token: str = "Yes",
) -> Tuple[float, float]:
    """
    Rebuild (P_safe, P_risky) from token-level logprobs that we directly evaluate for confidence levels.
    P_safe = Models confidence the prompt is safe (e.g. .7834567)
    P_risky = Models confidence the prompt breaks the policy (e.g. 0.2165433)

    We look at the models confidence for each word, normalise them so they add
    up to 1, and return the pair.
    Sometimes the word is broken into more than one token internally
    (e.g., “Yes” might be ["Y", "es"]). We add up the probabilities of every token
    piece that belongs to “Yes” and every piece that belongs to “No”.

    """
    if not logprobs or not logprobs.content:
        raise ValueError("Granite-Guardian response contained no logprobs.")

    safe_prob = 1e-50  # Essentially zero, safety measure to prevent math.log(0)
    risky_prob = 1e-50

    safe_token = safe_token.lower()
    risky_token = risky_token.lower()

    for step in logprobs.content:  # Loops over every token piece the model generated when it wrote its one-word answer.
        for token_prob in step.top_logprobs:
            token = token_prob.token.strip().lower()  # normalize to lowercase
            if token == safe_token:
                safe_prob += math.exp(token_prob.logprob)  # Adds that piece’s probability to safe_prob when it’s part of "No".
            elif token == risky_token:
                risky_prob += math.exp(token_prob.logprob)  # Same for "Yes"

    return _softmax2(math.log(safe_prob), math.log(risky_prob))


def parse_output(response: ChatCompletion) -> RiskProbability:
    """
    Parse Granite-Guardians response into a structured RiskProbability response.

    Args:
        response
            The ChatCompletion object returned by OpenAI client.

    Raises:
        ValueError: Raised if Granite Guardian response doesn't contain `logprobs`.

    Returns:
        RiskProbability
            is_risky - True if Guardians text answer is “Yes”.
            safe_confidence - safe probability score from get_probabilities.
            risky_confidence - risky probability score from get_probabilities.
    """
    label = response.choices[0].message.content.strip().lower()
    p_safe, p_risky = get_probabilities(response.choices[0].logprobs)

    return RiskProbability(
        is_risky=(label == "yes"),
        safe_confidence=p_safe,
        risky_confidence=p_risky,
    )
