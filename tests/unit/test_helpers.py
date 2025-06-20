import math
import pytest

from granite_guardian_shield.config import Risk
from granite_guardian_shield.helpers import _softmax2, get_probabilities, parse_output
from granite_guardian_shield.models import RiskProbability

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob


def test_softmax2_outputs_add_to_one():
    a, b = _softmax2(math.log(0.8), math.log(0.2))
    assert round(a + b, 5) == 1.0
    assert a > b


def test_softmax2_equal_probs():
    a, b = _softmax2(0.0, 0.0)
    assert abs(a - 0.5) < 1e-6
    assert abs(b - 0.5) < 1e-6


def test_get_probabilities_basic_case():
    logprobs = ChoiceLogprobs(
        content=[
            ChatCompletionTokenLogprob(
                token="Yes",
                logprob=math.log(0.9),
                top_logprobs=[
                    TopLogprob(token="Yes", logprob=math.log(0.9)),
                    TopLogprob(token="No", logprob=math.log(0.1)),
                ],
            )
        ]
    )

    p_safe, p_risky = get_probabilities(logprobs)
    assert 0 <= p_safe <= 1
    assert 0 <= p_risky <= 1
    assert round(p_safe + p_risky, 4) == 1.0
    assert p_risky > p_safe


def test_get_probabilities_missing_logprobs():
    with pytest.raises(ValueError):
        get_probabilities(None)

    with pytest.raises(ValueError):
        get_probabilities(ChoiceLogprobs(content=[]))


def test_parse_output_yes_label():
    response = ChatCompletion(
        id="cmpl-abc",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content="Yes",
                    role="assistant"
                ),
                logprobs=ChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token="Yes",
                            logprob=math.log(0.9),
                            top_logprobs=[
                                TopLogprob(token="Yes", logprob=math.log(0.9)),
                                TopLogprob(token="No", logprob=math.log(0.1)),
                            ]
                        )
                    ]
                )
            )
        ]
    )

    result = parse_output(response, Risk(name="test"))
    assert isinstance(result, RiskProbability)
    assert result.is_risky is True
    assert round(result.safe_confidence + result.risky_confidence, 4) == 1.0
    assert result.risky_confidence > result.safe_confidence
