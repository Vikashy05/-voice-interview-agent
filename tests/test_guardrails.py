from __future__ import annotations

import pytest

from interview_agent import guardrails


@pytest.mark.parametrize(
    ("answer", "valid", "reason"),
    [
        ("", False, "silence"),
        ("hello", False, "bare_greeting"),
        ("model model model model model model", False, "repeated_tokens"),
        ("I would compare validation and training error before tuning regularization.", True, "meaningful_response"),
    ],
)
def test_meaningful_answer_validation(answer: str, valid: bool, reason: str) -> None:
    result = guardrails.is_meaningful_answer(answer)
    assert result["is_valid"] is valid
    assert result["reason"] == reason


def test_find_leak_detects_answer_reveal() -> None:
    assert guardrails.find_leak("The correct answer is to validate the model.")
    assert guardrails.find_leak("Let's discuss the question instead.") is None
