from __future__ import annotations

from interview_agent.config import BLOCK_MS, SAMPLE_RATE
from interview_agent.questions import GREETING, QUESTIONS, SIGNOFF


def test_questions_structure():
    assert isinstance(QUESTIONS, list)
    assert len(QUESTIONS) > 0
    for q in QUESTIONS:
        assert "id" in q
        assert "text" in q
        assert isinstance(q["id"], int)
        assert isinstance(q["text"], str)


def test_greetings_and_signoffs():
    assert isinstance(GREETING, str) and len(GREETING) > 0
    assert isinstance(SIGNOFF, str) and len(SIGNOFF) > 0


def test_config_defaults():
    assert SAMPLE_RATE > 0
    assert BLOCK_MS > 0
