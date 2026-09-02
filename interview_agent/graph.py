r"""The LangGraph state machine driving the interview.

    greet -> ask -> listen -> evaluate -> advance -> ask
                                              \-> close

Interruptions are handled inside whichever node is speaking: any node that
talks can bounce control to `interrupt`, which yields to the candidate and
routes back to where it left off.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

import sys

from langgraph.graph import END, StateGraph

from . import audio, brain
from . import config as C
from .questions import GREETING, QUESTIONS, SIGNOFF


def append(left: list, right: list) -> list:
    return (left or []) + (right or [])


class InterviewState(TypedDict, total=False):
    q_index: int                      # which fixed question we are on
    followups: int                    # follow-ups spent on the current question
    history: Annotated[list, append]  # full conversational transcript
    answer: str                       # latest candidate utterance
    pending_audio: Any                # speech captured during a barge-in
    resume_to: str                    # node to return to after an interruption
    cut_off: str                      # what the agent was saying when cut off
    interrupted: bool
    silent_turns: int                 # consecutive no-response turns
    done: bool


# `mic` is a module-level handle so graph nodes stay serialisable.
MIC: audio.Microphone | None = None


def _out(line: str) -> None:
    """Print without ever raising on a console that cannot encode a character."""
    try:
        print(line)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"))


def _say(state: InterviewState, text: str, resume_to: str) -> dict:
    """Speak, and fold any interruption into the returned state delta."""
    if not text:
        return {}
    _out(f"\n  {C.BOT_NAME}: {text}")
    interrupted, spoken, captured = audio.speak_interruptible(MIC, text)
    delta: dict = {"history": [{"role": "agent", "text": spoken}]}
    if interrupted:
        _out("  [interrupted by candidate]")
        delta.update(
            interrupted=True,
            cut_off=spoken,
            pending_audio=captured,
            resume_to=resume_to,
        )
    return delta


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def greet(state: InterviewState) -> dict:
    delta = _say(state, GREETING, resume_to="ask")
    return {"q_index": 0, "followups": 0, "silent_turns": 0, **delta}


def ask(state: InterviewState) -> dict:
    idx = state.get("q_index", 0)
    question = QUESTIONS[idx]["text"]

    # The very first question is asked verbatim; later ones get a transition
    # so the interview flows instead of reading like a form.
    if idx == 0:
        text = question
    else:
        text = brain.bridge(state.get("history", []), question)

    return {"followups": 0, **_say(state, text, resume_to="listen")}


def listen_node(state: InterviewState) -> dict:
    prefix = state.get("pending_audio")
    if prefix is not None:
        _out("  (listening - continuing from your interruption)")
    else:
        _out("  (listening...)")

    clip = audio.listen(MIC, prefix=prefix, timeout=20.0)
    if clip is None:
        return {
            "answer": "",
            "pending_audio": None,
            "silent_turns": state.get("silent_turns", 0) + 1,
        }

    text = brain.transcribe(clip)
    if not text:
        return {
            "answer": "",
            "pending_audio": None,
            "silent_turns": state.get("silent_turns", 0) + 1,
        }

    _out(f"  You: {text}")
    return {
        "answer": text,
        "pending_audio": None,
        "silent_turns": 0,
        "history": [{"role": "user", "text": text}],
    }


def interrupt(state: InterviewState) -> dict:
    """The candidate talked over us. Yield, listen, respond, then resume."""
    clip = audio.listen(MIC, prefix=state.get("pending_audio"), timeout=12.0)
    said = brain.transcribe(clip) if clip is not None else ""

    if not said:
        # False alarm (a cough, background noise) — carry on quietly.
        return {"interrupted": False, "pending_audio": None}

    _out(f"  You: {said}")
    reply = brain.handle_interruption(
        state.get("history", []), state.get("cut_off", ""), said
    )
    delta = {
        "history": [{"role": "user", "text": said}],
        "interrupted": False,
        "pending_audio": None,
        # An interruption is never treated as an answer to the current
        # question, however it is phrased: _is_question() only checks
        # grammar, so a genuine clarifying question ("does that mean high
        # variance?") used to silently skip the question being asked. Only
        # evaluate() on a real, transcribed answer is allowed to advance.
        "resume_to": state.get("resume_to", "listen"),
    }
    if reply:
        spoken = _say(state, reply, resume_to=state.get("resume_to", "listen"))
        # Merge, preserving both history entries.
        merged = delta["history"] + spoken.get("history", [])
        delta.update(spoken)
        delta["history"] = merged
    return delta


def evaluate(state: InterviewState) -> dict:
    """React to the answer, and decide whether it needs a probe."""
    answer = state.get("answer", "")
    if not answer:
        return {}
    ack = brain.acknowledge(state.get("history", []), answer)
    return _say(state, ack, resume_to="advance") if ack else {}


def nudge(state: InterviewState) -> dict:
    """Gently re-prompt after silence."""
    text = (
        "Sorry, I didn't quite catch that. Could you say that again?"
        if state.get("silent_turns", 0) < 2
        else "No worries, take your time. Whenever you're ready."
    )
    return _say(state, text, resume_to="listen")


def advance(state: InterviewState) -> dict:
    return {"q_index": state.get("q_index", 0) + 1, "followups": 0}


def close(state: InterviewState) -> dict:
    delta = _say(state, SIGNOFF, resume_to="close")
    return {"done": True, **delta}


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def _after_speaking(
    state: InterviewState, default: str
) -> str:
    return "interrupt" if state.get("interrupted") else default


def route_greet(state) -> Literal["interrupt", "ask"]:
    return _after_speaking(state, "ask")


def route_ask(state) -> Literal["interrupt", "listen"]:
    return _after_speaking(state, "listen")


def route_interrupt(state) -> Literal["listen", "ask", "advance", "close"]:
    """Return to whatever the agent was doing before being cut off.

    If the interruption was a question we have now answered, `resume_to` is
    `advance` so the interview moves on. If it was an aside, we hand the floor
    back with `listen` so they can finish their thought.
    """
    if state.get("interrupted"):        # interrupted again while replying
        return "listen"
    target = state.get("resume_to", "listen")
    return target if target in ("listen", "ask", "advance", "close") else "listen"


def route_listen(state) -> Literal["nudge", "evaluate"]:
    if not state.get("answer"):
        return "nudge"
    return "evaluate"


def route_nudge(state) -> Literal["interrupt", "listen", "close"]:
    if state.get("interrupted"):
        return "interrupt"
    # Three strikes and we wrap up rather than loop forever.
    if state.get("silent_turns", 0) >= 3:
        return "close"
    return "listen"


def route_evaluate(state) -> Literal["interrupt", "advance"]:
    if state.get("interrupted"):
        return "interrupt"
    return "advance"


def route_advance(state) -> Literal["ask", "close"]:
    return "ask" if state.get("q_index", 0) < len(QUESTIONS) else "close"


def route_close(state) -> Literal["interrupt", "__end__"]:
    return "interrupt" if state.get("interrupted") else END


# --------------------------------------------------------------------------
def build() -> Any:
    g = StateGraph(InterviewState)

    g.add_node("greet", greet)
    g.add_node("ask", ask)
    g.add_node("listen", listen_node)
    g.add_node("interrupt", interrupt)
    g.add_node("evaluate", evaluate)
    g.add_node("nudge", nudge)
    g.add_node("advance", advance)
    g.add_node("close", close)

    g.set_entry_point("greet")
    g.add_conditional_edges("greet", route_greet)
    g.add_conditional_edges("ask", route_ask)
    g.add_conditional_edges("listen", route_listen)
    g.add_conditional_edges("interrupt", route_interrupt)
    g.add_conditional_edges("evaluate", route_evaluate)
    g.add_conditional_edges("nudge", route_nudge)
    g.add_conditional_edges("advance", route_advance)
    g.add_conditional_edges("close", route_close)

    return g.compile()
