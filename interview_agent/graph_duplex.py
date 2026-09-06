r"""Interview graph running on the full-duplex layer.

Same shape as `graph.py`, but speaking and understanding overlap: the agent
keeps listening (and transcribing) while it talks, so it can tell a nod from a
real interruption and only yields the floor for the latter.
"""
from __future__ import annotations

import random
import sys
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from . import brain, duplex, guardrails
from . import recorder as rec
from . import session as sess
from .questions import GREETING, QUESTIONS, SIGNOFF


def append(left: list, right: list) -> list:
    return (left or []) + (right or [])


class InterviewState(TypedDict, total=False):
    q_index: int
    followups: int
    history: Annotated[list, append]
    answer: str
    resume_to: str
    cut_off: str
    interrupted: bool
    silent_turns: int
    wants_end: bool     # candidate asked to stop
    wants_skip: bool    # candidate asked to move to the next question
    gives_up: bool      # candidate said they can't answer this one
    concern: str        # guardrail verdict on the last answer
    done: bool


# Both are set by main before the graph runs.
SESSION: duplex.DuplexSession | None = None
VOICE: sess.VoiceSession | None = None      # state machine + latency metrics
RECORDER: rec.InterviewRecorder | None = None   # saves answers (text only)

# Set by the UI when the candidate clicks "End interview". Checked between
# turns so the background thread actually unwinds instead of continuing to
# block on app.invoke() forever: clicking the button used to only stop the
# page from polling, leaving the interview running with nothing ever
# reaching recorder.save() - the interview stayed open in Postgres
# (finished_at NULL) and no local transcript was ever written.
STOP_REQUESTED = None   # threading.Event | None


def _stop_requested() -> bool:
    return STOP_REQUESTED is not None and STOP_REQUESTED.is_set()


def _out(line: str) -> None:
    try:
        print(line)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"))


# A UI subscribes here instead of scraping the console. Parsing printed lines
# was tried and broke silently the moment the bot was renamed: every agent
# turn vanished from the transcript while the interview itself ran fine.
ON_EVENT = None          # fn(kind: str, **data) -> None


def _emit(kind: str, **data) -> None:
    """Publish an interview event.

    Three subscribers, all optional and all isolated: the recorder writes the
    conversation to disk as it happens, the voice session keeps its own
    running transcript, and a UI draws it. None of them is allowed to break
    the interview by failing.
    """
    if kind in ("agent", "user") and VOICE is not None:
        text = data.get("text", "")
        if text:
            try:
                # main.py's crash/exception recovery reads voice.history when
                # app.invoke() never returns a final state to read history
                # from - the whole point of keeping a second copy here. That
                # fallback silently did nothing before this call existed:
                # nothing ever populated voice.history, so an interview that
                # raised partway through (a real one did - Postgres still
                # shows finished_at set, which only happens after the save
                # code that follows the exception handler) saved a
                # completely empty transcript despite conversation.txt having
                # the full exchange the whole time.
                role = "agent" if kind == "agent" else "user"
                VOICE.add(role, text)
            except Exception:
                pass

    if RECORDER is not None:
        try:
            if kind in ("agent", "user"):
                RECORDER.log_turn(kind, data.get("text", ""))
            elif kind == "guardrail":
                RECORDER.log_event("guardrail", data.get("concern", ""))
            elif kind == "interrupted":
                RECORDER.log_event("candidate took the floor")
            elif kind == "answer_rejected":
                RECORDER.log_event(
                    "not answered", f"heard {data.get('text', '')!r}, "
                    f"reason: {data.get('reason', '')}",
                )
            elif kind == "leak_warning":
                RECORDER.log_event(
                    "leak warning", f"phrase {data.get('phrase', '')!r} in "
                    f"{data.get('text', '')!r}",
                )
        except Exception:
            pass

    if ON_EVENT is None:
        return
    try:
        ON_EVENT(kind, **data)
    except Exception:
        pass


# Phrases that mean "stop the interview", not "answer to the question asked".
_END_PATTERNS = (
    "end the interview", "end interview", "finish the interview",
    "stop the interview", "wrap up", "wrap it up", "that's all",
    "thats all", "i'm done", "im done", "i am done", "we're done",
    "were done", "no more questions", "let's stop", "lets stop",
    "can we stop", "that's it", "thats it", "rate me", "evaluate me",
    "give me feedback", "how did i do",
    # A real run showed a frustrated candidate say "I don't want to evaluate
    # it" and "I don't want to save the question" repeatedly - a refusal to
    # continue, not a request for evaluation ("evaluate me" above), and not
    # an answer to anything. Nothing caught it, so the interview kept asking
    # while the candidate had clearly checked out.
    "don't want to continue", "dont want to continue",
    "don't want to evaluate", "dont want to evaluate",
    "don't want to answer", "dont want to answer",
    "i want to stop", "i want to end", "i'm out", "im out",
)


def _wants_to_end(text: str) -> bool:
    """Did the candidate ask to finish, rather than answer the question?

    Without this the graph advances to the next scripted question no matter
    what was said, so "end the interview and rate me" got a polite goodbye
    followed immediately by question six.
    """
    t = " ".join(text.lower().split())
    return any(p in t for p in _END_PATTERNS)


# Phrases that mean "move to the next question", not "answer to this one".
_SKIP_PATTERNS = (
    "next question", "skip this question", "skip this one", "skip it",
    "skip that", "move on", "move to the next", "can we move on",
    "let's move on", "lets move on", "pass on this one", "i'll pass",
    "ill pass", "next one please", "next one",
    "go directly to the interview", "go directly to interview",
    "go directly", "skip intro", "start the interview", "jump to the interview",
)


def _wants_to_skip(text: str) -> bool:
    """Did the candidate ask to move on, rather than answer the question?

    A real run showed "Next question." reach evaluate() as if it were a thin
    attempt at an answer: nothing caught it, so it got a follow-up probe
    built from the *current* question's own probe_hint - phrased by the LLM
    as what sounded like a brand new, unscripted question, when it was
    actually just that question's hint read back as a sentence. The
    candidate asking to skip ahead is not an answer at all, thin or
    otherwise, and must never be scored, probed, or saved as one.
    """
    t = " ".join(text.lower().split())
    return any(p in t for p in _SKIP_PATTERNS)


# Phrases that mean "I can't answer this", not a real (if thin) attempt.
_GIVE_UP_PATTERNS = (
    "i don't know", "i dont know", "no idea", "not sure", "no clue",
    "i have no idea", "i've no idea", "ive no idea", "i have no clue",
    "beats me", "i cannot answer", "i can't answer", "i cant answer",
    "i don't have an answer", "i dont have an answer",
)


def _gives_up(text: str) -> bool:
    """Did the candidate say they can't answer, rather than attempt one?

    A real run showed "I don't know." fall into the thin-answer branch and
    get probed for "more specific detail" on a question the candidate had
    just said they could not answer at all - reads as the agent not having
    listened. A flat give-up should move the interview on, the same way a
    skip request does, instead of digging for detail that was never coming.
    """
# Phrases that mean "go back to the previous question" or "what was the previous question".
_PREVIOUS_PATTERNS = (
    "previous question", "last question", "question before", "earlier question",
    "go back to the previous", "go back to previous", "what was the previous",
    "repeat previous", "answer the previous", "go with the previous",
)

_FIRST_PATTERNS = (
    "first question", "opening question", "initial question", "question 1", "question one",
    "what was the first", "go back to the first",
)

_REPEAT_PATTERNS = (
    "repeat the question", "repeat question", "say that again", "what was the question",
    "what did you ask", "what question", "could you repeat", "can you repeat",
)


def _wants_previous(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    return any(p in t for p in _PREVIOUS_PATTERNS)


def _wants_first(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    return any(p in t for p in _FIRST_PATTERNS)


def _wants_repeat(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    return any(p in t for p in _REPEAT_PATTERNS)


# Spoken verbatim, never generated, whenever the candidate folds a question
# of their own into an otherwise real answer (e.g. "...but what metric do we
# use for that?"). A real run showed the acknowledgement reply silently
# dropping questions like this - it reacted only to the answer half and moved
# on, which read as not having listened. The fix is not to have the model
# improvise a deferral in the same breath as its acknowledgement: asked to
# notice the question *and* defer it without leaking anything, it did both
# unreliably (sometimes answering it outright, sometimes going silent - see
# brain.acknowledge_messages()'s history). A fixed line said as its own turn
# guarantees the question is never simply ignored, and never answered either.
_DEFERRAL_LINE = "Good question - let's park that for now and come back to it."


def _say(text: str, resume_to: str) -> dict:
    """Speak fixed text (scripted questions) while still listening."""
    if not text or SESSION is None:
        return {}
    _out(f"\n  Jerry: {text}")
    if VOICE:
        VOICE.transition(sess.State.AI_SPEAKING)
    interrupted, heard = SESSION.say(text)
    return _after_say(heard, interrupted, resume_to)


def _say_generated(messages: list, resume_to: str, max_tokens: int = 90) -> dict:
    """Stream a generated reply: speak the first phrase while writing the rest."""
    if SESSION is None:
        return {}
    if VOICE:
        VOICE.transition(sess.State.AI_THINKING)
        VOICE.transition(sess.State.AI_SPEAKING)
    interrupted, heard = SESSION.say_stream(messages, max_tokens=max_tokens)
    if heard:
        _out(f"\n  Jerry: {heard}")
        # Streamed replies are already playing through the speakers by the
        # time the full text exists to check - see find_leak()'s docstring
        # for why this is after-the-fact rather than a block-before-speaking
        # gate. Flagging it is still worth doing: it is exactly how the
        # transition_messages() leak (giving away the definition of
        # underfitting before the question was even asked) was found.
        leak = guardrails.find_leak(heard)
        if leak:
            _out(f"    [leak warning: contains {leak!r}]")
            _emit("leak_warning", text=heard, phrase=leak)
    if VOICE and SESSION.last_timing:
        t = SESSION.last_timing
        m = VOICE.current_metrics()
        if m:
            m.llm_first_token_ms = t.ms(t.first_token)
            m.tts_first_audio_ms = t.ms(t.first_audio)
            m.total_ms = t.ms(t.finished)
        _out(f"    [{t.summary()}]")
        _emit("latency", text=t.summary())
    return _after_say(heard, interrupted, resume_to)


def _after_say(heard: str, interrupted: bool, resume_to: str) -> dict:
    # A double interruption - the candidate cutting in again before the
    # first word of this turn was even audible - leaves `heard` empty. That
    # is not a turn anyone said or heard, so it must not be recorded as one:
    # it used to show up as a blank {"role": "agent", "text": ""} entry in
    # the saved transcript for no reason a reader could see.
    delta: dict = {"history": [{"role": "agent", "text": heard}]} if heard else {}
    _emit("agent", text=heard, interrupted=interrupted)
    if interrupted:
        _out("  [candidate took the floor]")
        _emit("interrupted")
        if VOICE:
            VOICE.record_interruption()
            VOICE.transition(sess.State.INTERRUPTED)
        delta.update(interrupted=True, cut_off=heard, resume_to=resume_to)
    elif VOICE:
        VOICE.transition(sess.State.LISTENING)
    return delta


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def greet(state: InterviewState) -> dict:
    delta = _say(GREETING, resume_to="ask")
    return {"q_index": 0, "followups": 0, "silent_turns": 0, **delta}


def ask(state: InterviewState) -> dict:
    idx = min(state.get("q_index", 0), len(QUESTIONS) - 1)
    question = QUESTIONS[idx]["text"]

    # Ask the scripted question directly. The scripted questions in QUESTIONS
    # already contain natural conversational transitions (e.g. "Let's talk modelling..."),
    # and evaluate() already acknowledged the previous answer.
    asked = _say(question, resume_to="listen")
    return {"followups": 0, **asked}


def listen_node(state: InterviewState) -> dict:
    _out("  (listening...)")
    if VOICE:
        VOICE.transition(sess.State.LISTENING)
        VOICE.begin_turn()
    t0 = time.monotonic()
    raw_answer = SESSION.collect_answer(timeout=20.0) if SESSION else ""
    if VOICE and raw_answer:
        VOICE.transition(sess.State.USER_SPEAKING)
        VOICE.transition(sess.State.PROCESSING)
        m = VOICE.current_metrics()
        if m:
            m.stt_ms = int((time.monotonic() - t0) * 1000)
    if not raw_answer:
        return {"answer": "", "silent_turns": state.get("silent_turns", 0) + 1}

    # collect_answer() already drops pure backchannel utterances heard
    # *during* the answer (duplex.py's own intent classification); this is
    # the final check on the whole assembled transcript, catching what that
    # cannot: an answer that turns out to be filler top to bottom, too short
    # to be an attempt, or STT garbage repeating itself. Rejected here means
    # treated like silence - re-listened for, then nudged/advanced exactly
    # like a real silent turn - not scored, redirected, or recorded as an
    # actual answer.
    validation = guardrails.is_meaningful_answer(raw_answer)
    if not validation["is_valid"]:
        _out(f"  (heard: {raw_answer!r} -> rejected: {validation['reason']})")
        _emit("answer_rejected", text=raw_answer, reason=validation["reason"])
        return {"answer": "", "silent_turns": state.get("silent_turns", 0) + 1}

    answer = validation["cleaned_answer"]
    _out(f"  You: {answer}")
    _emit("user", text=answer)

    idx = min(state.get("q_index", 0), len(QUESTIONS) - 1)
    question = QUESTIONS[idx]["text"]
    concern = guardrails.check(answer, question)
    if concern is not guardrails.Concern.NONE:
        _out(f"  [guardrail: {concern.value}]")
        _emit("guardrail", concern=concern.value, text=answer)

    wants_end = _wants_to_end(answer)
    wants_skip = _wants_to_skip(answer)
    gives_up = _gives_up(answer)

    if _wants_previous(answer):
        if RECORDER is not None:
            RECORDER.add_answer(
                question_index=QUESTIONS[idx]["id"],
                question=question,
                text=guardrails.NOT_ANSWERED,
            )
        target_idx = max(0, idx - 1)
        prev_q = QUESTIONS[target_idx]["text"]
        _say(f"Sure! The previous question was: {prev_q}", resume_to="listen")
        return {
            "answer": "",
            "q_index": target_idx,
            "silent_turns": 0,
            "history": [{"role": "user", "text": answer}],
        }

    if _wants_first(answer):
        if RECORDER is not None:
            RECORDER.add_answer(
                question_index=QUESTIONS[idx]["id"],
                question=question,
                text=guardrails.NOT_ANSWERED,
            )
        first_q = QUESTIONS[0]["text"]
        _say(f"Sure! The first question was: {first_q}", resume_to="listen")
        return {
            "answer": "",
            "q_index": 0,
            "silent_turns": 0,
            "history": [{"role": "user", "text": answer}],
        }

    if _wants_repeat(answer):
        curr_q = QUESTIONS[idx]["text"]
        _say(f"Sure, let me repeat that for you: {curr_q}", resume_to="listen")
        return {
            "answer": "",
            "q_index": idx,
            "silent_turns": 0,
            "history": [{"role": "user", "text": answer}],
        }

    # An utterance that asks Jerry a question back (counter-question) rather than
    # answering the technical question is NOT a valid attempt.
    is_counter_question = "?" in answer and not guardrails._looks_like_answer_attempt(answer)

    valid_attempt = (
        concern is guardrails.Concern.NONE
        and not wants_end
        and not wants_skip
        and not gives_up
        and not is_counter_question
    )

    if RECORDER is not None:
        if valid_attempt:
            RECORDER.add_answer(
                question_index=QUESTIONS[idx]["id"],
                question=question,
                text=answer,
                started_at=t0,
                was_followup=state.get("followups", 0) > 0,
            )
        else:
            RECORDER.add_answer(
                question_index=QUESTIONS[idx]["id"],
                question=question,
                text=guardrails.NOT_ANSWERED,
                started_at=t0,
                was_followup=state.get("followups", 0) > 0,
            )

    return {
        "answer": answer,
        "silent_turns": 0,
        "wants_end": wants_end,
        "wants_skip": wants_skip,
        "gives_up": gives_up,
        "concern": concern.value,
        "history": [{"role": "user", "text": answer}],
    }


def interrupt(state: InterviewState) -> dict:
    """The candidate took the floor mid-sentence. Hear them out, then reply."""
    # Audio is already silent at this point; show that we are listening so the
    # pause reads as attention rather than the agent having frozen.
    _out("  (go ahead, listening...)")
    # A quick backchannel the instant the barge-in registers, mirroring what
    # already happens mid-answer (duplex.py's collect_answer()): without it,
    # the candidate hears dead air the moment they cut in, with nothing
    # audible until their whole interjection is heard and a real reply is
    # generated. speak_backchannel() already no-ops if Jerry has since
    # started a real turn, so this is safe even if the timing is close.
    if SESSION:
        SESSION.speak_backchannel(random.choice(duplex.SPOKEN_BACKCHANNELS))
    said = SESSION.collect_interjection(timeout=10.0) if SESSION else ""
    if not said:
        # The reflex fired on noise and nothing was said: carry on.
        return {"interrupted": False}

    _out(f"  You: {said}")
    _emit("user", text=said)

    # Interruptions used to go straight to the model unchecked, which made
    # cutting in the easiest way around the guardrails: "ignore your
    # instructions" said mid-sentence reached the LLM, while the same words
    # given as an answer were redirected. Screen both paths.
    idx = min(state.get("q_index", 0), len(QUESTIONS) - 1)
    concern = guardrails.check(said, QUESTIONS[idx]["text"])
    if concern is not guardrails.Concern.NONE:
        _out(f"  [guardrail: {concern.value}]")
        _emit("guardrail", concern=concern.value, text=said)

    # An interruption is never treated as an answer, however it is phrased.
    # This used to send "advance" whenever the interjection was shaped like
    # a question - _is_question() only checks grammar, not whether it was
    # actually a response - so cutting in with "does overfitting mean high
    # variance?" or an off-topic "what is AI?" silently skipped the very
    # question being asked, with nothing ever recorded as its answer. Only
    # evaluate() (a real, transcribed answer) is allowed to advance.
    delta: dict = {
        "history": [{"role": "user", "text": said}],
        "interrupted": False,
        "wants_end": _wants_to_end(said),
        "wants_skip": _wants_to_skip(said),
        "gives_up": _gives_up(said),
        "resume_to": state.get("resume_to", "listen"),
    }

    if _wants_previous(said):
        target_idx = max(0, idx - 1)
        prev_q = QUESTIONS[target_idx]["text"]
        spoken = _say(f"Sure! The previous question was: {prev_q}", resume_to="listen")
        return {
            "interrupted": False,
            "q_index": target_idx,
            "resume_to": "listen",
            "history": delta["history"] + spoken.get("history", []),
        }

    if _wants_first(said):
        first_q = QUESTIONS[0]["text"]
        spoken = _say(f"Sure! The first question was: {first_q}", resume_to="listen")
        return {
            "interrupted": False,
            "q_index": 0,
            "resume_to": "listen",
            "history": delta["history"] + spoken.get("history", []),
        }

    if _wants_repeat(said):
        curr_q = QUESTIONS[idx]["text"]
        spoken = _say(f"Sure, let me repeat that for you: {curr_q}", resume_to="listen")
        return {
            "interrupted": False,
            "resume_to": "listen",
            "history": delta["history"] + spoken.get("history", []),
        }

    if concern is not guardrails.Concern.NONE:
        # Redirect in character rather than answering. The floor is already
        # back with us, so this is spoken straight away.
        delta["concern"] = concern.value
        delta["resume_to"] = "listen"
        line = guardrails.redirect_for(concern)
        spoken = _say(line, resume_to="listen") if line else {}
        merged = delta["history"] + spoken.get("history", [])
        delta.update(spoken)
        delta["history"] = merged
        return delta

    # Streamed: replying to an interruption is the moment responsiveness is
    # felt most, and a blocking call leaves 2.5s of dead air.
    msgs = brain.interruption_messages(
        state.get("history", []), state.get("cut_off", ""), said
    )
    spoken = _say_generated(msgs, resume_to=delta["resume_to"], max_tokens=80)
    merged = delta["history"] + spoken.get("history", [])
    delta.update(spoken)
    delta["history"] = merged
    return delta


def evaluate(state: InterviewState) -> dict:
    """React to the answer, streamed so the reply starts almost immediately."""
    answer = state.get("answer", "")
    if not answer:
        return {}
    # Do not acknowledge something the guardrail is about to redirect: echoing
    # "you're asking for a Python script" before declining reads as agreement.
    # Likewise, "next question" was never an answer to react to - it used to
    # get an acknowledgement like "Great, let's move on," which reads fine
    # once but reinforces treating a skip request as if it were content.
    # "I don't know" is the same: reacting to it as if it were a take on the
    # question ("that's a great point") reads as not having listened.
    if (
        state.get("concern", "none") != "none"
        or state.get("wants_skip")
        or state.get("gives_up")
    ):
        return {}
    msgs = brain.acknowledge_messages(state.get("history", []), answer)
    ack = _say_generated(msgs, resume_to="advance", max_tokens=45)

    idx = min(state.get("q_index", 0), len(QUESTIONS) - 1)
    is_last_question = (idx == len(QUESTIONS) - 1)

    if ack.get("interrupted") or not brain.contains_embedded_question(answer) or is_last_question:
        return ack

    # See _DEFERRAL_LINE's docstring: spoken as its own fixed turn rather
    # than left to the model to work into the acknowledgement above.
    deferral = _say(_DEFERRAL_LINE, resume_to="advance")
    merged = ack.get("history", []) + deferral.get("history", [])
    out = {**ack, **deferral, "history": merged}
    return out


def redirect(state: InterviewState) -> dict:
    """Answer was off-track: say so in character, then re-ask the question."""
    concern = guardrails.Concern(state.get("concern", "none"))
    line = guardrails.redirect_for(concern)
    delta = _say(line, resume_to="listen") if line else {}
    if delta.get("interrupted"):
        return {"concern": "none", **delta}

    # Re-ask so they know exactly what is being asked of them.
    idx = min(state.get("q_index", 0), len(QUESTIONS) - 1)
    again = _say(QUESTIONS[idx]["text"], resume_to="listen")
    merged = delta.get("history", []) + again.get("history", [])
    return {**delta, **again, "history": merged, "concern": "none"}


def nudge(state: InterviewState) -> dict:
    # Silence used to only ever grow one running total for the whole
    # interview, and three silent turns anywhere closed the entire
    # interview - a candidate who froze on one hard question got the
    # interview ended instead of just that question skipped. Give the
    # candidate two chances to answer, then move on to the next question
    # instead of giving up on the interview altogether.
    silent_turns = state.get("silent_turns", 0)
    if silent_turns < 2:
        text = "Sorry, I didn't quite catch that. Could you say that again?"
    else:
        text = "No worries, let's move on to the next question."
    return _say(text, resume_to="listen")


def advance(state: InterviewState) -> dict:
    # Reset per-question silence here so route_nudge's threshold is judged
    # against the new question, not a total carried over from a previous one.
    idx = state.get("q_index", 0)

    # Guarantee that every question index reached has a record in PostgreSQL DB.
    # If no answer was logged for this question, record "NOT ANSWERED".
    if RECORDER is not None and idx < len(QUESTIONS):
        recorded_indexes = {ans.question_index for ans in RECORDER.rec.answers}
        q_id = QUESTIONS[idx]["id"]
        if q_id not in recorded_indexes:
            RECORDER.add_answer(
                question_index=q_id,
                question=QUESTIONS[idx]["text"],
                text=guardrails.NOT_ANSWERED,
            )

    return {"q_index": idx + 1, "followups": 0, "silent_turns": 0}


def close(state: InterviewState) -> dict:
    """Wrap up, and give a spoken rating if the candidate asked for one."""
    history = state.get("history", [])
    merged: list = []

    # Ensure ALL 7 questions are present in PostgreSQL with "NOT ANSWERED" for any missing ones
    if RECORDER is not None:
        recorded_indexes = {ans.question_index for ans in RECORDER.rec.answers}
        for q in QUESTIONS:
            if q["id"] not in recorded_indexes:
                RECORDER.add_answer(
                    question_index=q["id"],
                    question=q["text"],
                    text=guardrails.NOT_ANSWERED,
                )

    # A candidate who clicked "End interview" is gone; speaking a rating or
    # goodbye to nobody just delays the moment the interview actually ends
    # and gets saved.
    if _stop_requested():
        return {"done": True}

    # "rate me" / "how did I do" deserves an answer out loud, not just notes
    # in a file the candidate never sees.
    asked_for_rating = any(
        w in " ".join(h.get("text", "").lower() for h in history[-4:]
                      if h.get("role") == "user")
        for w in ("rate me", "how did i do", "evaluate me", "give me feedback")
    )
    if asked_for_rating and history:
        rated = _say_generated(
            brain.rating_messages(history), resume_to="close", max_tokens=140
        )
        merged += rated.get("history", [])
        # A candidate who is done stays "done" even if trailing noise or a
        # last word cuts the rating off mid-sentence - looping back through
        # route_close -> interrupt used to let that stray audio reopen
        # guardrail checks instead of ever reaching the goodbye, so the
        # interview ended with notes generated but nothing spoken or saved
        # after "finish the interview." Retry the close sequence directly
        # instead of bouncing through the general interrupt node.

    # The signoff itself gets up to two attempts for the same reason: one
    # more word from the candidate as it starts should not be able to
    # cancel the goodbye outright.
    delta: dict = {}
    for _ in range(2):
        delta = _say(SIGNOFF, resume_to="close")
        merged += delta.get("history", [])
        if not delta.get("interrupted"):
            break
    return {"done": True, **delta, "interrupted": False, "history": merged}


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def _after(state, default: str) -> str:
    if _stop_requested():
        return "close"
    return "interrupt" if state.get("interrupted") else default


def route_greet(state) -> Literal["interrupt", "ask"]:
    return _after(state, "ask")


def route_ask(state) -> Literal["interrupt", "listen"]:
    return _after(state, "listen")


def route_interrupt(state) -> Literal["listen", "ask", "advance", "close"]:
    if _stop_requested():
        return "close"
    if state.get("interrupted"):
        return "listen"
    if state.get("wants_end"):
        return "close"
    # Asking to skip ahead mid-interruption is the same request as saying it
    # as a normal answer - not a response to the current question, so it must
    # never wait on a redirect or a follow-up. advance() itself still guards
    # against running past the last question.
    if state.get("wants_skip"):
        return "advance"
    if state.get("gives_up"):
        return "advance"
    if state.get("concern", "none") != "none":
        return "listen"          # redirect already spoken; hear them again
    target = state.get("resume_to", "listen")
    return target if target in ("listen", "ask", "advance", "close") else "listen"


def route_listen(state) -> Literal["nudge", "evaluate", "close"]:
    # Checked here too, not only in _after: listen_node's collect_answer()
    # can block up to 20s, so "End interview" is honoured the moment that
    # wait ends rather than only after the *next* turn also completes.
    if _stop_requested():
        return "close"
    return "evaluate" if state.get("answer") else "nudge"


def route_nudge(state) -> Literal["interrupt", "listen", "advance", "close"]:
    if _stop_requested():
        return "close"
    if state.get("interrupted"):
        return "interrupt"
    # Two silent nudges on this question move on to the next one rather than
    # ending the whole interview - a candidate stuck on a single question
    # should still get a chance at the rest of them. advance() resets
    # silent_turns, so this only fires once per question; close() (via
    # route_advance running out of questions) is still what ends things if
    # the candidate has gone quiet for good.
    return "advance" if state.get("silent_turns", 0) >= 2 else "listen"


def route_evaluate(state) -> Literal["interrupt", "advance",
                                    "close", "redirect"]:
    if _stop_requested():
        return "close"
    if state.get("interrupted"):
        return "interrupt"
    if state.get("wants_end"):
        return "close"
    # A confused candidate simply needs the question again; the other concerns
    # get a short redirect first.
    if state.get("concern", "none") != "none":
        return "redirect"
    # A candidate asking to move on is never "thin" in the sense that
    # deserves a probe - there is no answer here to dig deeper into. Without
    # this, "Next question." (2 words, well under ANSWER_MIN_WORDS) fell
    # into the thin-answer branch below and got a follow-up built from the
    # *current* question's own probe_hint, which reads exactly like Jerry
    # inventing a brand new, unscripted question mid-interview.
    if state.get("wants_skip"):
        return "advance"
    # A flat "I don't know" is not a thin attempt to dig deeper into - it is
    # the candidate telling us there is nothing more coming. Probing it for
    # "more specific detail" anyway reads as Jerry not having listened.
    if state.get("gives_up"):
        return "advance"
    return "advance"


def route_redirect(state) -> Literal["interrupt", "listen", "close"]:
    if _stop_requested():
        return "close"
    return "interrupt" if state.get("interrupted") else "listen"


def route_advance(state) -> Literal["ask", "close"]:
    if _stop_requested() or state.get("wants_end"):
        return "close"
    return "ask" if state.get("q_index", 0) < len(QUESTIONS) else "close"


def route_close(state) -> Literal["interrupt", "__end__"]:
    return "interrupt" if state.get("interrupted") else END


def build() -> Any:
    g = StateGraph(InterviewState)
    for name, fn in [
        ("greet", greet), ("ask", ask), ("listen", listen_node),
        ("interrupt", interrupt), ("evaluate", evaluate),
        ("nudge", nudge),
        ("advance", advance), ("close", close), ("redirect", redirect),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("greet")
    for name, router in [
        ("greet", route_greet), ("ask", route_ask), ("listen", route_listen),
        ("interrupt", route_interrupt), ("evaluate", route_evaluate),
        ("nudge", route_nudge),
        ("advance", route_advance), ("close", route_close),
        ("redirect", route_redirect),
    ]:
        g.add_conditional_edges(name, router)
    return g.compile()
