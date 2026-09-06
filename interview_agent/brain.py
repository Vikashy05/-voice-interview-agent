"""Groq-backed speech-to-text and the interviewer's language model. this brain.py"""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import soundfile as sf
from groq import Groq

from . import config as C

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        if not C.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com "
                "and set it before running."
            )
        _client = Groq(api_key=C.GROQ_API_KEY)
    return _client


# Groq's free tier allows 20 speech requests a minute. A live interview can
# generate far more than that, so requests are paced here rather than letting
# the API refuse them mid-answer.
_stt_calls: list[float] = []
_stt_lock = threading.Lock()


def _stt_allowed(priority: bool) -> bool:
    """Is there budget for another transcription right now?

    Final transcripts always go through: losing one loses the answer. Partial
    ones are dropped when the budget is tight, because another will be along
    shortly and nothing depends on any single guess.
    """
    now = time.monotonic()
    with _stt_lock:
        _stt_calls[:] = [t for t in _stt_calls if now - t < 60]
        budget = C.STT_RPM if priority else C.STT_RPM - C.STT_RESERVE
        if len(_stt_calls) >= budget:
            return False
        _stt_calls.append(now)
        return True


def transcribe(audio: np.ndarray, priority: bool = True) -> str:
    """Send a mono float32 waveform to Whisper and return the text.

    `priority=False` marks a partial transcript, which may be skipped when the
    rate budget is low.
    """
    if audio is None or len(audio) == 0:
        return ""

    # Too short to be speech: transcribing a fragment of room noise invites
    # Whisper to hallucinate a plausible sentence out of nothing.
    if len(audio) < C.SAMPLE_RATE * (C.STT_MIN_SECONDS):
        return ""

    if not _stt_allowed(priority):
        return ""

    buf = io.BytesIO()
    sf.write(buf, audio, C.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)

    for attempt in range(C.STT_RETRIES):
        try:
            resp = client().audio.transcriptions.create(
                file=("speech.wav", buf.getvalue()),
                model=C.STT_MODEL,
                language="en",
            )
            return _drop_hallucination((resp.text or "").strip())
        except Exception as exc:
            msg = str(exc)
            if "rate_limit" in msg or "429" in msg:
                if attempt + 1 >= C.STT_RETRIES or not priority:
                    print("  [stt: rate limited, giving up on this transcript]")
                    return ""          # give up rather than stall further
                wait = C.STT_BACKOFF * (attempt + 1)
                print(f"  [stt: rate limited, retrying in {wait:.1f}s "
                      f"(attempt {attempt + 1}/{C.STT_RETRIES})]")
                time.sleep(wait)
                continue
            print(f"  [stt error: {msg[:120]}]")
            return ""
    return ""


# Whisper invents these when handed silence or noise; they are not speech.
_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "bye.", "so", "so.", "okay", ".", "..", "...", "uh", "um",
    "thanks", "please subscribe", "subtitles by the amara.org community",
    "i'm going to go to the house", "the end", "music", "[music]",
}


def _drop_hallucination(text: str) -> str:
    """Discard the stock phrases Whisper produces from background noise."""
    if not text:
        return ""
    bare = " ".join(text.lower().strip().strip(".,!?").split())
    bare = bare.replace(chr(8217), chr(39))       # curly apostrophe from TTS
    if bare in _HALLUCINATIONS:
        return ""
    # A very short result from a longer clip is usually noise, not an answer.
    if len(bare) <= 2:
        return ""
    return text


SYSTEM = """You are Jerry, a warm and experienced technical interviewer running \
a live voice interview. You are speaking out loud, so your replies are converted \
directly to speech.

Rules:
- Keep every reply to one or two short sentences. This is conversation, not prose.
- Sound like a real person: natural, warm, varied. Never robotic or listy.
- Never use markdown, bullet points, emoji, or stage directions.
- React genuinely to what the candidate actually said before moving on.
- Do not answer the interview questions yourself and do not coach the candidate.
"""


def _chat(messages: list[dict], max_tokens: int = 90) -> str:
    """One completion.

    gpt-oss models emit hidden reasoning tokens that can eat the whole budget
    and leave `content` empty, so reasoning is dialled down and the budget is
    given headroom.
    """
    try:
        resp = client().chat.completions.create(
            model=C.LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens + C.REASONING_HEADROOM,
            temperature=0.7,
            reasoning_effort=C.REASONING_EFFORT,
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text and choice.finish_reason == "length":
            print("  [llm truncated before answering]")
        return _clean(text)
    except Exception as exc:
        print(f"  [llm error: {exc}]")
        return ""


# Unicode the model likes to emit that a cp1252 console cannot print.
_PUNCT = {
    "—": ", ", "–": ", ", "‒": "-", "‑": "-", "‐": "-",
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "…": "...", " ": " ", " ": " ", "\u200b": "",
}


def _asciify(text: str) -> str:
    """Fold typographic Unicode down to plain ASCII.

    Speech synthesis does not care, but the Windows console does: one stray
    non-breaking hyphen used to abort the whole interview with a
    UnicodeEncodeError.
    """
    for bad, good in _PUNCT.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")


def _clean(text: str) -> str:
    """Strip anything that would sound wrong when spoken aloud."""
    if not text:
        return ""
    text = text.replace("*", "").replace("#", "").replace("`", "")
    # The model sometimes wraps a quoted question in stray quotes, which the
    # voice reads as a literal character or an odd pause.
    text = text.strip().strip('"').strip()
    text = _asciify(text)
    # Drop a leading speaker label the model sometimes adds.
    for label in ("Interviewer:", "Jerry:", "You:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    return " ".join(text.split())


def warm_up() -> None:
    """Open the connection before the interview starts.

    The first Groq call pays TLS and cold-start cost - measured at ~3.5s
    against ~0.4s once warm. Paying that here keeps it out of the first
    interruption, where the delay is most obvious.
    """
    try:
        client().chat.completions.create(
            model=C.LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            reasoning_effort=C.REASONING_EFFORT,
        )
    except Exception:
        pass


def _history_block(history: list[dict], limit: int | None = None) -> str:
    """Render the conversation so far for a prompt.

    The whole interview is sent, not a trailing window. Six turns meant
    the agent could not refer back to anything said earlier: by the third
    question the first answer had already scrolled out of view, so probes
    revisited ground the candidate had covered and the closing rating was
    based on the tail of the interview rather than all of it.

    Only a runaway session is trimmed, and then from the oldest end.
    """
    turns = list(history or [])
    limit = C.HISTORY_TURNS if limit is None else limit
    if limit:
        turns = turns[-limit:]

    lines = [_line(t) for t in turns if t.get("text")]

    # Drop from the front until it fits, so the most recent exchange -
    # the one the reply must actually answer - is never what gets cut.
    block = "\n".join(lines)
    while len(block) > C.HISTORY_MAX_CHARS and len(lines) > 2:
        lines.pop(0)
        block = "\n".join(lines)
    return block


def _line(turn: dict) -> str:
    who = "Interviewer" if turn.get("role") == "agent" else "Candidate"
    return f"{who}: {turn.get('text', '')}"


def full_transcript(history: list[dict]) -> str:
    """Every turn, verbatim - for the spoken rating and debrief notes."""
    return "\n".join(
        _line(t) for t in (history or []) if t.get("text")
    )


def contains_embedded_question(answer: str) -> bool:
    """Did the candidate ask something of their own inside this turn?

    Only flags candidate questions that ask for direct answers/solutions
    or ask personal questions about the interviewer. Meta-questions about
    the interview flow or question repetition return False.
    """
    if not answer or "?" not in answer:
        return False

    t = " ".join(answer.lower().split())

    # Do NOT trigger deferral line for interview flow questions or clarifications
    meta_phrases = (
        "previous question", "first question", "repeat", "what was the question",
        "what did you ask", "why should i", "how long", "what topics", "what is the scope"
    )
    if any(m in t for m in meta_phrases):
        return False

    # Defer when candidate asks for solutions or personal questions
    defer_phrases = (
        "can you answer", "give me the answer", "tell me the answer",
        "what is your name", "who are you", "what do you do", "solve this for me"
    )
    return any(p in t for p in defer_phrases)


def acknowledge_messages(history: list[dict], answer: str) -> list[dict]:
    """Prompt for a brief reaction. Shared by the blocking and streaming paths."""
    instruction = (
        "Reply with a single short, natural acknowledgement that shows you "
        "listened. Reference something specific they said. Do not ask a "
        "question. Maximum 15 words."
    )
    if contains_embedded_question(answer):
        instruction += (
            " They also asked you a question of their own in there - do not "
            "answer it, do not explain anything, and do not react to it at "
            "all. Acknowledge only the rest of what they said."
        )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{_history_block(history)}\n\n"
                f"The candidate just said: \"{answer}\"\n\n"
                f"{instruction}"
            ),
        },
    ]


def acknowledge(history: list[dict], answer: str) -> str:
    """A brief human reaction to the answer just given."""
    return _chat(acknowledge_messages(history, answer), max_tokens=45)


def transition_messages(history: list[dict], next_question: str) -> list[dict]:
    """Prompt for a short lead-in only.

    The scripted question is spoken verbatim afterwards rather than being
    regenerated: streaming a rewritten question would put audio on the
    speakers before it could be checked against the script.
    """
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{_history_block(history)}\n\n"
                f"You are about to ask: \"{next_question}\"\n\n"
                "Say only a very short, natural lead-in of three to six words "
                "that reacts to what they just said. Do NOT ask the question "
                "itself and do not add anything else. Critically: do not "
                "explain, define, or hint at the concept the upcoming "
                "question is about, even if the candidate just asked you to "
                "define it or explain it themselves - that is the question "
                "they are about to be asked, and answering it for them here "
                "defeats the point of asking it. Stay purely conversational "
                "(e.g. \"Sure, let's dive in\" or \"Got it, moving on\")."
            ),
        },
    ]


def bridge(history: list[dict], next_question: str) -> str:
    """Blend an acknowledgement into the next scripted question."""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{_history_block(history)}\n\n"
                f"You must now ask this exact question: \"{next_question}\"\n\n"
                "Add a very short natural transition (three to six words) before it, "
                "then ask the question with its meaning fully intact. "
                "Return only what you will say out loud."
            ),
        },
    ]
    out = _chat(msgs, max_tokens=90)
    # Never let the model drop or mangle the scripted question.
    if not out or len(out) < len(next_question) * 0.55:
        return next_question
    return _dedupe_transition(out, next_question)


def _dedupe_transition(said: str, question: str) -> str:
    """Drop a transition that just repeats the question's own opening words.

    The model likes to prepend things like "Last one," to a question that
    already begins "Last one." Spoken aloud that stutters, so the transition
    is trimmed back to the point where the real question starts.
    """
    idx = said.lower().find(question.lower()[:24])
    if idx <= 0:
        return said
    lead, rest = said[:idx].strip(), said[idx:]
    # Compare on words, ignoring punctuation and case.
    lead_words = {w.strip(".,!?;:").lower() for w in lead.split()}
    head_words = {w.strip(".,!?;:").lower() for w in question.split()[:5]}
    if lead_words & head_words:
        return rest.strip()
    return said


def interruption_messages(history: list[dict], cut_off: str, said: str) -> list[dict]:
    """Prompt for yielding gracefully. Shared by blocking and streaming paths.

    "If they asked something, answer it briefly" used to be unqualified, and
    a live run showed exactly the failure that invites: told to "continue
    the interview" mid-redirect, the model filled the gap by inventing its
    own interview question ("the difference between supervised and
    unsupervised learning") that was never in questions.py and was never
    tracked as asked. The fixed question list is the one thing that must
    never drift, so the prompt now says explicitly what it may not do.
    """
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{_history_block(history)}\n\n"
                f"You were mid-sentence saying: \"{cut_off}\"\n"
                f"The candidate interrupted with: \"{said}\"\n\n"
                "They cut you off, so yield gracefully. If they asked something "
                "about the interview process itself (how long, what's next, can "
                "we skip ahead), answer that briefly. If they were adding to "
                "their answer, acknowledge it. If they said something unrelated "
                "or just told you to continue, acknowledge them and say you'll "
                "get back to the question - do not ask a different or new "
                "interview question of your own; the only interview questions "
                "come from the fixed script, never invented here. One or two "
                "short sentences, and never scold them for interrupting."
            ),
        },
    ]


def handle_interruption(history: list[dict], cut_off: str, said: str) -> str:
    """Respond when the candidate talks over the agent."""
    return _chat(interruption_messages(history, cut_off, said), max_tokens=80)


def rating_messages(history: list[dict]) -> list[dict]:
    """Prompt for a spoken rating, addressed to the candidate.

    The debrief notes are written for a hiring team; this is said out loud to
    the person who just asked how they did, so it is short and second-person.
    """
    transcript = full_transcript(history)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Interview transcript:\n{transcript}\n\n"
                "The candidate asked how they did. Speaking directly to them, "
                "give an out-of-ten score, one thing they did well, and one "
                "thing to work on. Base it strictly on what they actually "
                "said. Three or four short sentences, warm and direct, "
                "spoken aloud so no markdown or lists."
            ),
        },
    ]


def summarize(history: list[dict], questions: list[dict]) -> str:
    """Post-interview notes for the hiring team."""
    transcript = full_transcript(history)
    msgs = [
        {
            "role": "system",
            "content": "You write concise, fair interview debrief notes.",
        },
        {
            "role": "user",
            "content": (
                f"Full interview transcript:\n{transcript}\n\n"
                "Write short debrief notes with these sections: Summary, "
                "Strengths, Areas to probe further, and Overall impression. "
                "Base every claim strictly on what the candidate actually said. "
                "Plain text, no markdown."
            ),
        },
    ]
    return _chat(msgs, max_tokens=600)
