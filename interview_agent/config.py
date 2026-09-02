"""Configuration for the voice interview agent."""
import os
from pathlib import Path

# Load .env from the project root so the key never has to live in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # dotenv is optional; env vars still work
    pass

_DATABASE_URL_SCHEMES = ("postgresql://", "postgres://")


def _validated_database_url(raw: str) -> str:
    """Catch a malformed DATABASE_URL here, with a clear message.

    Left unchecked, a typo'd scheme or a stray copy-paste only surfaces deep
    inside psycopg.connect() - questions.py and recorder.py both treat that
    as "no database configured" and silently fall back to the local file
    store, so a real typo could go unnoticed for an entire interview instead
    of failing loudly at startup where it is actually fixable.
    """
    if not raw:
        return raw
    if not raw.startswith(_DATABASE_URL_SCHEMES):
        raise ValueError(
            "DATABASE_URL must start with postgresql:// or postgres:// "
            f"- got: {raw[:20]!r}..."
        )
    return raw


# --- Database ---
# Questions are read from here (falling back to questions.py if unset), and
# every answer is written here in addition to the local recording.
DATABASE_URL = _validated_database_url(os.environ.get("DATABASE_URL", ""))

# --- Groq models (free tier) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "openai/gpt-oss-20b"          # fast conversational turns
STT_MODEL = "whisper-large-v3"      # fast transcription

# gpt-oss emits hidden reasoning tokens; keep them minimal and leave the
# answer budget room so `content` is never starved to empty.
REASONING_EFFORT = "low"
REASONING_HEADROOM = 120

# --- Identity ---
BOT_NAME = "Jerry"

# --- Voice ---
TTS_VOICE = "en-IN-NeerjaExpressiveNeural"   # Indian English, female
TTS_RATE = "+10%"      # brisk pace, chosen from a real A/B listening test
TTS_PITCH = "+5Hz"     # slightly higher, chosen from a real A/B listening test

# --- Audio I/O ---
SAMPLE_RATE = 16000        # mic capture rate (whisper-friendly)
BLOCK_MS = 30              # mic frame size in ms
PLAYBACK_BLOCK = 256       # frames per playback write; small => fast barge-in

# --- Voice activity detection (energy based) ---
VAD_START_RMS = 0.0210      # RMS above this = speech started
VAD_KEEP_RMS = 0.0105       # hysteresis: stay in speech above this
SILENCE_HANG_MS = 900      # trailing silence that ends a user turn
MIN_ANSWER_WORDS = 2       # a lone word while idle is almost always noise
MIN_SPEECH_MS = 500        # ignore blips shorter than this (noise, clicks)
MAX_UTTERANCE_S = 45       # hard cap on one answer

# --- Barge-in (interrupting the agent while it talks) ---
# Measured on this machine: room noise 0.0003, speaker bleed peaks at 0.012.
# The gate must clear the bleed without demanding a shout. 0.055 was ~5x above
# anything the microphone ever produced, so barge-in could never fire.
BARGE_RMS = 0.0160          # louder gate while agent speaks (mic hears speaker)
BARGE_FRAMES = 8           # consecutive loud frames (~240ms) - a cough or

# The barge gate adapts to measured speaker bleed rather than trusting one
# fixed number: BARGE_RMS above is the ceiling, BARGE_RMS_MIN the floor.
BARGE_RMS_MIN = 0.0092      # never gate lower than this
BLEED_MARGIN = 1.6         # gate sits this far above observed bleed
BLEED_DECAY = 0.995        # bleed estimate forgets old peaks slowly
BARGE_GRACE_MS = 500       # ignore the mic briefly after the agent starts

# How much of the conversation each prompt carries. The whole interview is
# sent so the agent can reference anything the candidate said earlier, not
# just the last few turns; the cap only guards against a runaway session.
HISTORY_TURNS = 0          # 0 = every turn
HISTORY_MAX_CHARS = 24000  # trim oldest only past this
ANSWER_MIN_WORDS = 8       # shorter than this => likely needs a probe
ANSWER_MIN_MEANINGFUL_WORDS = 2   # fewer non-filler words than this => not a real answer

# --- Full duplex ---
# Long answers are flushed to recognition every so often, so transcription
# overlaps speech instead of waiting for a full stop.
DUPLEX_FLUSH_MS = 12000    # cut a running utterance after this much speech

# Partial transcription of an answer that is still being spoken. This is what
# makes ordinary answers duplex: without it nothing is recognised until the
# speaker stops, so only very long answers ever overlapped.
PARTIAL_AFTER_MS = 1200    # start guessing once this much has been said
PARTIAL_EVERY_MS = 4000    # superseded below by STT pacing
BACKCHANNEL_MAX_WORDS = 3  # short filler that must not steal the floor

# --- Streaming LLM -> TTS ---
# Chunks are cut on sentence/clause boundaries so speech keeps its prosody.
CHUNK_MIN_CHARS = 18       # never speak a fragment shorter than this
CHUNK_SOFT_CHARS = 70      # past this, a comma is a good enough boundary
# Measured across every saved interview: the median agent reply is ~69 chars
# and Jerry is instructed to keep replies to one or two short sentences, so
# most turns are one sentence with no internal comma - nothing before the
# final period ever reached the old 140-char cap, meaning chunk_stream()
# could not emit anything until the whole sentence had already been
# generated. TTS then only ever started once the full reply existed, which
# is indistinguishable from not streaming at all. Lowering the cap close to
# that median means the hard cut actually fires on a typical reply, giving
# genuine early audio instead of only helping the longer quarter of turns.
CHUNK_MAX_CHARS = 65       # hard cap so a run-on sentence still starts
SYNTH_LOOKAHEAD = 2        # chunks synthesised ahead of playback

# --- Echo suppression ---
# Without headphones the mic hears the agent. Speech overlapping this much
# with what the agent is currently saying is treated as its own echo.
ECHO_OVERLAP = 0.55
# A user talking over the agent is heard as a mixture of both voices. This many
# words that are NOT the agent's means a real person is speaking, so the segment
# must never be discarded as echo.
ECHO_MIN_NOVEL_WORDS = 2

# While the agent is speaking, end the user's segment sooner: they are cutting
# in, so waiting the full conversational hangover feels unresponsive.
INTERRUPT_HANG_MS = 450

# --- Speech-to-text pacing ---
# Groq free tier allows 20 requests/minute. Partial transcripts are optional,
# so a slice of the budget is reserved for the final ones that carry answers.
STT_RPM = 18               # stay just under the limit
STT_RESERVE = 6            # budget kept for final transcripts only
STT_RETRIES = 3
STT_BACKOFF = 2.0          # seconds, multiplied by attempt number
STT_MIN_SECONDS = 0.45     # shorter clips are noise, not speech

# Partial transcription cadence. Every 1.5s used 40 requests/minute on its own,
# double the free-tier limit; 4s keeps the live display without exhausting it.
PARTIAL_EVERY_MS = 4000
