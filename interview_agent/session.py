"""Session state, turn-taking state machine, and latency metrics.

The store is defined as an interface with an in-memory implementation, so the
same calls work unchanged against Redis or Postgres later: only `SessionStore`
needs a new subclass, not any of the calling code.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class State(Enum):
    """Conversation states. Transitions are validated, not assumed."""

    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"
    AI_THINKING = "ai_thinking"
    AI_SPEAKING = "ai_speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


# An illegal transition means a bug in turn-taking, so it is worth catching
# rather than letting the session drift into a nonsensical state.
ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.AI_SPEAKING, State.ERROR},
    # A scripted question needs no generation step, so listening may go
    # straight to speaking; generated replies still pass through thinking.
    State.LISTENING: {State.USER_SPEAKING, State.AI_THINKING, State.AI_SPEAKING,
                      State.IDLE, State.ERROR},
    State.USER_SPEAKING: {State.PROCESSING, State.LISTENING, State.ERROR},
    State.PROCESSING: {State.AI_THINKING, State.AI_SPEAKING, State.LISTENING,
                       State.ERROR},
    State.AI_THINKING: {State.AI_SPEAKING, State.INTERRUPTED, State.LISTENING, State.ERROR},
    State.AI_SPEAKING: {State.INTERRUPTED, State.LISTENING, State.IDLE, State.ERROR},
    State.INTERRUPTED: {State.LISTENING, State.USER_SPEAKING, State.AI_THINKING,
                        State.AI_SPEAKING, State.ERROR},
    State.ERROR: {State.IDLE, State.LISTENING},
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class TurnMetrics:
    """Latency breakdown for a single conversational turn."""

    turn_id: str
    stt_ms: int | None = None
    llm_first_token_ms: int | None = None
    tts_first_audio_ms: int | None = None
    total_ms: int | None = None
    interrupted: bool = False


@dataclass
class Event:
    """One structured log record."""

    session_id: str
    turn_id: str | None
    event: str
    timestamp: float
    state_before: str | None = None
    state_after: str | None = None
    latency_ms: int | None = None
    detail: dict = field(default_factory=dict)


class VoiceSession:
    """Per-conversation state, guarded so concurrent threads stay consistent."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self._state = State.IDLE
        self._lock = threading.RLock()

        self.turn_id: str | None = None
        self.turn_index = 0
        self.user_speaking = False
        self.ai_speaking = False

        self.history: list[dict] = []
        self.events: list[Event] = []
        self.metrics: list[TurnMetrics] = []
        self.interruptions = 0

        # Set when the user takes the floor; shared with the streaming voice
        # so generation is cancelled, not merely muted.
        self.interruption = threading.Event()

    # -- state -------------------------------------------------------------
    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def transition(self, to: State, *, strict: bool = False) -> bool:
        """Move to a new state, refusing transitions that make no sense."""
        with self._lock:
            frm = self._state
            if to is frm:
                return True
            if to not in ALLOWED.get(frm, set()):
                self.log("invalid_transition", state_before=frm.value,
                         state_after=to.value)
                if strict:
                    raise InvalidTransition(f"{frm.value} -> {to.value}")
                return False
            self._state = to
            self.ai_speaking = to is State.AI_SPEAKING
            self.user_speaking = to is State.USER_SPEAKING
            self.log("state_change", state_before=frm.value, state_after=to.value)
            return True

    # -- turns -------------------------------------------------------------
    def begin_turn(self) -> str:
        with self._lock:
            self.turn_index += 1
            self.turn_id = f"turn_{self.turn_index}"
            self.metrics.append(TurnMetrics(turn_id=self.turn_id))
            return self.turn_id

    def current_metrics(self) -> TurnMetrics | None:
        return self.metrics[-1] if self.metrics else None

    def record_interruption(self, latency_ms: int | None = None) -> None:
        with self._lock:
            self.interruptions += 1
            m = self.current_metrics()
            if m is not None:
                m.interrupted = True
            self.log("user_interrupted_ai", latency_ms=latency_ms)

    def add(self, role: str, text: str) -> None:
        with self._lock:
            self.history.append({"role": role, "text": text, "t": time.time()})

    # -- logging -----------------------------------------------------------
    def log(self, event: str, **kw: Any) -> None:
        self.events.append(
            Event(
                session_id=self.session_id,
                turn_id=self.turn_id,
                event=event,
                timestamp=time.time(),
                **kw,
            )
        )

    def summary(self) -> dict:
        done = [m for m in self.metrics if m.total_ms]
        def avg(vals):
            vals = [v for v in vals if v]
            return int(sum(vals) / len(vals)) if vals else None
        return {
            "session_id": self.session_id,
            "duration_s": round(time.time() - self.started_at, 1),
            "turns": len(self.metrics),
            "interruptions": self.interruptions,
            "avg_stt_ms": avg([m.stt_ms for m in self.metrics]),
            "avg_llm_first_token_ms": avg([m.llm_first_token_ms for m in self.metrics]),
            "avg_tts_first_audio_ms": avg([m.tts_first_audio_ms for m in self.metrics]),
            "avg_turn_ms": avg([m.total_ms for m in done]),
        }


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
class SessionStore:
    """Interface. Swap in Redis/Postgres by subclassing this."""

    def save(self, session: VoiceSession) -> None: ...
    def load(self, session_id: str) -> dict | None: ...
    def list_sessions(self) -> list[str]: ...


class MemoryStore(SessionStore):
    """In-process store with an optional JSON mirror on disk.

    Keeps the system fully runnable with no external services, while matching
    the interface a Redis or Postgres backend would implement.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.path = Path(path) if path else None

    def _serialise(self, s: VoiceSession) -> dict:
        return {
            "session_id": s.session_id,
            "started_at": s.started_at,
            "state": s.state.value,
            "history": s.history,
            "summary": s.summary(),
            "metrics": [asdict(m) for m in s.metrics],
            "events": [asdict(e) for e in s.events],
        }

    def save(self, session: VoiceSession) -> None:
        blob = self._serialise(session)
        with self._lock:
            self._data[session.session_id] = blob
            if self.path is not None:
                try:
                    self.path.write_text(
                        json.dumps(blob, indent=2, default=str), encoding="utf-8"
                    )
                except OSError:
                    pass                    # persistence is best-effort

    def load(self, session_id: str) -> dict | None:
        with self._lock:
            return self._data.get(session_id)

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._data)


class LayeredMemory:
    """Working / session / long-term memory.

    Only what is relevant is handed to the model: the full transcript grows
    without bound, and stuffing all of it into every prompt costs latency and
    dilutes the context.
    """

    def __init__(self, store: SessionStore | None = None) -> None:
        self.store = store or MemoryStore()
        self.working: dict[str, Any] = {}       # current turn
        self.facts: list[str] = []              # durable notes about the user

    def note(self, fact: str) -> None:
        if fact and fact not in self.facts:
            self.facts.append(fact)

    def context(self, history: list[dict], limit: int = 6) -> list[dict]:
        """Recent turns plus any durable facts."""
        recent = history[-limit:]
        if not self.facts:
            return recent
        preamble = {"role": "note", "text": "Known: " + "; ".join(self.facts[-5:])}
        return [preamble] + recent
