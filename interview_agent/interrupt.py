"""Two-level barge-in.


This splits the decision in two, the way a person does it:

  LEVEL 1 - reflex, energy only, <100ms
      Sustained speech-band energy while the agent talks stops the audio at
      once. No transcription, no LLM, no waiting. Cheap and instant.

  LEVEL 2 - judgement, transcription, ~1-2s later
      The captured audio is transcribed and classified. A real interruption is
      confirmed and the floor stays with the user. A false trigger (a cough, a
      nod, the agent's own echo) is retracted and the response resumes.

Stopping instantly and occasionally resuming feels natural. Talking over
someone for two seconds never does.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import config as C


class Verdict(Enum):
    PENDING = "pending"        # stopped on reflex, awaiting transcription
    CONFIRMED = "confirmed"    # genuine interruption, stay stopped
    RETRACTED = "retracted"    # false alarm, resume speaking


@dataclass
class Interruption:
    """One barge-in, from reflex through to confirmation."""

    generation_id: int
    detected_at: float                      # VAD fired
    audio_stopped_at: float | None = None   # playback actually silent
    verdict: Verdict = Verdict.PENDING
    text: str = ""
    audio: list = field(default_factory=list)

    @property
    def latency_ms(self) -> int | None:
        """The number that matters: speech detected -> audio silent."""
        if self.audio_stopped_at is None:
            return None
        return int((self.audio_stopped_at - self.detected_at) * 1000)


class BargeInDetector:
    """Level 1. Frame-by-frame energy reflex, no transcription in the path."""

    def __init__(self) -> None:
        self.loud_frames = 0
        self._armed_at = 0.0

    def arm(self) -> None:
        """Called when the agent starts speaking."""
        self.loud_frames = 0
        self._armed_at = time.monotonic()

    def feed(self, frame: np.ndarray, rms: float,
             gate: float | None = None) -> bool:
        """Return True the instant a barge-in should fire.

        `gate` is supplied by the session so the threshold can adapt to the
        speaker bleed actually measured on this machine.
        """
        # The speaker's own onset must not trip the detector.
        if (time.monotonic() - self._armed_at) * 1000 < C.BARGE_GRACE_MS:
            return False

        if rms >= (C.BARGE_RMS if gate is None else gate):
            self.loud_frames += 1
            # Consecutive frames, so a click or a chair creak is not enough.
            return self.loud_frames >= C.BARGE_FRAMES
        self.loud_frames = 0
        return False


class GenerationGuard:
    """Invalidates in-flight work so stale chunks can never play.

    Cancelling a task is not sufficient on its own: a synthesis already in
    progress will still deliver its buffer. Every unit of work carries the
    generation it belongs to and is dropped if that generation is stale.
    """

    def __init__(self) -> None:
        self._id = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._id

    def begin(self) -> int:
        """Start a new generation and return its id."""
        with self._lock:
            self._id += 1
            return self._id

    def is_stale(self, generation_id: int) -> bool:
        with self._lock:
            return generation_id != self._id

    def invalidate(self) -> int:
        """Abandon the current generation; everything in flight is now stale."""
        return self.begin()


class InterruptionManager:
    """Coordinates the reflex stop, cancellation, and later confirmation."""

    def __init__(self, log=None) -> None:
        self.generations = GenerationGuard()
        self.detector = BargeInDetector()

        # Set the moment a barge-in fires. Shared with the streaming voice so
        # generation and playback both stop without another round trip.
        self.interrupted = threading.Event()

        self.current: Interruption | None = None
        self.history: list[Interruption] = []
        self._lock = threading.Lock()
        self._log = log or (lambda *a, **k: None)

        # Callbacks wired by the session.
        self.on_stop_audio = None      # stop playback right now
        self.on_clear_queue = None     # drop pending synthesized audio

    # -- level 1 -----------------------------------------------------------
    def arm(self) -> int:
        """Agent is about to speak: start a generation and arm the detector."""
        gen = self.generations.begin()
        self.interrupted.clear()
        self.detector.arm()
        with self._lock:
            self.current = None
        return gen

    def fire(self, generation_id: int) -> Interruption | None:
        """Level 1 reflex. Stops audio immediately; returns the record."""
        with self._lock:
            if self.interrupted.is_set():
                return self.current            # already interrupted
            detected = time.monotonic()
            self._log("[VAD] user speech detected")

            # Order matters: signal first so the generator and playback loops
            # see it on their next check, then tear down.
            self.interrupted.set()
            self._log("[INTERRUPT] triggered")

            self.generations.invalidate()
            self._log("[LLM/TTS] generation invalidated")

            rec = Interruption(generation_id=generation_id, detected_at=detected)
            self.current = rec
            self.history.append(rec)

        # Callbacks run outside the lock: stopping a stream can block briefly.
        if self.on_stop_audio is not None:
            try:
                self.on_stop_audio()
            except Exception:
                pass
        if self.on_clear_queue is not None:
            try:
                self.on_clear_queue()
            except Exception:
                pass

        rec.audio_stopped_at = time.monotonic()
        self._log(f"[AUDIO] playback stopped ({rec.latency_ms}ms)")
        return rec

    # -- level 2 -----------------------------------------------------------
    def confirm(self, text: str, is_real: bool) -> Verdict:
        """Transcription arrived: keep the floor with the user, or resume."""
        with self._lock:
            rec = self.current
            if rec is None:
                return Verdict.RETRACTED
            rec.text = text
            rec.verdict = Verdict.CONFIRMED if is_real else Verdict.RETRACTED
        if rec.verdict is Verdict.RETRACTED:
            self._log(f"[INTERRUPT] retracted (heard {text!r})")
        else:
            self._log(f"[INTERRUPT] confirmed: {text!r}")
        return rec.verdict

    # -- reporting ---------------------------------------------------------
    def stats(self) -> dict:
        done = [i for i in self.history if i.latency_ms is not None]
        lat = [i.latency_ms for i in done]
        return {
            "interruptions": len(self.history),
            "confirmed": sum(1 for i in self.history if i.verdict is Verdict.CONFIRMED),
            "retracted": sum(1 for i in self.history if i.verdict is Verdict.RETRACTED),
            "avg_stop_latency_ms": int(sum(lat) / len(lat)) if lat else None,
            "max_stop_latency_ms": max(lat) if lat else None,
        }
