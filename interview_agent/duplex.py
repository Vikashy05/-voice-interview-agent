"""Full-duplex conversation layer.

Playback, microphone capture, VAD, final recognition, and partial recognition
run concurrently so the interviewer can be interrupted naturally.

The public API is intentionally preserved:

    DuplexSession.start()
    DuplexSession.stop()
    DuplexSession.say()
    DuplexSession.say_stream()
    DuplexSession.next_utterance()
    DuplexSession.collect_interjection()
    DuplexSession.collect_answer()
    DuplexSession.drain_pending()

This module only handles conversational audio and turn-taking. Interview
semantics such as OFF_TOPIC, PROMPT_ATTACK, ASKS_ANSWER, etc. belong to the
separate concern/guard layer.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from . import audio, brain, interrupt
from . import config as C


# ============================================================================
# ENUMS
# ============================================================================

class Channel(Enum):
    """What the interviewer is currently doing."""

    IDLE = "idle"
    SPEAKING = "speaking"
    YIELDED = "yielded"


class Intent(Enum):
    """What the user's speech means for turn-taking."""

    BACKCHANNEL = "backchannel"
    INTERRUPT = "interrupt"
    CONTINUE = "continue"


# ============================================================================
# TURN-TAKING VOCABULARY
# ============================================================================

_FILLER = {
    "a",
    "the",
    "and",
    "so",
    "well",
    "just",
    "very",
    "really",
}

BACKCHANNELS = {
    "mhm",
    "mm",
    "mmm",
    "mm-hm",
    "mhmm",
    "um",
    "umm",
    "uh",
    "uh huh",
    "uh-huh",
    "ah",
    "yeah",
    "yep",
    "yes",
    "ok",
    "okay",
    "right",
    "sure",
    "got it",
    "i see",
    "makes sense",
    "of course",
    "true",
    "exactly",
    "cool",
    "hmm",
    "oh",
    "wow",
    "nice",
    "gotcha",
    "understood",
}

SPOKEN_BACKCHANNELS = (
    "Hmm.",
    "Right.",
    "Okay.",
    "I see.",
    "Got it.",
    "Carry on.",
)

_BACKCHANNEL_RATE = "-25%"
_BACKCHANNEL_VOLUME = 0.55


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def classify(text: str) -> Intent:
    """Classify speech as a backchannel or a real attempt to take the floor.

    Examples:

        "mm-hm"                 -> BACKCHANNEL
        "okay, got it"          -> BACKCHANNEL
        "yeah but wait"         -> INTERRUPT
        "okay so I think..."    -> INTERRUPT
        "can you repeat that?"  -> INTERRUPT
    """

    t = " ".join(
        text.lower()
        .strip()
        .strip(".,!?;:")
        .split()
    )

    if not t:
        return Intent.BACKCHANNEL

    if t in BACKCHANNELS:
        return Intent.BACKCHANNEL

    words = [
        word.strip(".,!?;:")
        for word in t.split()
        if word.strip(".,!?;:")
    ]

    if not words:
        return Intent.BACKCHANNEL

    # Long speech is substantive.
    if len(words) > C.BACKCHANNEL_MAX_WORDS + 2:
        return Intent.INTERRUPT

    i = 0

    while i < len(words):
        matched = False

        # Prefer longer phrases.
        for span in (3, 2, 1):
            phrase = " ".join(words[i:i + span])

            if phrase in BACKCHANNELS:
                i += span
                matched = True
                break

        if matched:
            continue

        if words[i] in _FILLER:
            i += 1
            continue

        # Any meaningful word indicates an actual attempt to speak.
        return Intent.INTERRUPT

    return Intent.BACKCHANNEL


# ============================================================================
# UTTERANCE
# ============================================================================

@dataclass
class Utterance:
    """One captured segment of user speech."""

    audio: np.ndarray
    started_at: float
    during_speech: bool

    text: str = ""
    intent: Intent = Intent.CONTINUE

    # Generation protects against stale async recognition results.
    generation: int = 0

    done: threading.Event = field(
        default_factory=threading.Event
    )


# ============================================================================
# DUPLEX SESSION
# ============================================================================

class DuplexSession:
    """Runs microphone capture, recognition, partial recognition and playback.

    The design intentionally separates:

        Intent
            Determines conversational turn-taking.

        Concern
            Determines how the interview should respond semantically.

    This class only handles the first responsibility.
    """

    def __init__(
        self,
        mic: audio.Microphone,
    ) -> None:

        self.mic = mic

        self.channel = Channel.IDLE

        self._speaker: audio.Speaker | None = None
        self._streaming = None

        self._spoken_text = ""
        self._backchannel_word = ""

        self._backchannel_speaker: audio.Speaker | None = None

        self._marks: list = []

        self._speech_started = 0.0

        self._bleed = (
            C.BARGE_RMS_MIN / C.BLEED_MARGIN
        )

        # ------------------------------------------------------------------
        # Queues
        # ------------------------------------------------------------------

        self.utterances: queue.Queue[Utterance] = (
            queue.Queue()
        )

        self._to_recognize: queue.Queue[
            Utterance | None
        ] = queue.Queue()

        # Only newest partial matters.
        self._to_partial: queue.Queue = (
            queue.Queue(maxsize=1)
        )

        # ------------------------------------------------------------------
        # Partial recognition
        # ------------------------------------------------------------------

        self.partial_text = ""

        self.on_partial: Callable[[str], None] | None = None

        # ------------------------------------------------------------------
        # Live speech state
        # ------------------------------------------------------------------

        self._in_speech = False

        # ------------------------------------------------------------------
        # Thread lifecycle
        # ------------------------------------------------------------------

        self._running = threading.Event()

        self._lock = threading.RLock()

        # ------------------------------------------------------------------
        # Generation / stale result protection
        # ------------------------------------------------------------------

        self._turn_generation = 0

        self._partial_generation = 0

        self._latest_partial_generation = 0

        # ------------------------------------------------------------------
        # Floor state
        # ------------------------------------------------------------------

        self.floor_taken = threading.Event()

        self._threads: list[threading.Thread] = []

        # ------------------------------------------------------------------
        # Optional callbacks
        # ------------------------------------------------------------------

        self.on_utterance: (
            Callable[[Utterance], None] | None
        ) = None

        self.on_speech: (
            Callable[[str, bool], None] | None
        ) = None

        self.last_timing = None

        # ------------------------------------------------------------------
        # Barge-in manager
        # ------------------------------------------------------------------

        self.barge = interrupt.InterruptionManager(
            log=self._log
        )

        self.barge.on_stop_audio = (
            self._stop_all_audio
        )

        self.barge.on_clear_queue = (
            self._clear_audio_queue
        )

        self.verbose = False

    # ========================================================================
    # STATE / GENERATION
    # ========================================================================

    def _new_generation(self) -> int:
        """Invalidate stale asynchronous recognition work."""

        with self._lock:

            self._turn_generation += 1

            self._partial_generation = 0
            self._latest_partial_generation = 0

            self.partial_text = ""

            return self._turn_generation

    def _current_generation(self) -> int:

        with self._lock:
            return self._turn_generation

    def _set_partial(
        self,
        text: str,
        generation: int,
        partial_id: int,
    ) -> bool:
        """Accept a partial only if it belongs to the current turn."""

        with self._lock:

            if generation != self._turn_generation:
                return False

            if (
                partial_id
                < self._latest_partial_generation
            ):
                return False

            self._latest_partial_generation = (
                partial_id
            )

            self.partial_text = text

            return True

    def _clear_partial(self) -> None:

        with self._lock:

            self.partial_text = ""

            self._latest_partial_generation = 0

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        if self._running.is_set():
            return

        self._running.set()

        for target in (
            self._capture_loop,
            self._recognize_loop,
            self._partial_loop,
        ):

            thread = threading.Thread(
                target=target,
                daemon=True,
            )

            thread.start()

            self._threads.append(thread)

    def stop(self) -> None:

        self._running.clear()

        self._to_recognize.put(None)

        try:
            self._to_partial.put_nowait(None)

        except queue.Full:
            pass

        speaker = self._speaker

        if speaker is not None:
            try:
                speaker.stop()
            except Exception:
                pass

        backchannel = self._backchannel_speaker

        if backchannel is not None:
            try:
                backchannel.stop()
            except Exception:
                pass

        streaming = self._streaming

        if streaming is not None:
            try:
                streaming.stop_now()
            except Exception:
                pass

        for thread in self._threads:

            thread.join(timeout=1.5)

        self._threads.clear()

    def __enter__(self) -> "DuplexSession":

        self.start()

        return self

    def __exit__(self, *exc) -> None:

        self.stop()

    # ========================================================================
    # BLEED / AUDIO GATING
    # ========================================================================

    def _observe_bleed(
        self,
        rms: float,
    ) -> None:
        """Track speaker audio leaking back into the microphone."""

        with self._lock:

            self._bleed = max(
                rms,
                self._bleed * C.BLEED_DECAY,
            )

    def _audible(self) -> bool:
        """Return True if sound is actually leaving the speakers."""

        speaker = self._speaker

        if speaker is not None and speaker.busy:
            return True

        streaming = self._streaming

        inner = (
            getattr(
                streaming,
                "_speaker",
                None,
            )
            if streaming is not None
            else None
        )

        return bool(
            inner is not None
            and inner.busy
        )

    def _gate(self) -> float:
        """Calculate adaptive speech threshold."""

        if self.channel is not Channel.SPEAKING:
            return C.VAD_START_RMS

        adaptive = (
            self._bleed
            * C.BLEED_MARGIN
        )

        return max(
            C.BARGE_RMS_MIN,
            min(
                adaptive,
                C.BARGE_RMS,
            ),
        )

    # ========================================================================
    # CAPTURE
    # ========================================================================

    def _capture_loop(self) -> None:
        """Continuously capture microphone audio and segment utterances."""

        collected: list[np.ndarray] = []

        silence_ms = 0.0
        speech_ms = 0.0

        in_speech = False

        began = 0.0

        overlapped = False

        partial_at = 0.0

        capture_generation = (
            self._current_generation()
        )

        while self._running.is_set():

            frame = self.mic.read(timeout=0.1)

            if frame is None:
                continue

            speaking = (
                self.channel
                is Channel.SPEAKING
            )

            # Ignore only the grace period after actual playback begins.
            if (
                speaking
                and self._speech_started > 0
                and (
                    time.monotonic()
                    - self._speech_started
                )
                * 1000
                < C.BARGE_GRACE_MS
            ):
                continue

            level = audio._rms(frame)

            # --------------------------------------------------------------
            # LEVEL 1 BARGE-IN
            # --------------------------------------------------------------

            if speaking and self.barge is not None:

                gate_now = self._gate()

                if (
                    level < gate_now
                    and self._audible()
                ):
                    self._observe_bleed(level)

                if self.barge.detector.feed(
                    frame,
                    level,
                    gate_now,
                ):
                    self.barge.fire(
                        self.barge.generations.current
                    )

            # --------------------------------------------------------------
            # VAD
            # --------------------------------------------------------------

            if in_speech:

                gate = (
                    C.VAD_KEEP_RMS
                    if not speaking
                    else self._gate() * 0.8
                )

            else:
                gate = self._gate()

            if level >= gate:

                if not in_speech:

                    in_speech = True

                    with self._lock:
                        self._in_speech = True

                    began = time.monotonic()

                    overlapped = speaking

                    capture_generation = (
                        self._current_generation()
                    )

                collected.append(frame)

                speech_ms += C.BLOCK_MS

                silence_ms = 0.0

            elif in_speech:

                collected.append(frame)

                silence_ms += C.BLOCK_MS

                hang = (
                    C.INTERRUPT_HANG_MS
                    if overlapped
                    else C.SILENCE_HANG_MS
                )

                if silence_ms >= hang:

                    self._emit(
                        collected,
                        began,
                        overlapped,
                        speech_ms,
                        capture_generation,
                    )

                    collected = []

                    in_speech = False

                    silence_ms = 0.0

                    speech_ms = 0.0

                    partial_at = 0.0

                    with self._lock:
                        self._in_speech = False

            # --------------------------------------------------------------
            # PARTIAL RECOGNITION
            # --------------------------------------------------------------

            if (
                in_speech
                and speech_ms >= C.PARTIAL_AFTER_MS
                and (
                    speech_ms - partial_at
                    >= C.PARTIAL_EVERY_MS
                )
            ):

                partial_at = speech_ms

                self._emit_partial(
                    list(collected),
                    overlapped,
                    capture_generation,
                )

            # --------------------------------------------------------------
            # FORCED FLUSH
            # --------------------------------------------------------------

            if (
                in_speech
                and speech_ms >= C.DUPLEX_FLUSH_MS
            ):

                self._emit(
                    collected,
                    began,
                    overlapped,
                    speech_ms,
                    capture_generation,
                )

                collected = []

                in_speech = False

                silence_ms = 0.0

                speech_ms = 0.0

                partial_at = 0.0

                # Important: keep live speech state consistent.
                with self._lock:
                    self._in_speech = False

    # ========================================================================
    # PARTIAL RECOGNITION
    # ========================================================================

    def _emit_partial(
        self,
        frames: list[np.ndarray],
        overlapped: bool,
        generation: int,
    ) -> None:
        """Queue the newest snapshot of in-progress speech."""

        if not frames:
            return

        try:
            audio_data = np.concatenate(frames)

        except ValueError:
            return

        with self._lock:

            self._partial_generation += 1

            partial_id = (
                self._partial_generation
            )

        item = (
            audio_data,
            overlapped,
            generation,
            partial_id,
        )

        try:

            self._to_partial.put_nowait(item)

        except queue.Full:

            try:
                self._to_partial.get_nowait()

            except queue.Empty:
                pass

            try:
                self._to_partial.put_nowait(item)

            except queue.Full:
                pass

    def _partial_loop(self) -> None:
        """Recognize active speech while the user is still talking."""

        while self._running.is_set():

            try:

                item = self._to_partial.get(
                    timeout=0.2
                )

            except queue.Empty:
                continue

            if item is None:
                break

            (
                audio_data,
                overlapped,
                generation,
                partial_id,
            ) = item

            # Don't spend resources on obsolete work.
            if (
                generation
                != self._current_generation()
            ):
                continue

            try:

                text = brain.transcribe(
                    audio_data,
                    priority=False,
                )

            except Exception:
                continue

            if not text:
                continue

            # Recognition may have completed after the turn changed.
            if (
                generation
                != self._current_generation()
            ):
                continue

            if (
                overlapped
                and self._is_echo(text)
            ):
                continue

            if overlapped:
                text = self.strip_echo(text)

            if not text:
                continue

            accepted = self._set_partial(
                text=text,
                generation=generation,
                partial_id=partial_id,
            )

            if not accepted:
                continue

            callback = self.on_partial

            if callback is not None:

                try:
                    callback(text)

                except Exception:
                    pass

    # ========================================================================
    # EMIT UTTERANCE
    # ========================================================================

    def _emit(
        self,
        frames,
        began,
        overlapped,
        speech_ms,
        generation: int,
    ) -> None:

        if (
            speech_ms < C.MIN_SPEECH_MS
            or not frames
        ):
            return

        try:

            audio_data = np.concatenate(frames)

        except ValueError:
            return

        utterance = Utterance(
            audio=audio_data,
            started_at=began,
            during_speech=overlapped,
            generation=generation,
        )

        self._to_recognize.put(
            utterance
        )

        self.utterances.put(
            utterance
        )

    # ========================================================================
    # FINAL RECOGNITION
    # ========================================================================

    def _recognize_loop(self) -> None:
        """Recognize completed utterances."""

        while self._running.is_set():

            utterance = (
                self._to_recognize.get()
            )

            if utterance is None:
                break

            # Ignore stale work.
            if (
                utterance.generation
                != self._current_generation()
            ):
                utterance.done.set()
                continue

            try:

                utterance.text = (
                    brain.transcribe(
                        utterance.audio
                    )
                )

            except Exception:

                utterance.text = ""

            # Turn may have changed while transcription was running.
            if (
                utterance.generation
                != self._current_generation()
            ):

                utterance.text = ""

                utterance.intent = (
                    Intent.BACKCHANNEL
                )

                utterance.done.set()

                continue

            # --------------------------------------------------------------
            # ECHO SUPPRESSION
            # --------------------------------------------------------------

            if (
                utterance.text
                and utterance.during_speech
            ):

                if self._is_echo(
                    utterance.text
                ):

                    utterance.text = ""

                    utterance.intent = (
                        Intent.BACKCHANNEL
                    )

                    utterance.done.set()

                    continue

                utterance.text = (
                    self.strip_echo(
                        utterance.text
                    )
                )

            # --------------------------------------------------------------
            # FILTER VERY SHORT NOISE
            # --------------------------------------------------------------

            if (
                utterance.text
                and len(
                    utterance.text.split()
                )
                < C.MIN_ANSWER_WORDS
                and not utterance.during_speech
            ):
                utterance.text = ""

            utterance.intent = (
                classify(utterance.text)
                if utterance.text
                else Intent.BACKCHANNEL
            )

            utterance.done.set()

            # --------------------------------------------------------------
            # CALLBACK
            # --------------------------------------------------------------

            callback = self.on_utterance

            if callback is not None:

                try:
                    callback(utterance)

                except Exception:
                    pass

            # --------------------------------------------------------------
            # LEVEL 2 BARGE VERDICT
            # --------------------------------------------------------------

            if (
                utterance.during_speech
                and self.barge.current
                is not None
            ):

                real = (
                    utterance.intent
                    is Intent.INTERRUPT
                    and bool(
                        utterance.text
                    )
                )

                verdict = (
                    self.barge.confirm(
                        utterance.text,
                        real,
                    )
                )

                if (
                    verdict
                    is interrupt.Verdict.CONFIRMED
                ):
                    self.yield_floor()

                else:
                    self._resume_after_false_alarm()

            elif (
                utterance.during_speech
                and utterance.intent
                is Intent.INTERRUPT
                and self.channel
                is Channel.SPEAKING
            ):

                self.yield_floor()

    # ========================================================================
    # ECHO DETECTION
    # ========================================================================

    @staticmethod
    def _words(text: str) -> set[str]:

        return {
            word.strip(
                ".,!?;:'\""
            ).lower()

            for word in text.split()

            if len(
                word.strip(
                    ".,!?;:'\""
                )
            )
            > 3
        }

    def _is_echo(
        self,
        heard: str,
    ) -> bool:
        """Return True only when heard speech is almost entirely our own."""

        said = " ".join(
            (
                self._spoken_text,
                self._backchannel_word,
            )
        ).strip()

        if not said:
            return False

        heard_words = self._words(
            heard
        )

        if not heard_words:
            return False

        our_words = self._words(
            said
        )

        novel = (
            heard_words
            - our_words
        )

        if (
            len(novel)
            >= C.ECHO_MIN_NOVEL_WORDS
        ):
            return False

        overlap = (
            len(
                heard_words
                & our_words
            )
            / len(heard_words)
        )

        return (
            overlap
            >= C.ECHO_OVERLAP
        )

    def strip_echo(
        self,
        heard: str,
    ) -> str:
        """Remove known agent words from mixed recognition."""

        said = " ".join(
            (
                self._spoken_text,
                self._backchannel_word,
            )
        ).strip()

        our_words = self._words(
            said
        )

        if not our_words:
            return heard

        kept = [

            word

            for word in heard.split()

            if word.strip(
                ".,!?;:'\""
            ).lower()
            not in our_words

        ]

        cleaned = " ".join(
            kept
        ).strip()

        return cleaned or heard

    # ========================================================================
    # UI / LOGGING
    # ========================================================================

    def _announce(
        self,
        text: str,
        done: bool = False,
    ) -> None:

        callback = self.on_speech

        if callback is None:
            return

        try:

            callback(
                text,
                done,
            )

        except Exception:
            pass

    def _log(
        self,
        message: str,
    ) -> None:

        if self.verbose:
            print(
                f"    {message}"
            )

    # ========================================================================
    # BACKCHANNEL
    # ========================================================================

    def speak_backchannel(
        self,
        word: str,
    ) -> None:
        """Speak a short acknowledgement without changing turn ownership."""

        if (
            self.channel
            is Channel.SPEAKING
        ):
            return

        # If the candidate is actively speaking, the backchannel is okay.
        # If the interviewer has started another real turn, do nothing.

        try:

            data, sample_rate, _ = (
                audio.synthesize(
                    word,
                    rate=_BACKCHANNEL_RATE,
                )
            )

        except Exception:
            return

        data = (
            data
            * _BACKCHANNEL_VOLUME
        )

        self._backchannel_word = word

        speaker = audio.Speaker()

        self._backchannel_speaker = (
            speaker
        )

        try:

            speaker.start(
                data,
                sample_rate,
            )

            speaker.wait()

        finally:

            self._backchannel_speaker = None

            self._backchannel_word = ""

    # ========================================================================
    # AUDIO CONTROL
    # ========================================================================

    def _stop_all_audio(
        self,
    ) -> None:
        """Immediately stop all active interviewer audio."""

        speaker = self._speaker

        if speaker is not None:

            try:
                speaker.stop()

            except Exception:
                pass

        streaming = self._streaming

        if streaming is not None:

            try:
                streaming.stop_now()

            except Exception:
                pass

    def _clear_audio_queue(
        self,
    ) -> None:
        """Cancel queued streaming audio."""

        streaming = self._streaming

        if streaming is not None:

            try:
                streaming.stop.set()

            except Exception:
                pass

    def peek(self) -> str:
        """Return the latest partial transcript."""

        with self._lock:

            return self.partial_text

    # ========================================================================
    # INTERRUPTION VERDICT
    # ========================================================================

    def _await_verdict(
        self,
        timeout: float = 3.0,
    ) -> "interrupt.Verdict":

        deadline = (
            time.monotonic()
            + timeout
        )

        seen = self.peek()

        while (
            time.monotonic()
            < deadline
        ):

            record = (
                self.barge.current
            )

            if (
                record is not None
                and record.verdict
                is not interrupt.Verdict.PENDING
            ):

                return record.verdict

            live = self.peek()

            if (
                live
                and live != seen
                and classify(live)
                is Intent.INTERRUPT
            ):

                return (
                    interrupt.Verdict.CONFIRMED
                )

            time.sleep(0.02)

        # Fail safe: if someone triggered a barge-in and recognition is slow,
        # assume they intended to speak.
        return (
            interrupt.Verdict.CONFIRMED
        )

    def _resume_after_false_alarm(
        self,
    ) -> None:

        self.floor_taken.clear()

        with self._lock:

            if (
                self.channel
                is Channel.YIELDED
            ):

                self.channel = Channel.IDLE

    # ========================================================================
    # STREAMING PLAYBACK
    # ========================================================================

    def say_stream(
        self,
        messages: list,
        max_tokens: int = 90,
    ) -> tuple[bool, str]:
        """Generate and speak concurrently while still listening."""

        from . import streaming

        generation = self.barge.arm()

        with self._lock:

            self.channel = (
                Channel.SPEAKING
            )

            # Grace begins only once sound starts.
            self._speech_started = 0.0

            self.floor_taken.clear()

            self._spoken_text = ""

        voice = (
            streaming.StreamingVoice(
                on_stop=self.barge.interrupted,
                generation=generation,
            )
        )

        def on_chunk(
            chunk: str,
        ) -> None:

            with self._lock:

                self._spoken_text = (
                    self._spoken_text
                    + " "
                    + chunk
                ).strip()

        voice.on_chunk = on_chunk

        started = threading.Event()

        def on_playing(
            text: str,
        ) -> None:

            if not started.is_set():

                started.set()

                with self._lock:

                    self._speech_started = (
                        time.monotonic()
                    )

                    self.barge.detector.arm()

            self._announce(
                text
            )

        voice.on_playing = on_playing

        self._streaming = voice

        spoken = (
            voice.speak_stream(
                messages,
                max_tokens=max_tokens,
            )
        )

        self.last_timing = (
            voice.timing
        )

        interrupted = (
            self.barge.interrupted.is_set()
            or self.floor_taken.is_set()
        )

        with self._lock:

            self.channel = (
                Channel.YIELDED
                if interrupted
                else Channel.IDLE
            )

            self._streaming = None

        self._announce(
            spoken,
            done=True,
        )

        return (
            interrupted,
            spoken,
        )

    # ========================================================================
    # NORMAL PLAYBACK
    # ========================================================================

    def say(
        self,
        text: str,
    ) -> tuple[bool, str]:
        """Speak while capture and recognition remain active."""

        if not text:
            return False, ""

        self.barge.arm()

        with self._lock:

            self.channel = (
                Channel.SPEAKING
            )

            # Important: no grace while TTS is still synthesizing.
            self._speech_started = 0.0

            self.floor_taken.clear()

            self._spoken_text = ""

        # --------------------------------------------------------------
        # TTS SYNTHESIS
        # --------------------------------------------------------------

        data, sample_rate, marks = (
            audio.synthesize(
                text
            )
        )

        # Candidate may have spoken while TTS was being generated.
        if (
            self.barge.interrupted.is_set()
            or self.floor_taken.is_set()
        ):

            with self._lock:

                self.channel = (
                    Channel.YIELDED
                )

                self._speaker = None

            return True, ""

        # --------------------------------------------------------------
        # START PLAYBACK
        # --------------------------------------------------------------

        with self._lock:

            self._marks = marks

            self._spoken_text = text

            speaker = (
                audio.Speaker()
            )

            self._speaker = speaker

            # Grace begins when sound actually begins.
            self._speech_started = (
                time.monotonic()
            )

            speaker.start(
                data,
                sample_rate,
            )

        self._announce(
            text
        )

        played = 0.0

        while True:

            while (
                speaker.busy
                and not (
                    self.floor_taken.is_set()
                    or self.barge.interrupted.is_set()
                )
            ):

                time.sleep(0.005)

            if not (
                self.floor_taken.is_set()
                or self.barge.interrupted.is_set()
            ):
                break

            played += (
                speaker.elapsed
            )

            verdict = (
                self._await_verdict()
            )

            if (
                verdict
                is not interrupt.Verdict.RETRACTED
            ):
                break

            remainder = data[
                int(
                    played
                    * sample_rate
                ):
            ]

            if (
                len(remainder)
                < sample_rate * 0.15
            ):
                break

            self.barge.interrupted.clear()

            self.floor_taken.clear()

            with self._lock:

                self._speech_started = (
                    time.monotonic()
                )

                self.barge.detector.arm()

                speaker = (
                    audio.Speaker()
                )

                self._speaker = speaker

                speaker.start(
                    remainder,
                    sample_rate,
                )

        interrupted = (
            self.floor_taken.is_set()
            or self.barge.interrupted.is_set()
        )

        if interrupted:

            heard = (
                audio._spoken_prefix(
                    text,
                    marks,
                    speaker.elapsed,
                )
            )

        else:

            speaker.wait()

            heard = text

        with self._lock:

            self.channel = (
                Channel.YIELDED
                if interrupted
                else Channel.IDLE
            )

            self._speaker = None

            self._spoken_text = ""

        self._announce(
            heard,
            done=True,
        )

        return (
            interrupted,
            heard,
        )

    # ========================================================================
    # YIELD FLOOR
    # ========================================================================

    def yield_floor(
        self,
    ) -> None:
        """Immediately stop speaking and give the user the conversational floor."""

        with self._lock:

            self.floor_taken.set()

            speaker = self._speaker

            if speaker is not None:

                try:
                    speaker.stop()

                except Exception:
                    pass

            self.channel = (
                Channel.YIELDED
            )

    # ========================================================================
    # CONSUME USER SPEECH
    # ========================================================================

    def next_utterance(
        self,
        timeout: float = 20.0,
    ) -> Utterance | None:
        """Wait for the next valid transcribed utterance."""

        deadline = (
            time.monotonic()
            + timeout
        )

        current_generation = (
            self._current_generation()
        )

        while (
            time.monotonic()
            < deadline
        ):

            try:

                utterance = (
                    self.utterances.get(
                        timeout=0.2
                    )
                )

            except queue.Empty:
                continue

            # Ignore stale queued utterances.
            if (
                utterance.generation
                != current_generation
            ):
                continue

            utterance.done.wait(
                timeout=15.0
            )

            if (
                utterance.text
                and utterance.generation
                == self._current_generation()
            ):

                return utterance

        return None

    def collect_interjection(
        self,
        timeout: float = 10.0,
    ) -> str:
        """Return the first substantive interjection."""

        utterance = (
            self.next_utterance(
                timeout=timeout
            )
        )

        if utterance is None:
            return ""

        self._clear_partial()

        return (
            utterance.text.strip()
        )

    def collect_answer(
        self,
        timeout: float = 20.0,
        settle: float = 1.2,
    ) -> str:
        """Collect a complete answer across multiple speech segments."""

        parts: list[str] = []

        first = (
            self.next_utterance(
                timeout=timeout
            )
        )

        if first is None:
            return ""

        if (
            first.intent
            is not Intent.BACKCHANNEL
        ):
            parts.append(
                first.text
            )

        nodded = False

        while True:

            nxt = (
                self._next_utterance_after_silence(
                    settle
                )
            )

            if nxt is None:
                break

            if (
                nxt.intent
                is not Intent.BACKCHANNEL
            ):

                parts.append(
                    nxt.text
                )

                if not nodded:

                    # Avoid starting a backchannel if Jerry has already
                    # entered a real speaking state.
                    if (
                        self.channel
                        is not Channel.SPEAKING
                    ):

                        self.speak_backchannel(
                            random.choice(
                                SPOKEN_BACKCHANNELS
                            )
                        )

                    nodded = True

        self._clear_partial()

        return " ".join(
            parts
        ).strip()

    def _next_utterance_after_silence(
        self,
        settle: float,
    ) -> Utterance | None:
        """Wait through genuine silence without timing out active speech."""

        poll = 0.2

        quiet_for = 0.0

        deadline = (
            time.monotonic()
            + max(settle, poll) * 4
        )

        while quiet_for < settle:

            if (
                time.monotonic()
                > deadline
            ):
                return None

            try:

                utterance = (
                    self.utterances.get(
                        timeout=poll
                    )
                )

            except queue.Empty:

                with self._lock:
                    in_speech = (
                        self._in_speech
                    )

                if in_speech:
                    quiet_for = 0.0

                else:
                    quiet_for += poll

                continue

            utterance.done.wait(
                timeout=15.0
            )

            if (
                utterance.text
                and utterance.generation
                == self._current_generation()
            ):
                return utterance

            quiet_for = 0.0

        return None

    # ========================================================================
    # RESET / DRAIN
    # ========================================================================

    def drain_pending(
        self,
    ) -> None:
        """Invalidate the current async work and clear pending queues.

        This prevents late Whisper/partial results from a previous interview
        turn being interpreted as part of the next question.
        """

        self._new_generation()

        queues = (
            self.utterances,
            self._to_recognize,
            self._to_partial,
        )

        for q in queues:

            while True:

                try:
                    q.get_nowait()

                except queue.Empty:
                    break

        self._clear_partial()