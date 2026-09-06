"""Streaming LLM -> chunker -> TTS pipeline.

The non-streaming path waits for the whole reply, synthesises it, and only then
makes a sound. Here the reply is cut into speech-sized chunks as tokens arrive,
so synthesis of chunk N overlaps generation of chunk N+1 and the user hears the
first words far sooner.

Chunking is on sentence and clause boundaries, never on raw token counts: TTS
prosody depends on getting whole phrases, so "I think" / "it overfits" spoken as
two fragments sounds visibly wrong.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import audio, brain
from . import config as C

# Strong boundaries end a spoken chunk outright; weak ones only count once the
# buffer is long enough to be worth speaking on its own.
_STRONG = re.compile(r"[.!?]['\")\]]*\s")
_WEAK = re.compile(r"[,;:]['\")\]]*\s")


def chunk_stream(tokens: Iterator[str]) -> Iterator[str]:
    """Group a token stream into speech-friendly chunks."""
    buf = ""
    for tok in tokens:
        if not tok:
            continue
        buf += tok

        # Emit as soon as a sentence completes.
        m = None
        for m in _STRONG.finditer(buf):
            pass
        if m and len(buf[: m.end()].strip()) >= C.CHUNK_MIN_CHARS:
            out, buf = buf[: m.end()].strip(), buf[m.end():]
            if out:
                yield out
            continue

        # Otherwise fall back to a clause break once we have enough words.
        if len(buf) >= C.CHUNK_SOFT_CHARS:
            m2 = None
            for m2 in _WEAK.finditer(buf):
                pass
            if m2 and len(buf[: m2.end()].strip()) >= C.CHUNK_MIN_CHARS:
                out, buf = buf[: m2.end()].strip(), buf[m2.end():]
                if out:
                    yield out
                continue

        # Hard cap so a run-on sentence still starts playing.
        if len(buf) >= C.CHUNK_MAX_CHARS:
            cut = buf.rfind(" ", 0, C.CHUNK_MAX_CHARS)
            if cut > C.CHUNK_MIN_CHARS:
                out, buf = buf[:cut].strip(), buf[cut:]
                if out:
                    yield out

    if buf.strip():
        yield buf.strip()


def stream_llm(messages: list[dict], max_tokens: int = 90) -> Iterator[str]:
    """Yield content tokens from Groq as they arrive."""
    try:
        stream = brain.client().chat.completions.create(
            model=C.LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens + C.REASONING_HEADROOM,
            temperature=0.7,
            reasoning_effort=C.REASONING_EFFORT,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta else None
            if piece:
                yield piece
    except Exception as exc:
        print(f"  [llm stream error: {exc}]")


@dataclass
class Timing:
    """Latency breakdown for one spoken response."""

    started: float = field(default_factory=time.monotonic)
    first_token: float | None = None
    first_chunk: float | None = None
    first_audio: float | None = None
    finished: float | None = None

    def ms(self, mark: float | None) -> int | None:
        return None if mark is None else int((mark - self.started) * 1000)

    def summary(self) -> str:
        return (
            f"first token {self.ms(self.first_token)}ms | "
            f"first chunk {self.ms(self.first_chunk)}ms | "
            f"first audio {self.ms(self.first_audio)}ms | "
            f"total {self.ms(self.finished)}ms"
        )


class StreamingVoice:
    """Speaks an LLM response as it is generated.

    Synthesis runs one chunk ahead of playback, so the gap between chunks is
    hidden and the whole reply plays as one continuous utterance.
    """

    def __init__(
        self,
        on_stop: threading.Event | None = None,
        generation: int | None = None,
    ) -> None:
        # Externally-owned event; set it to cut the response short.
        self.stop = on_stop or threading.Event()
        # Work tagged with a different generation is stale and never played.
        self._generation = generation
        self._speaker: audio.Speaker | None = None
        self.spoken: list[str] = []
        self.timing = Timing()
        # Called with each chunk as it is queued, so callers can track what
        # is currently being said (echo suppression relies on this).
        self.on_chunk = None
        # Called as each chunk starts playing, with everything voiced so far.
        # Synthesis runs up to SYNTH_LOOKAHEAD chunks ahead of the speakers,
        # so `on_chunk` is the wrong signal for a caption: it would show a
        # sentence the listener has not heard yet.
        self.on_playing = None

    def _synth_worker(
        self, chunks: queue.Queue, out: queue.Queue
    ) -> None:
        while not self.stop.is_set():
            item = chunks.get()
            if item is None:
                out.put(None)
                return
            try:
                data, sr, _ = audio.synthesize(item)
                # Synthesis takes ~1s; an interruption during it means this
                # audio must be discarded rather than queued for playback.
                if self.stop.is_set():
                    continue
                if self.timing.first_audio is None:
                    self.timing.first_audio = time.monotonic()
                out.put((self._generation, item, data, sr))
            except Exception:
                continue

    def speak_stream(self, messages: list[dict], max_tokens: int = 90) -> str:
        """Generate and speak, returning the text actually voiced."""
        text_chunks: queue.Queue = queue.Queue()
        ready: queue.Queue = queue.Queue(maxsize=C.SYNTH_LOOKAHEAD)

        worker = threading.Thread(
            target=self._synth_worker, args=(text_chunks, ready), daemon=True
        )
        worker.start()

        def produce() -> None:
            tokens = stream_llm(messages, max_tokens)

            def marked() -> Iterator[str]:
                for t in tokens:
                    if self.stop.is_set():
                        return
                    if self.timing.first_token is None:
                        self.timing.first_token = time.monotonic()
                    yield t

            for chunk in chunk_stream(marked()):
                if self.stop.is_set():
                    break
                if self.timing.first_chunk is None:
                    self.timing.first_chunk = time.monotonic()
                cleaned = brain._clean(chunk)
                if self.on_chunk is not None:
                    try:
                        self.on_chunk(cleaned)
                    except Exception:
                        pass
                text_chunks.put(cleaned)
            text_chunks.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()

        speaker = audio.Speaker()
        self._speaker = speaker
        while not self.stop.is_set():
            # Never block indefinitely: if the stop event arrives while this
            # is waiting on an empty queue, an unbounded get() would hold the
            # interruption until synthesis happened to deliver something.
            try:
                item = ready.get(timeout=0.05)
            except queue.Empty:
                if not producer.is_alive() and text_chunks.empty():
                    break
                continue
            if item is None:
                break

            # Drop anything synthesised for a generation we have abandoned.
            gen, chunk, data, sr = item
            if self._generation is not None and gen != self._generation:
                continue

            speaker.start(data, sr)
            if self.on_playing is not None:
                try:
                    self.on_playing(" ".join(self.spoken + [chunk]))
                except Exception:
                    pass
            chunk_duration_s = len(data) / float(sr or 1)
            while speaker.busy and not self.stop.is_set():
                time.sleep(0.005)
            # Whether this chunk counts as spoken must come from how much of
            # it actually played (speaker.elapsed), not from a race between
            # two threads: stop_now() (called from another thread on
            # barge-in) stops the speaker directly, which makes speaker.busy
            # false as a side effect - indistinguishable, by timing alone,
            # from this chunk having finished on its own at the same moment.
            # That used to either drop a chunk the candidate heard in full,
            # or keep one that was genuinely cut off, depending on which of
            # the two threads happened to act first.
            played_fully = speaker.elapsed >= chunk_duration_s - 0.05
            if self.stop.is_set():
                speaker.stop()
            if not played_fully:
                break
            self.spoken.append(chunk)

        self.timing.finished = time.monotonic()
        if self.stop.is_set():
            self._drain(ready)          # pending audio must never play later
            text_chunks.put(None)       # release the synth worker
        producer.join(timeout=0.5)
        self._speaker = None
        return " ".join(self.spoken)

    def stop_now(self) -> None:
        """Cut playback this instant, from another thread."""
        self.stop.set()
        sp = self._speaker
        if sp is not None:
            sp.stop()

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
