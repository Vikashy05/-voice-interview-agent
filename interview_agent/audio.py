"""Audio engine: speaking with barge-in, and listening with VAD.

The key behaviour is that playback happens on a worker thread in small chunks
while the microphone is monitored on the caller's thread. As soon as the user
starts talking over the agent, playback is cut and the words that were never
spoken are reported back so the transcript stays honest.
"""
from __future__ import annotations

import asyncio
import io
import queue
import threading
import time

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config as C


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


# --------------------------------------------------------------------------
# Text to speech
# --------------------------------------------------------------------------
async def _synth(
    text: str, rate: str | None = None, pitch: str | None = None
) -> tuple[np.ndarray, int, list]:
    """Render text to a float32 mono waveform via edge-tts."""
    com = edge_tts.Communicate(
        text, C.TTS_VOICE,
        rate=rate or C.TTS_RATE, pitch=pitch or C.TTS_PITCH,
        boundary="WordBoundary",
    )
    buf = io.BytesIO()
    # word_boundary events let us estimate how far playback actually got
    marks: list[tuple[float, str]] = []
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            marks.append((chunk["offset"] / 1e7, chunk["text"]))
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr, marks


def synthesize(
    text: str, rate: str | None = None, pitch: str | None = None
) -> tuple[np.ndarray, int, list]:
    return asyncio.run(_synth(text, rate=rate, pitch=pitch))


class Speaker:
    """Plays a waveform in small chunks so it can be stopped almost instantly."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._played_frames = 0
        self._sr = C.SAMPLE_RATE
        self._stream: sd.OutputStream | None = None

    def start(self, data: np.ndarray, sr: int) -> None:
        self._stop.clear()
        self._played_frames = 0
        self._sr = sr

        def run() -> None:
            try:
                stream = sd.OutputStream(
                    samplerate=sr, channels=1, dtype="float32",
                    blocksize=C.PLAYBACK_BLOCK,
                )
                self._stream = stream
                stream.start()
                for i in range(0, len(data), C.PLAYBACK_BLOCK):
                    if self._stop.is_set():
                        break
                    block = data[i : i + C.PLAYBACK_BLOCK]
                    stream.write(np.ascontiguousarray(block, dtype="float32"))
                    self._played_frames = i + len(block)
                if not self._stop.is_set():
                    stream.stop()      # let the tail drain naturally
                stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Cut playback now. `abort` discards buffered audio instead of
        draining it, which is the difference between stopping in tens of
        milliseconds and stopping in hundreds."""
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        return self._played_frames / float(self._sr or 1)

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()


class Microphone:
    """Continuous mic capture pushing fixed-size frames onto a queue."""

    def __init__(self) -> None:
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self.block = int(C.SAMPLE_RATE * C.BLOCK_MS / 1000)

    def __enter__(self) -> Microphone:
        def cb(indata, _frames, _t, status):
            self.frames.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=C.SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=self.block, callback=cb,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def drain(self) -> None:
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None


def speak_interruptible(
    mic: Microphone, text: str, allow_barge: bool = True
) -> tuple[bool, str, np.ndarray | None]:
    """Speak `text`, watching the mic for an interruption.

    Returns (interrupted, spoken_text, captured_audio). `captured_audio` holds
    the user's opening words when they barged in, so no speech is lost.
    """
    data, sr, marks = synthesize(text)
    speaker = Speaker()
    mic.drain()
    speaker.start(data, sr)

    started = time.monotonic()
    loud = 0
    captured: list[np.ndarray] = []

    while speaker.busy:
        frame = mic.read(timeout=0.1)
        if frame is None:
            continue
        if not allow_barge:
            continue
        if (time.monotonic() - started) * 1000 < C.BARGE_GRACE_MS:
            continue

        if _rms(frame) >= C.BARGE_RMS:
            loud += 1
            captured.append(frame)
            if loud >= C.BARGE_FRAMES:
                speaker.stop()
                spoken = _spoken_prefix(text, marks, speaker.elapsed)
                return True, spoken, np.concatenate(captured)
        else:
            loud = 0
            captured.clear()

    speaker.wait()
    return False, text, None


def _spoken_prefix(text: str, marks: list, elapsed: float) -> str:
    """Best-effort reconstruction of how much of `text` was actually heard."""
    if not marks:
        return text
    said = [w for off, w in marks if off <= elapsed]
    if not said:
        return ""
    if len(said) >= len(marks):
        return text
    return " ".join(said) + "..."


# --------------------------------------------------------------------------
# Listening
# --------------------------------------------------------------------------
def listen(
    mic: Microphone, prefix: np.ndarray | None = None, timeout: float = 20.0
) -> np.ndarray | None:
    """Capture one utterance using energy VAD with a silence hangover."""
    collected: list[np.ndarray] = []
    speech_ms = 0.0
    silence_ms = 0.0
    in_speech = False

    if prefix is not None and len(prefix):
        collected.append(prefix)
        speech_ms = len(prefix) / C.SAMPLE_RATE * 1000
        in_speech = True
    else:
        mic.drain()

    waited = 0.0
    start = time.monotonic()

    while True:
        frame = mic.read(timeout=0.2)
        if frame is None:
            if not in_speech:
                waited += 0.2
                if waited >= timeout:
                    return None
            continue

        level = _rms(frame)
        gate = C.VAD_KEEP_RMS if in_speech else C.VAD_START_RMS

        if level >= gate:
            in_speech = True
            collected.append(frame)
            speech_ms += C.BLOCK_MS
            silence_ms = 0.0
        elif in_speech:
            collected.append(frame)
            silence_ms += C.BLOCK_MS
            if silence_ms >= C.SILENCE_HANG_MS:
                break
        else:
            waited += C.BLOCK_MS / 1000
            if waited >= timeout:
                return None

        if time.monotonic() - start > C.MAX_UTTERANCE_S:
            break

    if speech_ms < C.MIN_SPEECH_MS or not collected:
        return None
    return np.concatenate(collected)
