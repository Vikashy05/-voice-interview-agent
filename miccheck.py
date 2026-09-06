"""Thirty-second check that the agent can actually hear you interrupt.

Run this before an interview. It measures your room, your voice, and the
speaker bleed, then writes the right thresholds into config.py.

    python miccheck.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from interview_agent import audio
from interview_agent import config as C


def rms_series(x, block):
    return np.array(
        [np.sqrt(np.mean(x[i:i + block] ** 2))
         for i in range(0, max(0, len(x) - block), block)]
    )


BLOCK = int(C.SAMPLE_RATE * C.BLOCK_MS / 1000)

print("=" * 64)
print("  MIC CHECK")
print("=" * 64)
mic_name = sd.query_devices(kind="input")["name"].strip()
print(f"  mic: {mic_name}")
print()

# 1. room noise
print("  [1/3] Silence please, 3 seconds...", flush=True)
d = sd.rec(int(3 * C.SAMPLE_RATE), samplerate=C.SAMPLE_RATE,
           channels=1, dtype="float32")
sd.wait()
noise = rms_series(d[:, 0], BLOCK)
noise_p95 = float(np.percentile(noise, 95))
print(f"        room noise: {noise_p95:.5f}")
print()

# 2. the user's voice
print("  [2/3] TALK NORMALLY for 5 seconds - say anything...", flush=True)
d = sd.rec(int(5 * C.SAMPLE_RATE), samplerate=C.SAMPLE_RATE,
           channels=1, dtype="float32")
sd.wait()
v = rms_series(d[:, 0], BLOCK)
voiced = v[v > noise_p95 * 2]
if len(voiced) < 5:
    print()
    print("  Could not hear you at all. Check that the right microphone is")
    print("  selected in Windows sound settings, then run this again.")
    sys.exit(1)
voice_med = float(np.median(voiced))
voice_p25 = float(np.percentile(voiced, 25))
print(f"        your voice: median {voice_med:.5f}, quieter quarter {voice_p25:.5f}")
print()

# 3. speaker bleed
print("  [3/3] Measuring speaker bleed - stay quiet...", flush=True)
data, sr, _ = audio.synthesize(
    "I am measuring how much of my own voice reaches your microphone. "
    "Please stay quiet for a moment while I finish talking."
)
cap = sd.rec(int(len(data) / sr * C.SAMPLE_RATE) + C.SAMPLE_RATE,
             samplerate=C.SAMPLE_RATE, channels=1, dtype="float32")
time.sleep(0.15)
sp = audio.Speaker()
sp.start(data, sr)
sp.wait()
sd.wait()
b = rms_series(cap[:, 0], BLOCK)
bleed_max = float(np.percentile(b, 99))
print(f"        speaker bleed: {bleed_max:.5f}")
print()

# --- decide -------------------------------------------------------------
print("=" * 64)
print("  RESULT")
print("=" * 64)

headroom = voice_p25 / bleed_max if bleed_max > 0 else 99
print(f"  your quiet speech is {headroom:.1f}x the speaker bleed")

if headroom < 1.3:
    print()
    print("  PROBLEM: your voice and the speakers are too close in level.")
    print("  The agent cannot reliably tell you apart from itself.")
    print()
    print("  Fix: use headphones, or turn the speaker volume down, then")
    print("  run this again. This is the single biggest quality factor.")
    sys.exit(1)

barge = max(bleed_max * 1.5, voice_p25 * 0.5)
barge_min = max(bleed_max * 1.2, 0.004)
start = max(noise_p95 * 3, voice_p25 * 0.35)
keep = start * 0.5

# Ceilings. If the microphone runs hot - loud speech, high gain, or breath
# straight into it - these formulas produce gates so high that ordinary
# speech never reaches them, and the agent goes deaf. Better to be slightly
# too sensitive than to write a threshold nothing can cross.
CEIL_START, CEIL_BARGE = 0.045, 0.060
if start > CEIL_START or barge > CEIL_BARGE:
    print()
    print("  Note: your microphone is running hot (very high levels).")
    print("  Capping the thresholds so normal speech still registers.")
    start = min(start, CEIL_START)
    keep = min(keep, start * 0.5)
    barge = min(barge, CEIL_BARGE)
    barge_min = min(barge_min, barge * 0.7)

# Floors, so a very quiet mic does not trigger on room hiss.
start = max(start, 0.004)
keep = max(keep, 0.002)
barge = max(barge, 0.008)
barge_min = max(barge_min, 0.004)

print()
print("  Writing thresholds into interview_agent/config.py:")
print(f"    VAD_START_RMS = {start:.4f}")
print(f"    VAD_KEEP_RMS  = {keep:.4f}")
print(f"    BARGE_RMS     = {barge:.4f}")
print(f"    BARGE_RMS_MIN = {barge_min:.4f}")

p = Path(__file__).parent / "interview_agent" / "config.py"
src = p.read_text(encoding="utf-8")
for name, val in [
    ("VAD_START_RMS", start), ("VAD_KEEP_RMS", keep),
    ("BARGE_RMS", barge), ("BARGE_RMS_MIN", barge_min),
]:
    src = re.sub(rf"^{name} = [0-9.]+", f"{name} = {val:.4f}",
                 src, count=1, flags=re.MULTILINE)
p.write_text(src, encoding="utf-8")

print()
print("  Done. Now run:  python -m interview_agent.main")
print()
