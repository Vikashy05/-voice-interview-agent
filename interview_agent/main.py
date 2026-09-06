"""Run the voice interview."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from . import audio, brain, duplex, graph, graph_duplex
from . import config as C
from . import recorder as rec
from . import session as sess

# Windows consoles default to cp1252, which raises on characters the model or
# Whisper may legitimately produce. Never let printing abort an interview.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _banner() -> None:
    print("\n" + "=" * 62)
    print("  AI VOICE INTERVIEW".center(62))
    print("=" * 62)
    print("  Speak naturally. Interrupt any time - Jerry listens while she talks.")
    print("  Ctrl+C to end early.\n")


def _check_devices() -> bool:
    import sounddevice as sd

    try:
        sd.check_input_settings(samplerate=C.SAMPLE_RATE, channels=1)
        sd.check_output_settings(channels=1)
    except Exception as exc:
        print(f"Audio device problem: {exc}\n")
        return False
    mic_name = sd.query_devices(kind="input")["name"].strip()
    out_name = sd.query_devices(kind="output")["name"].strip()
    print(f"  mic: {mic_name}")
    print(f"  out: {out_name}")
    return True


def _show_backchannel(utt) -> None:
    """Surface nods heard mid-sentence without interrupting the flow."""
    if utt.during_speech and utt.intent is duplex.Intent.BACKCHANNEL and utt.text:
        print(f"    (you: {utt.text.strip()})")


_partial_state = {"last": ""}


def _show_partial(text: str) -> None:
    """Echo the running transcript so it is visible that we are following."""
    if not text or text == _partial_state["last"]:
        return
    _partial_state["last"] = text
    line = text if len(text) <= 68 else "..." + text[-65:]
    # Overwrite in place: this is a live guess, not a log entry.
    print(chr(13) + "    ~ " + f"{line:<70}", end="", flush=True)


def _report_metrics(voice, session=None) -> None:
    """Print the latency breakdown gathered during the session."""
    summary = voice.summary()
    if session is not None and getattr(session, "barge", None) is not None:
        summary.update(
            {f"barge_{k}": v for k, v in session.barge.stats().items()}
        )
    print("\n" + "=" * 62)
    print("  LATENCY / SESSION METRICS".center(62))
    print("=" * 62)
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:<26} {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description="LangGraph voice interview agent")
    ap.add_argument("--no-notes", action="store_true", help="skip debrief notes")
    ap.add_argument("--save", default=None, help="write transcript to this path")
    ap.add_argument("--outdir", default=None,
                    help="folder for saved answers and transcript")
    ap.add_argument("--debug-barge", action="store_true",
                    help="log every interruption stage and its latency")
    ap.add_argument(
        "--half-duplex", action="store_true",
        help="legacy mode: stop listening while speaking (volume-based barge-in)",
    )
    args = ap.parse_args()

    if not C.GROQ_API_KEY:
        print("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")
        return 1

    _banner()
    if not _check_devices():
        return 1

    # Warm the connection so the first spoken turn is not slowed by cold start.
    brain.warm_up()

    app = graph.build() if args.half_duplex else graph_duplex.build()
    final = None
    # Bound here so the save path below can see them in either mode.
    recorder = voice = session = None

    try:
        with audio.Microphone() as mic:
            seed = {"q_index": 0, "followups": 0, "history": [], "silent_turns": 0}
            if args.half_duplex:
                graph.MIC = mic
                final = app.invoke(seed, config={"recursion_limit": 250})
            else:
                # Capture and recognition run alongside playback.
                voice = sess.VoiceSession()
                store = sess.MemoryStore()
                recorder = rec.InterviewRecorder(
                    voice.session_id, outdir=args.outdir,
                )
                with duplex.DuplexSession(mic) as dsession:
                    session = dsession
                    session.on_utterance = _show_backchannel
                    session.on_partial = _show_partial
                    session.verbose = args.debug_barge
                    graph_duplex.SESSION = session
                    graph_duplex.VOICE = voice
                    graph_duplex.RECORDER = recorder
                    try:
                        final = app.invoke(seed, config={"recursion_limit": 250})
                    finally:
                        store.save(voice)
                        _report_metrics(voice, session)
    except KeyboardInterrupt:
        print("\n\n  Interview ended early.")
    except Exception as exc:
        print(f"\n  Error: {exc}")
        # Fall through to the save below rather than returning here: an
        # interview that crashed mid-question still has real answers in the
        # recorder (add_answer runs per-turn, independent of whether invoke()
        # ever returns) and a dangling `interviews` row in Postgres with
        # finished_at still NULL. Losing both on top of the crash turned one
        # bug into two - five of seven Postgres-backed sessions were never
        # marked finished and never got a local transcript at all, purely
        # because this function returned before reaching the save code.
        #
        # The printed message alone disappears once the terminal scrolls
        # past it, and a crash mid-interview otherwise leaves nothing but an
        # empty results.json to explain why - this writes the real traceback
        # next to that interview's other saved files instead.
        if recorder is not None:
            try:
                import traceback
                recorder.dir.mkdir(parents=True, exist_ok=True)
                (recorder.dir / "error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            except Exception:
                pass

    # session.history (not `final`, which is None unless invoke() returned
    # normally) is what a Ctrl+C or an exception mid-interview still leaves
    # behind - the transcript up to the point things stopped.
    history = (final or {}).get("history", [])
    if not history and voice is not None:
        history = voice.history

    notes = ""
    if history:
        print("\n" + "=" * 62)
        print("  TRANSCRIPT".center(62))
        print("=" * 62)
        for turn in history:
            who = "Jerry" if turn["role"] == "agent" else "You "
            print(f"  {who}: {turn['text']}")

        if not args.no_notes:
            print("\n  Generating debrief notes...\n")
            notes = brain.summarize(history, [])
            if notes:
                print("=" * 62)
                print("  DEBRIEF NOTES".center(62))
                print("=" * 62)
                print(notes)

    if recorder is not None:
        # Always save, even with an empty transcript: whatever per-question
        # answers were already recorded during the session (in this run's
        # own database table, if configured) do not depend on this.
        recorder.set_transcript(history)
        recorder.set_notes(notes)
        if voice is not None:
            recorder.set_metrics(
                voice.summary(),
                session.barge.stats() if session is not None else None,
            )
        outdir = recorder.save()
        print(f"\n  Saved {recorder.summary_line()}")
        print(f"  -> {outdir}")
        print("     answers.md      readable answers per question")
        print("     results.json    full structured record")
        print("     transcript.txt  plain transcript")
        print()
    elif history:
        # Half-duplex mode keeps the original flat transcript file.
        path = args.save or f"interview_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "transcript": history,
                    "notes": notes,
                },
                fh,
                indent=2,
            )
        print(f"\n  Saved to {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
