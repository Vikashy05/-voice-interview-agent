"""Live web dashboard for a locally running interview.

This module never edits or replaces the existing CLI engine - it only
observes it. `graph_duplex.ON_EVENT` is already a public hook meant for
exactly this (see its docstring), and `VoiceSession.transition` is wrapped
at the class level (not modified in session.py) so avatar/voice state
changes are visible too. The interview itself keeps running exactly as
`python -m interview_agent.main` always has: local microphone in, local
speakers out. This process just broadcasts what is happening so a browser
on the same machine can show it in real time.

Run with:  python -m interview_agent.dashboard
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import graph_duplex
from . import session as sess
from .questions import QUESTIONS, SIGNOFF_OPTIONS

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
URL = "http://127.0.0.1:8756"

app = FastAPI()

_clients: set[WebSocket] = set()
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

# Counts agent turns so the dashboard can show "question N of len(QUESTIONS)"
# without graph_duplex needing to broadcast q_index itself.
_progress = {"asked": 0}

# A tab that opens (or reopens, e.g. after a refresh) partway through an
# interview used to see nothing until the next event fired - broadcasts only
# ever reached whoever was already connected at the moment they were sent,
# so a client that connected late showed the lobby forever even though the
# interview had moved well past the greeting. Keep the last message of each
# type so a newly-connected client can be caught up immediately.
_last_by_type: dict[str, dict[str, Any]] = {}
_history: list[dict[str, Any]] = []


def _remember(payload: dict[str, Any]) -> None:
    _last_by_type[payload["type"]] = payload
    if payload["type"] == "event":
        _history.append(payload)


def _broadcast(payload: dict[str, Any]) -> None:
    """Thread-safe fan-out: called from the interview's own worker threads."""
    _remember(payload)
    if _loop is None:
        return
    message = json.dumps(payload)
    with _clients_lock:
        targets = list(_clients)
    for ws in targets:
        asyncio.run_coroutine_threadsafe(_safe_send(ws, message), _loop)


async def _safe_send(ws: WebSocket, message: str) -> None:
    try:
        await ws.send_text(message)
    except Exception:
        with _clients_lock:
            _clients.discard(ws)


def _on_event(kind: str, **data: Any) -> None:
    if kind == "agent" and data.get("text") in {q["text"] for q in QUESTIONS}:
        _progress["asked"] += 1
    _broadcast({
        "type": "event",
        "kind": kind,
        "data": data,
        "progress": {"asked": _progress["asked"], "total": len(QUESTIONS)},
    })
    # A signoff line is only ever spoken by close() once the interview is
    # over, so matching against the full set of possible signoffs (imported,
    # not duplicated) is a reliable "we're done" signal for the page without
    # graph_duplex needing a dedicated event.
    if kind == "agent" and data.get("text") in SIGNOFF_OPTIONS:
        _broadcast({"type": "done", "progress": {"asked": _progress["asked"],
                                                  "total": len(QUESTIONS)}})


def _check_mic() -> dict[str, Any]:
    """Same check main.py's _check_devices() does, structured for the page.

    Kept as its own function rather than importing main._check_devices()
    directly because that one prints to the console and returns only a
    bool - the dashboard needs the device names and error text to show on
    the page, not just pass/fail.
    """
    import sounddevice as sd
    from . import config as C

    try:
        sd.check_input_settings(samplerate=C.SAMPLE_RATE, channels=1)
        sd.check_output_settings(channels=1)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "mic": sd.query_devices(kind="input")["name"].strip(),
        "out": sd.query_devices(kind="output")["name"].strip(),
    }


def _broadcast_mic_check() -> dict[str, Any]:
    result = _check_mic()
    _broadcast({"type": "mic_check", **result})
    return result


def _install_hooks() -> None:
    """Wire observation points without touching graph_duplex.py or session.py."""
    graph_duplex.ON_EVENT = _on_event

    # graph_duplex.STOP_REQUESTED is already public API: every routing
    # function in the graph checks _stop_requested() before deciding where
    # to go next (see route_listen's comment on why it is checked there and
    # not only after _after()). main.py never sets it today, so it is
    # permanently None/False for the CLI - assigning a real Event here is
    # additive, not a behavior change, and gives the dashboard's End
    # Interview button a real way to stop the graph at its next check
    # instead of the page merely pretending to do so.
    graph_duplex.STOP_REQUESTED = threading.Event()

    original_transition = sess.VoiceSession.transition

    def transition_and_broadcast(self, to, *, strict: bool = False) -> bool:
        ok = original_transition(self, to, strict=strict)
        if ok:
            _broadcast({"type": "state", "state": to.value})
        return ok

    sess.VoiceSession.transition = transition_and_broadcast


@app.post("/api/end-interview")
async def end_interview() -> JSONResponse:
    if graph_duplex.STOP_REQUESTED is not None:
        graph_duplex.STOP_REQUESTED.set()
    return JSONResponse({"ok": True})


async def _catch_up(ws: WebSocket) -> None:
    """Replay everything a client needs to render the current moment.

    Order matters: the full transcript first (so turns appear in the right
    order), then the latest state/progress on top, then a mic-check or done
    result if either already happened - each of those is a single snapshot,
    not something to replay from history.
    """
    for payload in _history:
        await ws.send_text(json.dumps(payload))
    for kind in ("mic_check", "state", "done"):
        payload = _last_by_type.get(kind)
        if payload is not None:
            await ws.send_text(json.dumps(payload))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    await ws.accept()
    with _clients_lock:
        _clients.add(ws)
    try:
        await _catch_up(ws)
        while True:
            # The dashboard is read-only; any client message is ignored, but
            # the receive keeps the connection alive and detects disconnects.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except ConnectionResetError:
        # Windows/Proactor logs this on an abrupt disconnect (e.g. a page
        # refresh) even though it is the same ordinary close as
        # WebSocketDisconnect elsewhere - nothing to report, just cleanup.
        pass
    finally:
        with _clients_lock:
            _clients.discard(ws)


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def _quiet_proactor_reset(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Silence a known Windows/Proactor noise, not a real failure.

    A browser tab closing or refreshing resets the socket; on Windows the
    Proactor loop's own connection-lost callback then logs a
    ConnectionResetError traceback for it - harmless (the interview keeps
    running, as confirmed live: this fired mid-interview with no other
    effect), but alarming enough in the console to look like a crash.
    """
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return
    loop.default_exception_handler(context)


async def _serve() -> None:
    import uvicorn

    # uvicorn.run() creates and owns its own event loop internally, so a
    # loop configured with set_exception_handler() before calling it (the
    # previous approach here) is simply discarded before uvicorn ever
    # touches it - the handler never actually applied and the Proactor
    # ConnectionResetError kept printing. Driving uvicorn's Server directly
    # inside a loop this function controls is what lets the handler stick.
    asyncio.get_running_loop().set_exception_handler(_quiet_proactor_reset)
    config = uvicorn.Config(app, host="127.0.0.1", port=8756, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


def _run_server() -> None:
    asyncio.run(_serve())


def main() -> int:
    _install_hooks()
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    # Give uvicorn a moment to actually bind before the browser requests the
    # page, and before a client could be connected to receive the mic-check
    # broadcast below.
    time.sleep(1.0)
    print(f"\n  Dashboard: {URL}\n")
    try:
        webbrowser.open(URL)
    except Exception:
        pass

    # A few seconds for the page to load and the WebSocket to connect, so
    # the mic-check result is not broadcast into an empty room and missed.
    time.sleep(2.0)
    result = _broadcast_mic_check()
    if not result["ok"]:
        print(f"  Audio device problem: {result['error']}\n")
        return 1
    print(f"  mic: {result['mic']}")
    print(f"  out: {result['out']}\n")

    from . import main as cli_main
    return cli_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
