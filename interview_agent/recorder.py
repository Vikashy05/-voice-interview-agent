"""Records what the candidate actually said: text, timings, and the flow.

Keeping answers keyed to questions means the results are reviewable per
question rather than as one flat log. No audio is recorded - only the
conversation itself, in text.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config as C


@dataclass
class Answer:
    """One answer, tied to the question that prompted it.

    question_index is the question's 1-based position in questions.QUESTIONS
    (1 for the first question, and so on) - questions are plain text with no
    separate name, so position is what ties an answer back to which question
    it was given for. The caller (graph_duplex.py) converts from the 0-based
    list index it uses internally before this is ever set.
    """

    question_index: int
    question: str
    text: str
    started_at: float
    duration_s: float = 0.0
    was_followup: bool = False
    interrupted_agent: bool = False
    word_count: int = 0


@dataclass
class Recording:
    """Everything captured from one interview."""

    session_id: str
    started_at: str
    answers: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    notes: str = ""
    metrics: dict = field(default_factory=dict)
    barge_in: dict = field(default_factory=dict)


class InterviewRecorder:
    """Writes answers and a readable transcript to an output folder."""

    def __init__(self, session_id: str, outdir: str | Path | None = None) -> None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(outdir or f"interviews/{stamp}_{session_id}")
        self.rec = Recording(
            session_id=session_id,
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
        self._n = 0

        # The local files above are always written; Postgres is additional
        # and optional, so its absence or a connection failure must never
        # stop an interview that would otherwise run fine. This run's table
        # name matches its interviews/ folder name exactly (same stamp and
        # session_id), so the two are trivial to line up by eye.
        self._db = None
        self._run_id = f"{stamp}_{session_id}"
        if C.DATABASE_URL:
            try:
                from . import db as db_module
                self._db = db_module.get_db()
                self._db.create_interview_table(self._run_id)
            except Exception as exc:
                print(f"  [db: could not start interview table: {exc}]")
                self._db = None

    def _ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def add_answer(
        self,
        question_index: int,
        question: str,
        text: str,
        started_at: float = 0.0,
        was_followup: bool = False,
        interrupted_agent: bool = False,
    ) -> Answer:
        self._n += 1

        ans = Answer(
            question_index=question_index,
            question=question,
            text=text,
            started_at=started_at,
            was_followup=was_followup,
            interrupted_agent=interrupted_agent,
            word_count=len(text.split()),
        )
        self.rec.answers.append(ans)

        if self._db is not None:
            try:
                self._db.add_answer(
                    run_id=self._run_id,
                    question_no=question_index,
                    question=question,
                    answer=text,
                )
            except Exception as exc:
                print(f"  [db: could not save answer: {exc}]")

        return ans

    def log_turn(self, role: str, text: str) -> None:
        """Append one turn to conversation.txt the moment it happens.

        The full transcript is only written when the interview finishes, so
        a session ended with Ctrl+C - or one that crashed - used to leave
        nothing behind at all. This costs one small append per turn and
        means the conversation survives however the interview ends.
        """
        if not text:
            return
        who = C.BOT_NAME if role == "agent" else "You"
        try:
            self._ensure()
            with open(self.dir / "conversation.txt", "a", encoding="utf-8") as fh:
                fh.write(f"{who}: {text}\n\n")
        except Exception as exc:
            # Losing the log must never stop the interview, but this is the
            # one thing meant to survive a hang or a kill - if disk space,
            # permissions, or the path itself is the problem, that needs to
            # be visible instead of a folder that silently has no
            # conversation.txt and no explanation why.
            print(f"  [recorder: could not append to conversation.txt: {exc}]")

    def log_event(self, label: str, detail: str = "") -> None:
        """Record something worth seeing in the log but not spoken aloud."""
        try:
            self._ensure()
            with open(self.dir / "conversation.txt", "a", encoding="utf-8") as fh:
                fh.write(f"    [{label}{': ' + detail if detail else ''}]\n\n")
        except Exception as exc:
            print(f"  [recorder: could not append to conversation.txt: {exc}]")

    def set_transcript(self, history: list[dict]) -> None:
        self.rec.transcript = history

    def set_notes(self, notes: str) -> None:
        self.rec.notes = notes or ""

    def set_metrics(self, metrics: dict, barge_in: dict | None = None) -> None:
        self.rec.metrics = metrics or {}
        self.rec.barge_in = barge_in or {}

    # -- output ------------------------------------------------------------
    def save(self) -> Path:
        """Write results.json, transcript.txt, and answers.md."""
        self._ensure()

        blob = asdict(self.rec)
        (self.dir / "results.json").write_text(
            json.dumps(blob, indent=2, default=str), encoding="utf-8"
        )

        lines = [f"Interview {self.rec.session_id}  {self.rec.started_at}", ""]
        for turn in self.rec.transcript:
            who = C.BOT_NAME.upper() if turn.get("role") == "agent" else "YOU"
            lines.append(f"{who}: {turn.get('text','')}")
        (self.dir / "transcript.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        md = [
            "# Interview Answers",
            "",
            f"**Session:** {self.rec.session_id}  ",
            f"**Date:** {self.rec.started_at}  ",
            f"**Answers:** {len(self.rec.answers)}",
            "",
        ]
        for i, a in enumerate(self.rec.answers, 1):
            md.append(f"## {i}. {a.question}")
            md.append("")
            if a.was_followup:
                md.append("*(follow-up probe)*")
                md.append("")
            md.append(f"> {a.text or '_no answer captured_'}")
            md.append("")
            bits = [f"{a.word_count} words"]
            if a.interrupted_agent:
                bits.append("interrupted the interviewer")
            md.append(f"*{' · '.join(bits)}*")
            md.append("")
        if self.rec.notes:
            md += ["---", "", "## Debrief notes", "", self.rec.notes, ""]
        (self.dir / "answers.md").write_text("\n".join(md), encoding="utf-8")

        return self.dir

    def summary_line(self) -> str:
        n = len(self.rec.answers)
        words = sum(a.word_count for a in self.rec.answers)
        return f"{n} answers, {words} words"

    @property
    def conversation_file(self) -> Path:
        return self.dir / "conversation.txt"
