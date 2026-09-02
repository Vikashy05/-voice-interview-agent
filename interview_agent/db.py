"""PostgreSQL-backed storage for one interview: a table per run.

Each interview gets its own table, named after when and which session it
was (see `_table_name`), with exactly the columns the interview needs:
  question_no, question, answer

No shared questions/interviews/answers tables anymore - the question list
itself always comes from questions.py now, and this module's only job is to
record what happened during one specific run. There is deliberately no
cross-interview table to join or filter: reviewing an interview means
opening its own table (or its files under interviews/, which are written
regardless of whether a database is configured at all).

`answer` holds the candidate's real answer text, or guardrails.NOT_ANSWERED
when what they said was rejected as silence, filler, or STT garbage rather
than a genuine attempt - see guardrails.is_meaningful_answer().

Connect via DATABASE_URL in .env, e.g.:
  DATABASE_URL=postgresql://postgres:password@localhost:5432/interview_agent
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager

from . import config as C

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.sql import SQL, Identifier
    _DRIVER = "psycopg"
except ImportError:  # pragma: no cover - fallback for psycopg2-only envs
    import psycopg2
    import psycopg2.extras
    from psycopg2 import sql as _pg2_sql
    SQL = _pg2_sql.SQL
    Identifier = _pg2_sql.Identifier
    _DRIVER = "psycopg2"


_VALID_TABLE_SUFFIX = re.compile(r"^[a-zA-Z0-9_]+$")


def table_name(run_id: str) -> str:
    """A safe, unique table name for one run, e.g. interview_20260826_144759_a1b2c3.

    Table names cannot be parameterised the way values can in SQL, so this
    is the one thing here that must be validated rather than trusted: a
    session id containing anything other than letters, digits, and
    underscores would otherwise let arbitrary identifiers (or, worst case,
    injected SQL) into a query built with plain string formatting.
    """
    if not _VALID_TABLE_SUFFIX.match(run_id):
        raise ValueError(
            f"run_id must contain only letters, digits, and underscores - got: {run_id!r}"
        )
    return f"interview_{run_id}"


class Database:
    """One connection, guarded by a lock.

    An interview is a single-user, mostly-sequential session, so a pool would
    be unused complexity; a lock is enough to make it safe against the
    background threads (recorder, UI meter) that might touch it concurrently.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or C.DATABASE_URL
        if not self.dsn:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to your .env file, e.g. "
                "DATABASE_URL=postgresql://postgres:password@localhost:5432/interview_agent"
            )
        self._lock = threading.Lock()
        self._conn = self._connect()

    def _connect(self):
        if _DRIVER == "psycopg":
            return psycopg.connect(self.dsn, autocommit=True)
        return psycopg2.connect(self.dsn)

    def _reconnect_if_needed(self) -> None:
        """Replace a connection that Postgres has already dropped.

        A network blip or a DB restart mid-interview used to kill
        persistence for the rest of the session: every call after the drop
        raised the same OperationalError forever, since nothing ever
        replaced the dead connection. `.closed` only catches a connection
        this process closed itself, not one the server dropped out from
        under it, so a lightweight probe query is what actually detects that
        case before a real caller's query hits it.
        """
        if self._conn.closed:
            print("  [db: connection was closed, reconnecting]")
            self._conn = self._connect()
            return
        try:
            with self._conn.cursor() as probe:
                probe.execute("SELECT 1")
        except Exception as exc:
            print(f"  [db: connection appears dead ({exc}), reconnecting]")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = self._connect()

    @contextmanager
    def _cursor(self):
        with self._lock:
            self._reconnect_if_needed()
            if _DRIVER == "psycopg":
                with self._conn.cursor(row_factory=dict_row) as cur:
                    yield cur
            else:
                with self._conn.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cur:
                    yield cur
                    self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -- one table per interview --------------------------------------------
    def create_interview_table(self, run_id: str) -> str:
        """Create this run's table if it does not already exist.

        Returns the table name actually used, so the caller can record it.
        """
        name = table_name(run_id)
        with self._cursor() as cur:
            cur.execute(
                SQL(
                    "CREATE TABLE IF NOT EXISTS {} ("
                    "question_no INTEGER PRIMARY KEY, "
                    "question TEXT NOT NULL, "
                    "answer TEXT NOT NULL"
                    ")"
                ).format(Identifier(name))
            )
        return name

    def add_answer(self, run_id: str, question_no: int, question: str, answer: str) -> None:
        """Save one question's answer, overwriting if this question_no was
        already recorded (e.g. the candidate re-answered after a redirect)."""
        name = table_name(run_id)
        with self._cursor() as cur:
            cur.execute(
                SQL(
                    "INSERT INTO {} (question_no, question, answer) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (question_no) DO UPDATE SET "
                    "question = EXCLUDED.question, answer = EXCLUDED.answer"
                ).format(Identifier(name)),
                (question_no, question, answer),
            )

    def get_answers(self, run_id: str) -> list[dict]:
        """This run's answers, in question order."""
        name = table_name(run_id)
        with self._cursor() as cur:
            cur.execute(
                SQL("SELECT question_no, question, answer FROM {} ORDER BY question_no")
                .format(Identifier(name))
            )
            return list(cur.fetchall())


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    """The process-wide database handle, created on first use."""
    global _db
    with _db_lock:
        if _db is None:
            _db = Database()
        return _db


def close_db() -> None:
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
            _db = None
