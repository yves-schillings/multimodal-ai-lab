"""SQLite workflow with simulated identities, scoped access and revision checks.

This is a local demonstration. The caller-selected actor is not authentication.
Original transcription segments are append-only; human corrections live separately.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .fixtures import ACTORS, DEMO_CASES


class ConflictError(ValueError):
    """The client edited a stale case revision."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _stamp(seconds):
    milliseconds = round(float(seconds) * 1000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class StatementStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "statements.sqlite3"
        with self._connection(write=True) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, owner_actor TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'empty',
                    active_transcript_id TEXT, draft_json TEXT, approved_revision INTEGER,
                    approved_by TEXT, last_editor TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id),
                    created_at TEXT NOT NULL, actor TEXT NOT NULL, engine TEXT NOT NULL,
                    language TEXT, content_hash TEXT NOT NULL, case_revision INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS initial_segments (
                    transcript_id TEXT NOT NULL REFERENCES transcripts(id), id TEXT NOT NULL,
                    position INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
                    speaker TEXT NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY(transcript_id, id)
                );
                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id),
                    transcript_id TEXT NOT NULL REFERENCES transcripts(id), position INTEGER NOT NULL,
                    start REAL NOT NULL, end REAL NOT NULL, speaker TEXT NOT NULL,
                    text TEXT NOT NULL, reviewed INTEGER NOT NULL DEFAULT 0,
                    reviewed_by TEXT, reviewed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(id),
                    timestamp TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
                    revision INTEGER NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            if db.execute("SELECT value FROM settings WHERE key='seeded'").fetchone() is None:
                for case in DEMO_CASES:
                    now = _now()
                    db.execute("INSERT INTO cases(id,title,owner_actor,created_at,updated_at) VALUES(?,?,?,?,?)",
                               (case["id"], case["title"], case["owner_actor"], now, now))
                    self._audit(db, case["id"], case["owner_actor"], "case_created", 0, {"synthetic": True})
                    self._save_transcript(db, case["owner_actor"], case["id"], {
                        "engine": "synthetic-fixture", "language": "en", "segments": case["segments"]})
                db.execute("INSERT INTO settings(key,value) VALUES('seeded','1')")

    @contextmanager
    def _connection(self, write=False):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _actor(actor):
        if actor not in {item["id"] for item in ACTORS}:
            raise PermissionError("Unknown simulated actor.")
        return actor

    def _case(self, db, actor, case_id):
        self._actor(actor)
        case = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        if case is None:
            raise KeyError("Case not found.")
        if actor != "reviewer" and case["owner_actor"] != actor:
            raise PermissionError("This simulated actor cannot access this case.")
        return case

    @staticmethod
    def _revision(case, expected_revision):
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("expected_revision must be an integer case revision.")
        if case["revision"] != expected_revision:
            raise ConflictError("The case has changed. Reload before saving.")

    @staticmethod
    def _audit(db, case_id, actor, action, revision, metadata):
        db.execute("INSERT INTO audit(case_id,timestamp,actor,action,revision,metadata_json) VALUES(?,?,?,?,?,?)",
                   (case_id, _now(), actor, action, revision, json.dumps(metadata, sort_keys=True)))

    def actors(self):
        return [dict(actor) for actor in ACTORS]

    def list_cases(self, actor):
        self._actor(actor)
        with self._connection() as db:
            query = "SELECT id,title,owner_actor,revision,status,approved_revision,created_at,updated_at FROM cases"
            rows = db.execute(query + ("" if actor == "reviewer" else " WHERE owner_actor=?") + " ORDER BY created_at,id",
                              () if actor == "reviewer" else (actor,)).fetchall()
            return [dict(row) for row in rows]

    def create_case(self, actor, title):
        self._actor(actor)
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 200:
            raise ValueError("Provide a case title of 1 to 200 characters.")
        case_id, now = "case-" + uuid.uuid4().hex, _now()
        with self._connection(write=True) as db:
            db.execute("INSERT INTO cases(id,title,owner_actor,created_at,updated_at) VALUES(?,?,?,?,?)",
                       (case_id, title.strip(), actor, now, now))
            self._audit(db, case_id, actor, "case_created", 0, {"title_hash": _hash(title.strip())})
        return self.get_case(actor, case_id)

    def get_case(self, actor, case_id):
        with self._connection() as db:
            case = dict(self._case(db, actor, case_id))
            rows = db.execute("""SELECT s.*, i.text AS original_text FROM segments s
                JOIN initial_segments i ON s.transcript_id=i.transcript_id AND s.id=i.id
                WHERE s.case_id=? ORDER BY s.position""", (case_id,)).fetchall()
            case["segments"] = [{**dict(row), "reviewed": bool(row["reviewed"])} for row in rows]
            case["draft"] = json.loads(case.pop("draft_json")) if case["draft_json"] else None
            case.pop("draft_json", None)
            case["transcript_history"] = [dict(row) for row in db.execute(
                "SELECT id,created_at,actor,engine,language,content_hash,case_revision FROM transcripts WHERE case_id=? ORDER BY case_revision",
                (case_id,))]
            case["audit"] = []
            for row in db.execute("SELECT * FROM audit WHERE case_id=? ORDER BY id", (case_id,)):
                event = dict(row)
                event["metadata"] = json.loads(event.pop("metadata_json"))
                case["audit"].append(event)
            case["simulated_identity"] = getattr(self, "simulated_identity", True)
            return case

    @staticmethod
    def _normalize_transcription(transcription):
        if not isinstance(transcription, dict):
            raise ValueError("Transcription must be an object.")
        raw = transcription.get("segments")
        if not isinstance(raw, list) or not raw or len(raw) > 5000:
            raise ValueError("Transcription requires 1 to 5000 timestamped segments.")
        normalized = []
        for segment in raw:
            if not isinstance(segment, dict):
                raise ValueError("Each segment must be an object.")
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 20000:
                raise ValueError("Each segment requires non-empty text of at most 20000 characters.")
            try:
                start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
            except (ValueError, TypeError):
                raise ValueError("Segment timestamps must be numeric.") from None
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError("Segment timestamps must be finite, ordered and non-negative.")
            speaker = segment.get("speaker") or "Unverified speaker"
            if not isinstance(speaker, str) or len(speaker) > 100:
                raise ValueError("Speaker label must be a string of at most 100 characters.")
            normalized.append({"start": start, "end": end, "speaker": speaker, "text": text.strip()})
        if sum(len(item["text"]) for item in normalized) > 1000000:
            raise ValueError("Transcript is too large for this local demonstration.")
        return normalized

    def _save_transcript(self, db, actor, case_id, transcription):
        case = self._case(db, actor, case_id)
        segments = self._normalize_transcription(transcription)
        transcript_id, now, revision = "transcript-" + uuid.uuid4().hex, _now(), case["revision"] + 1
        engine = str(transcription.get("engine") or transcription.get("backend") or "unspecified")[:160]
        language = str(transcription.get("language") or "unknown")[:40]
        digest = _hash(segments)
        db.execute("INSERT INTO transcripts VALUES(?,?,?,?,?,?,?,?)",
                   (transcript_id, case_id, now, actor, engine, language, digest, revision))
        # Only the editable projection is replaced. Every original transcript remains immutable.
        db.execute("DELETE FROM segments WHERE case_id=?", (case_id,))
        for position, segment in enumerate(segments):
            segment_id = "segment-" + uuid.uuid4().hex
            values = (transcript_id, segment_id, position, segment["start"], segment["end"], segment["speaker"], segment["text"])
            db.execute("INSERT INTO initial_segments VALUES(?,?,?,?,?,?,?)", values)
            db.execute("""INSERT INTO segments(id,case_id,transcript_id,position,start,end,speaker,text)
                VALUES(?,?,?,?,?,?,?,?)""", (segment_id, case_id, transcript_id, position, segment["start"], segment["end"], segment["speaker"], segment["text"]))
        db.execute("""UPDATE cases SET revision=?,status='transcribed',active_transcript_id=?,draft_json=NULL,
            approved_revision=NULL,approved_by=NULL,last_editor=?,updated_at=? WHERE id=?""",
                   (revision, transcript_id, actor, now, case_id))
        self._audit(db, case_id, actor, "transcript_saved", revision,
                    {"transcript_id": transcript_id, "segment_count": len(segments), "content_hash": digest,
                     "invalidated_approval": case["approved_revision"] is not None})

    def save_transcript(self, actor, case_id, transcription):
        with self._connection(write=True) as db:
            self._save_transcript(db, actor, case_id, transcription)
        return self.get_case(actor, case_id)

    def update_segment(self, actor, case_id, segment_id, text, expected_revision: int):
        with self._connection(write=True) as db:
            case = self._case(db, actor, case_id)
            self._revision(case, expected_revision)
            row = db.execute("SELECT * FROM segments WHERE id=? AND case_id=?", (segment_id, case_id)).fetchone()
            if row is None:
                raise KeyError("Segment not found in this case.")
            if not isinstance(text, str) or not text.strip() or len(text) > 20000:
                raise ValueError("Reviewed text must contain 1 to 20000 characters.")
            text, now, revision = text.strip(), _now(), case["revision"] + 1
            db.execute("UPDATE segments SET text=?,reviewed=1,reviewed_by=?,reviewed_at=? WHERE id=?",
                       (text, actor, now, segment_id))
            db.execute("""UPDATE cases SET revision=?,status='in_review',draft_json=NULL,
                approved_revision=NULL,approved_by=NULL,last_editor=?,updated_at=? WHERE id=?""", (revision, actor, now, case_id))
            self._audit(db, case_id, actor, "segment_reviewed", revision,
                        {"segment_id": segment_id, "before_hash": _hash(row["text"]), "after_hash": _hash(text),
                         "invalidated_approval": case["approved_revision"] is not None})
        return self.get_case(actor, case_id)

    def make_draft(self, actor, case_id):
        with self._connection(write=True) as db:
            case = self._case(db, actor, case_id)
            segments = db.execute("SELECT * FROM segments WHERE case_id=? ORDER BY position", (case_id,)).fetchall()
            if not segments or any(not item["reviewed"] for item in segments):
                raise ValueError("Review every transcript segment before generating a draft.")
            revision, now = case["revision"] + 1, _now()
            sources = [{"id": row["id"], "title": f"Transcript segment {index + 1}", "text": row["text"],
                        "start": row["start"], "end": row["end"], "case_id": case_id,
                        "speaker": row["speaker"]} for index, row in enumerate(segments)]
            narrative = "\n\n".join(f"{index + 1}. {row['text']} [{_stamp(row['start'])}–{_stamp(row['end'])}]"
                                    for index, row in enumerate(segments))
            draft = {"text": narrative, "narrative": narrative, "sources": sources, "revision": revision,
                     "based_on_revision": case["revision"], "created_at": now, "mode": "extractive",
                     "notice": "Draft from reviewed testimony; timestamped excerpts, not verified facts or an official certified record."}
            db.execute("""UPDATE cases SET revision=?,status='draft',draft_json=?,approved_revision=NULL,
                approved_by=NULL,updated_at=? WHERE id=?""", (revision, json.dumps(draft, ensure_ascii=False), now, case_id))
            self._audit(db, case_id, actor, "draft_created", revision,
                        {"content_hash": _hash(narrative), "segment_ids": [row["id"] for row in segments],
                         "based_on_revision": case["revision"]})
        return self.get_case(actor, case_id)

    def approve(self, actor, case_id, expected_revision: int):
        with self._connection(write=True) as db:
            case = self._case(db, actor, case_id)
            if actor != "reviewer":
                raise PermissionError("Only the simulated reviewer may approve a draft.")
            self._revision(case, expected_revision)
            reviewer_edits = db.execute("SELECT COUNT(*) FROM segments WHERE case_id=? AND reviewed_by=?", (case_id, actor)).fetchone()[0]
            if case["last_editor"] == actor or reviewer_edits:
                raise PermissionError("The reviewer cannot approve their own transcript edits.")
            draft = json.loads(case["draft_json"]) if case["draft_json"] else None
            if not draft or draft["revision"] != case["revision"]:
                raise ValueError("Generate a current draft before approval.")
            counts = db.execute("SELECT COUNT(*) AS total, SUM(reviewed) AS reviewed FROM segments WHERE case_id=?", (case_id,)).fetchone()
            if not counts["total"] or counts["total"] != counts["reviewed"]:
                raise ValueError("Review every transcript segment before approval.")
            revision = case["revision"] + 1
            db.execute("UPDATE cases SET revision=?,status='approved',approved_revision=?,approved_by=?,updated_at=? WHERE id=?",
                       (revision, draft["revision"], actor, _now(), case_id))
            self._audit(db, case_id, actor, "draft_approved", revision,
                        {"draft_revision": draft["revision"], "content_hash": _hash(draft["text"])})
        return self.get_case(actor, case_id)
