"""Optional local password sessions. No public accounts, cloud identity or default passwords."""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .fixtures import ACTORS

COOKIE = "lab_session"


class AuthenticationError(PermissionError):
    pass


def password_hash(password):
    if not isinstance(password, str) or not 16 <= len(password) <= 256:
        raise ValueError("Local passwords require 16 to 256 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=131072, r=8, p=1,
                            maxmem=256 * 1024 * 1024, dklen=32)
    return "scrypt$" + salt.hex() + "$" + digest.hex()


def verify_password(password, encoded):
    _, salt, expected = encoded.split("$")
    actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=131072, r=8, p=1,
                           maxmem=256 * 1024 * 1024, dklen=32)
    return hmac.compare_digest(actual.hex(), expected)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class LocalSessions:
    def __init__(self, root, credential_file):
        self.path = Path(root) / "identity.sqlite3"
        self.credential_file = Path(credential_file)
        self.credentials()  # Missing/invalid configuration fails startup, never falls back to simulation.
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY, actor TEXT NOT NULL, csrf TEXT NOT NULL,
                    credential TEXT NOT NULL, expires REAL NOT NULL, last_seen REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS login_attempts (
                    key TEXT PRIMARY KEY, failures INTEGER NOT NULL, blocked_until REAL NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def credentials(self):
        data = json.loads(self.credential_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError("Configure at least one local account.")
        known = {a["id"] for a in ACTORS}
        for actor, entry in data.items():
            if actor not in known or not isinstance(entry, dict) or type(entry.get("enabled")) is not bool:
                raise ValueError("Invalid local account configuration.")
            parts = entry.get("password_hash", "").split("$")
            if len(parts) != 3 or parts[0] != "scrypt" or len(parts[1]) != 32 or len(parts[2]) != 64:
                raise ValueError("Invalid local password hash.")
            bytes.fromhex(parts[1]); bytes.fromhex(parts[2])
        return data

    def active_actor(self, actor):
        entry = self.credentials().get(actor)
        if not entry or not entry["enabled"]:
            raise PermissionError("Local account is disabled or unavailable.")
        return actor

    def login(self, actor, password, client):
        accounts = self.credentials()
        entry = accounts.get(actor)
        # The same expensive check is used for unknown/disabled accounts.
        encoded = entry["password_hash"] if entry else next(iter(accounts.values()))["password_hash"]
        keys = [token_hash("account:" + actor), token_hash("client:" + client)]
        now = time.time()
        with self.connection() as db:
            rows = [db.execute("SELECT * FROM login_attempts WHERE key=?", (key,)).fetchone() for key in keys]
            if any(row and row["blocked_until"] > now for row in rows):
                raise AuthenticationError("Login temporarily unavailable. Try again later.")
            valid = verify_password(password, encoded) and entry and entry["enabled"]
            if not valid:
                for key, row in zip(keys, rows):
                    count = (row["failures"] if row and row["blocked_until"] == 0 else 0) + 1
                    db.execute("INSERT OR REPLACE INTO login_attempts VALUES(?,?,?)",
                               (key, count, now + 300 if count >= 5 else 0))
            else:
                for key in keys:
                    db.execute("DELETE FROM login_attempts WHERE key=?", (key,))
                token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
                db.execute("DELETE FROM sessions WHERE expires<=? OR last_seen<=?", (now, now - 900))
                db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)",
                           (token_hash(token), actor, token_hash(csrf), token_hash(encoded), now + 3600, now))
        if not valid:
            raise AuthenticationError("Invalid local credentials.")
        return token, csrf, self.principal(actor)

    @staticmethod
    def principal(actor):
        profile = next(a for a in ACTORS if a["id"] == actor)
        return {**profile, "simulated_identity": False}

    def authenticate(self, token, csrf=None, mutation=False):
        if not token or len(token) > 128:
            raise AuthenticationError("Sign in to the local lab.")
        now = time.time()
        with self.connection() as db:
            session = db.execute("SELECT * FROM sessions WHERE digest=?", (token_hash(token),)).fetchone()
            if not session or session["expires"] <= now or session["last_seen"] <= now - 900:
                raise AuthenticationError("Local session is missing or expired.")
            entry = self.credentials().get(session["actor"])
            if not entry or not entry["enabled"] or session["credential"] != token_hash(entry["password_hash"]):
                raise AuthenticationError("Local session has been revoked.")
            if mutation and (not csrf or not hmac.compare_digest(token_hash(csrf), session["csrf"])):
                raise AuthenticationError("Request requires the local session verification token.")
            db.execute("UPDATE sessions SET last_seen=? WHERE digest=?", (now, token_hash(token)))
        return self.principal(session["actor"])

    def logout(self, token):
        with self.connection() as db:
            db.execute("DELETE FROM sessions WHERE digest=?", (token_hash(token),))

    def authorize_job(self, actor, kind):
        self.active_actor(actor)
        if kind == "train" and actor != "reviewer":
            raise PermissionError("Local classifier training requires the reviewer account.")


def configured_auth(root):
    mode = os.environ.get("LAB_AUTH_MODE", "simulated")
    if mode == "simulated":
        return None
    if mode != "local_sessions":
        raise ValueError("LAB_AUTH_MODE must be simulated or local_sessions.")
    return LocalSessions(root, os.environ["LAB_CREDENTIAL_FILE"])
