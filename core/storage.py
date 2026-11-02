"""Persistência em SQLite: documentos, achados e trilha dos agentes."""
from __future__ import annotations

import json
import shutil
import sqlite3
import time

from .config import DATA_DIR, PAGES_DIR
from .models import Finding

DB_PATH = DATA_DIR / "app.db"


class DB:
    def __init__(self):
        self.con = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY, name TEXT, kind TEXT, sha TEXT, n_pages INTEGER, n_chunks INTEGER,
            indexed_at REAL, verdict TEXT, risk REAL, conformity REAL, rationale TEXT, trace TEXT, params TEXT,
            pages TEXT, extra TEXT, source_path TEXT);
        CREATE TABLE IF NOT EXISTS findings(id TEXT PRIMARY KEY, doc_id TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS logs(ts REAL, nivel TEXT, msg TEXT);
        """)

    # documentos
    def find_by_sha(self, sha: str):
        r = self.con.execute("SELECT * FROM documents WHERE sha=?", (sha,)).fetchone()
        return dict(r) if r else None

    def save_document(self, doc: dict, findings: list[Finding]):
        doc = dict(doc)
        for k in ("trace", "params", "pages", "extra"):
            doc[k] = json.dumps(doc.get(k) or ([] if k in ("trace", "pages") else {}), ensure_ascii=False)
        cols = ["id", "name", "kind", "sha", "n_pages", "n_chunks", "indexed_at", "verdict", "risk", "conformity",
                "rationale", "trace", "params", "pages", "extra", "source_path"]
        self.con.execute(f"INSERT OR REPLACE INTO documents({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                         [doc.get(c) for c in cols])
        self.con.execute("DELETE FROM findings WHERE doc_id=?", (doc["id"],))
        self.con.executemany("INSERT INTO findings(id, doc_id, data) VALUES(?,?,?)",
                             [(f.id, doc["id"], json.dumps(f.to_dict(), ensure_ascii=False)) for f in findings])
        self.con.commit()

    def documents(self) -> list[dict]:
        rows = self.con.execute("SELECT * FROM documents ORDER BY indexed_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("trace", "params", "pages", "extra"):
                try:
                    d[k] = json.loads(d[k]) if d[k] else None
                except Exception:
                    d[k] = None
            out.append(d)
        return out

    def document(self, doc_id: str) -> dict | None:
        for d in self.documents():
            if d["id"] == doc_id:
                return d
        return None

    def findings(self, doc_id: str | None = None) -> list[Finding]:
        if doc_id:
            rows = self.con.execute("SELECT data FROM findings WHERE doc_id=?", (doc_id,)).fetchall()
        else:
            rows = self.con.execute("SELECT data FROM findings").fetchall()
        return [Finding.from_dict(json.loads(r["data"])) for r in rows]

    def delete_document(self, doc_id: str):
        self.con.execute("DELETE FROM findings WHERE doc_id=?", (doc_id,))
        self.con.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.con.commit()
        shutil.rmtree(PAGES_DIR / doc_id, ignore_errors=True)

    def reset(self):
        self.con.executescript("DELETE FROM findings; DELETE FROM documents;")
        self.con.commit()
        shutil.rmtree(PAGES_DIR, ignore_errors=True)
        PAGES_DIR.mkdir(parents=True, exist_ok=True)

    # logs
    def log(self, nivel: str, msg: str):
        self.con.execute("INSERT INTO logs VALUES(?,?,?)", (time.time(), nivel, msg))
        self.con.commit()

    def logs(self, n: int = 200) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM logs ORDER BY ts DESC LIMIT ?", (n,)).fetchall()]
