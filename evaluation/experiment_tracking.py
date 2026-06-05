import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """Records and reproduces experiment configurations.
    
    Saves victim config, ontology, intervention history, hypotheses,
    and git metadata to a SQLite database for full reproducibility.
    """

    def __init__(self, db_path: str = "experiments.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    results TEXT,
                    git_hash TEXT,
                    timestamp REAL NOT NULL,
                    duration REAL,
                    status TEXT DEFAULT 'running'
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_git_hash(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def save_experiment(
        self,
        exp_id: str,
        config: Dict[str, Any],
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save or update an experiment record."""
        conn = sqlite3.connect(self.db_path)
        try:
            existing = conn.execute(
                "SELECT id FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
            if existing:
                if results is not None:
                    conn.execute(
                        "UPDATE experiments SET results = ?, status = ? WHERE id = ?",
                        (json.dumps(results), "completed", exp_id),
                    )
                else:
                    conn.execute(
                        "UPDATE experiments SET config = ? WHERE id = ?",
                        (json.dumps(config), exp_id),
                    )
            else:
                conn.execute(
                    """INSERT INTO experiments (id, config, results, git_hash, timestamp, status)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        exp_id,
                        json.dumps(config),
                        json.dumps(results) if results else None,
                        self._get_git_hash(),
                        time.time(),
                        "running" if results is None else "completed",
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def load_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved experiment."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT id, config, results, git_hash, timestamp, duration, status "
                "FROM experiments WHERE id = ?",
                (exp_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "config": json.loads(row[1]) if row[1] else {},
                "results": json.loads(row[2]) if row[2] else None,
                "git_hash": row[3],
                "timestamp": row[4],
                "duration": row[5],
                "status": row[6],
            }
        finally:
            conn.close()

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all registered experiments."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id, config, git_hash, timestamp, status "
                "FROM experiments ORDER BY timestamp DESC"
            ).fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "config": json.loads(row[1]) if row[1] else {},
                    "git_hash": row[2],
                    "timestamp": row[3],
                    "status": row[4],
                })
            return results
        finally:
            conn.close()

    def delete_experiment(self, exp_id: str) -> None:
        """Remove an experiment record."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            conn.commit()
        finally:
            conn.close()
