"""Checkpointer adapter.

Supports memory, sqlite, and postgres backends for LangGraph state persistence.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def build_checkpointer(  # noqa: ANN201
    kind: str = "memory",
    database_url: str | None = None,
) -> Any | None:  # noqa: ANN401
    """Return a LangGraph checkpointer.

    Supported kinds:
    - 'none'   → no persistence
    - 'memory' → MemorySaver (in-process, dev/test)
    - 'sqlite' → SqliteSaver with WAL mode (single-node production)
    - 'postgres' → PostgresSaver (multi-node production)
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = database_url or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # Enable WAL mode for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc
        return PostgresSaver.from_conn_string(database_url or "")

    raise ValueError(f"Unknown checkpointer kind: {kind}")
