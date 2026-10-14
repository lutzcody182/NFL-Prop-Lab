"""Database helpers: connection, schema setup, bulk loading, content hashing."""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import psycopg

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def connect() -> psycopg.Connection:
    """Connect using DATABASE_URL, defaulting to the local dev database."""
    url = os.environ.get("DATABASE_URL", "postgresql://postgres@/nflprop?host=/tmp")
    return psycopg.connect(url)


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())
    conn.commit()


def replace_rows(conn: psycopg.Connection, table: str, df: pl.DataFrame,
                 delete_where: str | None = None, params: tuple = ()) -> int:
    """Delete the slice being reloaded, then COPY the new rows in.

    Reloading a whole slice (e.g. a season) makes every run idempotent:
    running it twice leaves identical content.
    """
    cols = df.columns
    with conn.cursor() as cur:
        if delete_where:
            cur.execute(f"DELETE FROM {table} WHERE {delete_where}", params)
        else:
            cur.execute(f"DELETE FROM {table}")
        with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as copy:
            for row in df.iter_rows():
                copy.write_row(row)
    return df.height


def table_hash(conn: psycopg.Connection, table: str, order_by: str) -> str:
    """md5 of the table's content (minus loaded_at) in a fixed order."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND column_name <> 'loaded_at' ORDER BY ordinal_position",
            (table,),
        )
        cols = [r[0] for r in cur.fetchall()]
        row_expr = "ROW(" + ", ".join(cols) + ")::text"
        cur.execute(f"SELECT md5(coalesce(string_agg({row_expr}, '|' ORDER BY {order_by}), '')) FROM {table}")
        return cur.fetchone()[0]
