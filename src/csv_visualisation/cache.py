from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import duckdb


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT,
                model TEXT,
                output_dim INTEGER,
                source_text TEXT,
                embedding DOUBLE[],
                created_at TIMESTAMP,
                PRIMARY KEY (text_hash, model, output_dim)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS text_cache (
                kind TEXT,
                cache_key TEXT,
                value TEXT,
                created_at TIMESTAMP,
                PRIMARY KEY (kind, cache_key)
            )
            """
        )

    def fetch_embeddings(self, text_hashes: Iterable[str], *, model: str, output_dim: int) -> dict[str, list[float]]:
        text_hashes = list(dict.fromkeys(text_hashes))
        if not text_hashes:
            return {}

        found: dict[str, list[float]] = {}
        chunk_size = 500
        for offset in range(0, len(text_hashes), chunk_size):
            chunk = text_hashes[offset : offset + chunk_size]
            placeholders = ", ".join("?" for _ in chunk)
            params = [model, output_dim, *chunk]
            rows = self.conn.execute(
                f"""
                SELECT text_hash, embedding
                FROM embeddings
                WHERE model = ? AND output_dim = ? AND text_hash IN ({placeholders})
                """,
                params,
            ).fetchall()
            for text_hash, embedding in rows:
                found[str(text_hash)] = [float(value) for value in embedding]
        return found

    def upsert_embedding(
        self,
        *,
        text_hash: str,
        model: str,
        output_dim: int,
        source_text: str,
        embedding: list[float],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO embeddings (text_hash, model, output_dim, source_text, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (text_hash, model, output_dim) DO UPDATE
            SET source_text = excluded.source_text,
                embedding = excluded.embedding,
                created_at = excluded.created_at
            """,
            [
                text_hash,
                model,
                output_dim,
                source_text,
                embedding,
                datetime.now(timezone.utc),
            ],
        )

    def export_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        escaped = str(path).replace("'", "''")
        self.conn.execute(
            f"""
            COPY (
                SELECT text_hash, model, output_dim, source_text, embedding, created_at
                FROM embeddings
                ORDER BY created_at, text_hash
            )
            TO '{escaped}'
            (FORMAT PARQUET)
            """
        )

    def fetch_text(self, kind: str, cache_key: str) -> str:
        row = self.conn.execute(
            "SELECT value FROM text_cache WHERE kind = ? AND cache_key = ?",
            [kind, cache_key],
        ).fetchone()
        return "" if not row else str(row[0])

    def upsert_text(self, kind: str, cache_key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO text_cache (kind, cache_key, value, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (kind, cache_key) DO UPDATE
            SET value = excluded.value,
                created_at = excluded.created_at
            """,
            [kind, cache_key, value, datetime.now(timezone.utc)],
        )

    def close(self) -> None:
        self.conn.close()
