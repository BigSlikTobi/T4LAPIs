from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheItem:
    value: Any
    expires_at: float


class TTLCache:
    """Simple in-memory TTL cache with thread safety.

    - Stores key -> (value, expires_at)
    - Cleans up expired entries opportunistically on get/set
    """

    def __init__(self, default_ttl_s: int = 3600) -> None:
        self._default_ttl = int(default_ttl_s)
        self._data: dict[str, CacheItem] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            if item.expires_at <= now:
                # Expired; remove
                try:
                    del self._data[key]
                except Exception:
                    pass
                return None
            return item.value

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        ttl = int(ttl_s or self._default_ttl)
        expires = time.time() + max(1, ttl)
        with self._lock:
            self._data[key] = CacheItem(value=value, expires_at=expires)


class SQLiteTTLCache:
    """SQLite-backed TTL cache.

    - Thread-safe with a lock; uses a single connection with check_same_thread=False
    - Values stored as JSON text
    - Expires entries on read; optional best-effort cleanup on set
    """

    def __init__(self, db_path: str, default_ttl_s: int = 3600) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._default_ttl = int(default_ttl_s)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            """
            create table if not exists entries (
                k text primary key,
                v text not null,
                expires_at integer not null
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Any]:
        now = int(time.time())
        with self._lock:
            cur = self._conn.execute("select v, expires_at from entries where k = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            v_txt, exp = row
            if int(exp) <= now:
                try:
                    self._conn.execute("delete from entries where k = ?", (key,))
                    self._conn.commit()
                except Exception:
                    pass
                return None
            try:
                return json.loads(v_txt)
            except Exception:
                return None

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        ttl = int(ttl_s or self._default_ttl)
        exp = int(time.time()) + max(1, ttl)
        v_txt = json.dumps(value)
        with self._lock:
            self._conn.execute(
                "insert into entries(k, v, expires_at) values (?, ?, ?) on conflict(k) do update set v=excluded.v, expires_at=excluded.expires_at",
                (key, v_txt, exp),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


class LLMResponseCache:
    """Convenience wrapper combining in-memory and optional SQLite TTL cache."""

    def __init__(self, ttl_s: int = 86400, sqlite_path: Optional[str] = None) -> None:
        self._mem = TTLCache(ttl_s)
        self._disk: Optional[SQLiteTTLCache] = SQLiteTTLCache(sqlite_path, ttl_s) if sqlite_path else None

    def get(self, key: str) -> Optional[Any]:
        # First check memory
        val = self._mem.get(key)
        if val is not None:
            return val
        # Check disk
        if self._disk is not None:
            val = self._disk.get(key)
            if val is not None:
                # Promote to memory
                self._mem.set(key, val)
            return val
        return None

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        self._mem.set(key, value, ttl_s)
        if self._disk is not None:
            self._disk.set(key, value, ttl_s)
