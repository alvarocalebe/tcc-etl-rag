#!/usr/bin/env python3
"""Aguarda o Postgres aceitar conexões (útil no startup do container app)."""

import os
import sys
import time

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://dengue_user:dengue_pass@postgres:5432/dengue_db",
)


def _psycopg2_dsn(url: str) -> str:
    """SQLAlchemy usa postgresql+psycopg2://; psycopg2 espera postgresql://."""
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + url[len("postgresql+psycopg2://") :]
    return url


MAX_ATTEMPTS = 60
SLEEP_SEC = 2


def main() -> int:
    dsn = _psycopg2_dsn(DATABASE_URL)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            conn.close()
            print("Postgres está pronto.", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{attempt}/{MAX_ATTEMPTS}] Aguardando Postgres: {exc}",
                flush=True,
            )
            time.sleep(SLEEP_SEC)
    print("Timeout aguardando Postgres.", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
