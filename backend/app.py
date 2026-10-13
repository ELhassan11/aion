"""AION API — minimal skeleton (P0-T02).

Full routers (ingestion, RAG, agents, customers, eval) land in later phases.
This skeleton proves the unified dev environment: one `docker compose up`
brings up postgres + neo4j + redis + this API, and /health + /ready verify it.
"""

import os
import socket

from fastapi import FastAPI

app = FastAPI(title="AION API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Liveness: is the API process itself running?"""
    return {"status": "ok", "service": "aion-api"}


def _tcp_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@app.get("/ready")
def ready() -> dict:
    """Readiness: can the API reach its dependencies?"""
    deps = {
        "postgres": (os.getenv("POSTGRES_HOST", "postgres"), 5432),
        "redis": (os.getenv("REDIS_HOST", "redis"), 6379),
        "neo4j": (os.getenv("NEO4J_HOST", "neo4j"), 7687),
    }
    status = {name: "up" if _tcp_ok(host, port) else "down" for name, (host, port) in deps.items()}
    return {"ready": all(v == "up" for v in status.values()), "dependencies": status}
