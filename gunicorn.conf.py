"""Gunicorn configuration for the sashar.dev ML + LLM FastAPI service.

Run with:  poetry run gunicorn app.main:app -c gunicorn.conf.py

Uses Uvicorn workers so FastAPI's ASGI app runs under gunicorn's process
manager. Keep worker count low — each worker loads its own copy of distilgpt2
into memory, and the target VM is small and GPU-less (CLAUDE.md).
"""
import os
import sys

# ── Native-library / fork safety ──────────────────────────────────────────────
# This config file runs in the gunicorn master BEFORE workers are forked, so
# environment set here is inherited by every worker. torch and lightgbm each
# bundle an OpenMP runtime; loading both (app.main imports the ml + llm routers)
# can crash a forked worker with SIGSEGV, and macOS additionally needs objc
# fork-safety disabled. OMP_NUM_THREADS=1 also avoids CPU thread oversubscription
# on the small GPU-less VM (CLAUDE.md). setdefault() so explicit env wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Bind address / port (PORT overridable to match the rest of the stack).
bind = f"0.0.0.0:{os.getenv('PORT', '8002')}"

# ASGI worker class. uvicorn.workers.UvicornWorker ships with uvicorn<0.34
# (pinned via pyproject uvicorn ^0.30). On newer uvicorn use the standalone
# `uvicorn-worker` package and set worker_class = "uvicorn_worker.UvicornWorker".
worker_class = "uvicorn.workers.UvicornWorker"

# One worker by default — the model is memory-heavy. Scale via WEB_CONCURRENCY.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))

# CPU inference (model download on first call, generation) can be slow.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5

# Preloading would load the model once in the master, but distilgpt2 is loaded
# lazily on first request anyway, so leave preload off to keep worker restarts cheap.
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
