# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, process-local M0 lifecycle traces (no device synchronization)."""

import atexit
import contextvars
import functools
import json
import os
import time
from pathlib import Path

ENABLED = os.environ.get("AGENTKV_TRACE") == "1"
_request: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentkv_request", default=None
)
_buffer: list[dict] = []
_buffer_pid: int | None = None
_jobs: dict[int, str] = {}


def _json_default(value):
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(type(value).__name__)


def flush_trace() -> None:
    """Serialize buffered events outside the measured request path."""
    if not _buffer:
        return
    directory = Path(os.environ.get("AGENTKV_TRACE_DIR", "/tmp/agentkv-trace"))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"trace-{os.getpid()}.jsonl").open("a") as sink:
        for event in _buffer:
            sink.write(json.dumps(event, default=_json_default) + "\n")
    _buffer.clear()


def trace(event: str, request_id: str | None = None, **fields) -> None:
    """Buffer a monotonic timestamp without device synchronization or JSON I/O."""
    if not ENABLED:
        return
    stamp = time.monotonic_ns()
    global _buffer_pid
    pid = os.getpid()
    if _buffer_pid != pid:
        _buffer.clear()
        _buffer_pid = pid
        atexit.register(flush_trace)
    # Bound memory for long debug runs; M0 stays below this threshold.
    if len(_buffer) >= 65536:
        flush_trace()
    _buffer.append(
        dict(
            ns=stamp,
            pid=pid,
            event=event,
            request_id=request_id or _request.get(),
            **fields,
        )
    )


def bind_job(job_id: int, request_id: str) -> None:
    if ENABLED:
        _jobs[job_id] = request_id


def job_request(job_id: int) -> str | None:
    return _jobs.get(job_id)


def forget_job(job_id: int) -> None:
    _jobs.pop(job_id, None)


def trace_span(event: str):
    """Time a scheduler operation and propagate its request to block events."""

    def decorate(fn):
        if not ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            req_id = None
            for arg in (*args[1:], *kwargs.values()):
                req_id = getattr(arg, "request_id", None) or getattr(
                    arg, "req_id", None
                )
                if req_id:
                    break
            token = _request.set(req_id or _request.get())
            try:
                trace(event + "_begin")
                result = fn(*args, **kwargs)
                scalar_result: int | str | list[int | str | None] | None = None
                if isinstance(result, (int, bool, str)):
                    scalar_result = result
                elif isinstance(result, tuple):
                    scalar_result = [
                        x if isinstance(x, (int, bool, str)) or x is None else None
                        for x in result
                    ]
                trace(event + "_end", result=scalar_result)
                return result
            finally:
                _request.reset(token)

        return wrapped

    return decorate
