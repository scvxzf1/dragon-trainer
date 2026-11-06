"""Fresh-process runner for adaptive training workers with structured results."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from library.training.auto_block_swap.process import stop_process, write_result
from library.training.auto_block_swap.resources import host_memory


@contextmanager
def _cancellation_signals():
    """Defer main-thread cancellation to the polling loop for owned cleanup."""
    received = []
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda number, frame: received.append(number))
    try:
        yield received
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class IsolatedRunner:
    """command(request_path) returns argv, never a shell command.

    Workers read request.json, write result.json atomically, and exit zero even
    for a handled CUDA OOM. Crashes, timeout and host pressure are not retried as
    GPU OOM. Every attempt owns a new directory and process group.
    """

    def __init__(self, directory: Path, command, *, timeout=1800, env=None,
                 resources=host_memory):
        if timeout <= 0:
            raise ValueError("Worker timeout must be positive")
        self.directory = Path(directory)
        self.command = command
        self.timeout = timeout
        self.env = env or {}
        self.resources = resources

    def __call__(self, plan, *, attempt, resume):
        directory = self.directory / f"attempt-{attempt:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        request = directory / "request.json"
        write_result(request, {"plan": asdict(plan), "resume": resume})
        request.chmod(0o600)
        result = {"status": "error", "error": "worker did not produce a result"}
        started = time.monotonic()
        process = None
        with _cancellation_signals() as cancelled:
            try:
                memory = self.resources()
                if memory.available <= memory.reserve:
                    result = {"status": "host_limit"}
                else:
                    with (directory / "worker.log").open("wb") as log:
                        process = subprocess.Popen(
                            self.command(request), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, **self.env}, start_new_session=os.name == "posix",
                        )
                        output_thread = threading.Thread(
                            target=self._relay_output,
                            args=(process.stdout, log),
                            name="adaptive-worker-output",
                            daemon=True,
                        )
                        output_thread.start()
                        failure = self._wait(process, started, cancelled)
                        process.wait()
                        output_thread.join(timeout=2.0)
                        if output_thread.is_alive():
                            # A detached descendant can keep the pipe open after
                            # the worker itself exits.  Do not let that delay
                            # timeout/cancel handling indefinitely.
                            process.stdout.close()
                            output_thread.join(timeout=1.0)
                    result = self._result(directory, process.returncode, failure)
            finally:
                if process is not None:
                    stop_process(process)
                if cancelled:
                    result = {"status": "cancelled", "signal": cancelled[0]}
                request.unlink(missing_ok=True)
                result["elapsed_seconds"] = time.monotonic() - started
                write_result(directory / "supervisor.json", result)
        return result

    @staticmethod
    def _relay_output(stream, log) -> None:
        """Persist worker output and mirror it to the supervisor's stdout."""
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                log.write(chunk)
                log.flush()
                try:
                    target = getattr(sys.stdout, "buffer", None)
                    if target is None:
                        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                    else:
                        target.write(chunk)
                        target.flush()
                except (BrokenPipeError, OSError, ValueError):
                    # The WebUI/parent pipe may close while the worker is
                    # still unwinding; the on-disk log remains authoritative.
                    pass
        except (OSError, ValueError):
            return

    def _wait(self, process, started, cancelled):
        while process.poll() is None:
            memory = self.resources()
            failure = None
            if cancelled:
                failure = "cancelled"
            elif memory.available <= memory.reserve:
                failure = "host_limit"
            elif time.monotonic() - started >= self.timeout:
                failure = "timeout"
            if failure:
                stop_process(process)
                return failure
            time.sleep(0.05)
        return None

    @staticmethod
    def _result(directory, returncode, failure):
        if failure:
            return {"status": failure}
        if returncode != 0:
            return {"status": "error", "returncode": returncode}
        try:
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"status": "error", "error": str(exc)}
        if not isinstance(result, dict) or result.get("status") not in {
            "ok",
            "cuda_oom",
            "error",
            "host_limit",
            "nonfinite_skip",
        }:
            return {"status": "error", "error": "invalid structured worker result"}
        return result
