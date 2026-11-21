# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-batch composition tracing for the v1 scheduler.

When `VLLM_BATCH_TRACE_DIR` is set, the scheduler appends one JSON record
per scheduled batch (plus one per batch completion) to rotating `.jsonl`
files in that directory. Records contain only request identifiers and token
counts — never prompt text or token ids — so traces are safe to keep on
shared storage.

Records are flushed to the OS after every write, so a `kill -9` of the
engine process does not lose records that were already produced. There is
no `fsync`: durability against a machine crash is intentionally out of
scope to keep the per-step overhead in the microsecond range.

The recorder only ever runs on the engine-core thread that owns the
scheduler, so it needs no locking. Given the same scheduling decisions the
output is deterministic; scheduling itself is timing-dependent, which is
exactly what the `diff` command below is meant to surface.

Analysis CLI:

    python -m vllm.v1.core.batch_trace summarize <trace-dir-or-file>
    python -m vllm.v1.core.batch_trace diff <run-a> <run-b>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

TRACE_VERSION = 1

# Disable the recorder after this many consecutive write failures so a
# broken disk cannot add per-step overhead or log spam indefinitely.
_MAX_CONSECUTIVE_FAILURES = 64

_DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024


class BatchTraceRecorder:
    """Rotating JSONL writer for per-batch composition records.

    All degradation paths are fail-open: any I/O error is logged and
    counted, and inference continues. Use `enabled` to check health and
    `records_dropped` for the number of records that failed to write.
    """

    def __init__(
        self,
        directory: str,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        max_file_seconds: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.directory = directory
        self.max_file_bytes = max_file_bytes
        self.max_file_seconds = max_file_seconds
        self.meta = meta or {}

        self.enabled = False
        self._file: Any = None
        self._segment = -1
        self._segment_bytes = 0
        self._segment_start_mono = 0.0
        self._seq = 0
        self._open_batches: deque[int] = deque()

        # Observability into the recorder itself.
        self.records_written = 0
        self.records_dropped = 0
        self.unmatched_step_ends = 0
        self.overhead_ns = 0
        self._consecutive_failures = 0

        try:
            os.makedirs(directory, exist_ok=True)
            self._open_next_segment()
            self.enabled = True
        except OSError as e:
            logger.warning(
                "Batch trace directory %s is not writable (%s); "
                "batch tracing is disabled, inference continues.",
                directory,
                e,
            )

    @classmethod
    def from_env(cls, meta: dict[str, Any] | None = None) -> BatchTraceRecorder | None:
        """Create a recorder from env config, or None when tracing is off."""
        import vllm.envs as envs

        directory = envs.VLLM_BATCH_TRACE_DIR
        if not directory:
            return None
        return cls(
            directory,
            max_file_bytes=envs.VLLM_BATCH_TRACE_MAX_FILE_BYTES,
            max_file_seconds=envs.VLLM_BATCH_TRACE_MAX_FILE_SECONDS,
            meta=meta,
        )

    @property
    def avg_record_overhead_ns(self) -> float:
        attempts = self.records_written + self.records_dropped
        return self.overhead_ns / attempts if attempts else 0.0

    def _segment_path(self, segment: int) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"batch_trace-{os.getpid()}-{stamp}-{segment:04d}.jsonl"
        return os.path.join(self.directory, name)

    def _open_next_segment(self) -> None:
        if self._file is not None:
            self._file.close()
        self._segment += 1
        self._segment_bytes = 0
        self._segment_start_mono = time.monotonic()
        self._file = open(  # noqa: SIM115 - handle owned by the recorder
            self._segment_path(self._segment), "w", encoding="utf-8"
        )
        self._write_record(
            {
                "type": "meta",
                "trace_version": TRACE_VERSION,
                "pid": os.getpid(),
                "segment": self._segment,
                "ts": time.time(),
                **self.meta,
            }
        )

    def _rotate_if_needed(self, mono_now: float) -> None:
        if self._segment_bytes < self.max_file_bytes and (
            self.max_file_seconds <= 0
            or mono_now - self._segment_start_mono < self.max_file_seconds
        ):
            return
        self._open_next_segment()

    def _write_record(self, record: dict[str, Any]) -> None:
        # ensure_ascii=True keeps len(line) == len(line.encode()) so the
        # byte counter used for rotation stays exact.
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._file.write(line)
        # Flush after every record: already-produced records must survive
        # SIGKILL. The data reaches the OS page cache; no fsync by design.
        self._file.flush()
        self._segment_bytes += len(line)
        self.records_written += 1

    def _emit(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._rotate_if_needed(time.monotonic())
            self._write_record(record)
            self._consecutive_failures = 0
        except OSError as e:
            self.records_dropped += 1
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                logger.warning(
                    "Failed to write batch trace record (%s); "
                    "dropping it and continuing.",
                    e,
                )
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Batch trace writes failed %d times in a row; "
                    "disabling batch tracing, inference continues.",
                    self._consecutive_failures,
                )
                self.enabled = False

    def record_batch(
        self,
        *,
        new_reqs: list[Request],
        cached_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        num_waiting: int,
        running_reqs: list[Request],
        kv_free_blocks: int,
        kv_total_blocks: int,
        block_size: int,
        num_preempted: int,
        finished_req_ids: Iterable[str],
    ) -> None:
        """Record one scheduled batch. Only ids and lengths are captured."""
        if not self.enabled:
            return
        start_ns = time.perf_counter_ns()
        try:
            reqs = []
            for is_new, req_list in ((True, new_reqs), (False, cached_reqs)):
                for req in req_list:
                    reqs.append(
                        {
                            "id": req.request_id,
                            "new": is_new,
                            "phase": (
                                "prefill" if req.num_output_tokens == 0 else "decode"
                            ),
                            "sched": num_scheduled_tokens[req.request_id],
                            "prompt": req.num_prompt_tokens,
                            "computed": req.num_computed_tokens,
                            "output": req.num_output_tokens,
                        }
                    )
            kv_used_blocks = kv_total_blocks - kv_free_blocks
            record = {
                "type": "batch",
                "seq": self._seq,
                "ts": time.time(),
                "mono": time.monotonic(),
                "reqs": reqs,
                "num_reqs": len(reqs),
                "tokens": sum(num_scheduled_tokens.values()),
                "waiting": num_waiting,
                "running": len(running_reqs),
                "kv_usage": (kv_used_blocks / kv_total_blocks)
                if kv_total_blocks
                else 0.0,
                "kv_free": kv_free_blocks,
                "kv_total": kv_total_blocks,
                # Allocated KV slots vs. tokens actually held by running
                # requests; the gap is internal fragmentation (tail-block
                # padding plus slots reserved for future appends).
                "kv_slots": kv_used_blocks * block_size,
                "kv_tokens": sum(req.num_tokens for req in running_reqs),
                "preempted": num_preempted,
                "finished": sorted(finished_req_ids),
            }
            self._open_batches.append(self._seq)
            self._seq += 1
            self._emit(record)
        finally:
            self.overhead_ns += time.perf_counter_ns() - start_ns

    def record_step_end(self) -> None:
        """Record completion of the oldest unclosed batch (FIFO order)."""
        if not self.enabled:
            return
        start_ns = time.perf_counter_ns()
        try:
            if not self._open_batches:
                # A batch was scheduled before tracing started or its record
                # was dropped; count and move on.
                self.unmatched_step_ends += 1
                return
            seq = self._open_batches.popleft()
            self._emit(
                {
                    "type": "step_end",
                    "seq": seq,
                    "ts": time.time(),
                    "mono": time.monotonic(),
                }
            )
        finally:
            self.overhead_ns += time.perf_counter_ns() - start_ns

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None
        if self.records_dropped:
            logger.warning(
                "Batch trace recorder dropped %d records due to I/O errors.",
                self.records_dropped,
            )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _resolve_files(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")
        )
    return [path]


def load_records(*paths: str) -> list[dict[str, Any]]:
    """Load all trace records from files or directories, in file order."""
    records: list[dict[str, Any]] = []
    for path in paths:
        for file in _resolve_files(path):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * pct / 100.0))
    return sorted_vals[idx]


def summarize(
    records: list[dict[str, Any]],
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> dict[str, Any]:
    """Aggregate batch composition and KV-cache utilization over a window."""
    batches = [
        r
        for r in records
        if r["type"] == "batch"
        and (start_ts is None or r["ts"] >= start_ts)
        and (end_ts is None or r["ts"] <= end_ts)
    ]
    if not batches:
        return {"num_batches": 0}
    step_ends = {r["seq"]: r for r in records if r["type"] == "step_end"}
    durations = sorted(
        step_ends[b["seq"]]["mono"] - b["mono"]
        for b in batches
        if b["seq"] in step_ends
    )
    sizes = sorted(b["num_reqs"] for b in batches)
    kv_usage = sorted(b["kv_usage"] for b in batches)
    waste = sorted(
        1.0 - b["kv_tokens"] / b["kv_slots"] for b in batches if b["kv_slots"] > 0
    )
    total_tokens = sum(b["tokens"] for b in batches)
    span = max(b["ts"] for b in batches) - min(b["ts"] for b in batches)
    return {
        "num_batches": len(batches),
        "span_s": round(span, 3),
        "total_scheduled_tokens": total_tokens,
        "tokens_per_s": round(total_tokens / span, 1) if span > 0 else None,
        "batch_size_reqs": {
            "avg": round(sum(sizes) / len(sizes), 2),
            "min": sizes[0],
            "p50": _percentile(sizes, 50),
            "max": sizes[-1],
        },
        "kv_usage": {
            "avg": round(sum(kv_usage) / len(kv_usage), 4),
            "peak": kv_usage[-1],
        },
        "kv_waste_ratio": (
            {"avg": round(sum(waste) / len(waste), 4), "peak": waste[-1]}
            if waste
            else None
        ),
        "step_duration_ms": (
            {
                "avg": round(sum(durations) / len(durations) * 1e3, 3),
                "p50": round(_percentile(durations, 50) * 1e3, 3),
                "p95": round(_percentile(durations, 95) * 1e3, 3),
                "max": round(durations[-1] * 1e3, 3),
            }
            if durations
            else None
        ),
    }


def _batch_req_ids(record: dict[str, Any]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for req in record["reqs"]:
        ids[req["id"]] = ids.get(req["id"], 0) + 1
    return ids


def diff_traces(
    records_a: list[dict[str, Any]], records_b: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare batch composition between two runs of the same workload.

    Batches are aligned by schedule order (`seq`). The report pinpoints the
    first batch whose request set differs and whether any throughput gap
    comes from smaller batches or from longer per-batch execution time.
    """
    batches_a = [r for r in records_a if r["type"] == "batch"]
    batches_b = [r for r in records_b if r["type"] == "batch"]

    first_divergence = None
    num_differing = 0
    for i, (ba, bb) in enumerate(zip(batches_a, batches_b)):
        ids_a, ids_b = _batch_req_ids(ba), _batch_req_ids(bb)
        if ids_a != ids_b:
            num_differing += 1
            if first_divergence is None:
                first_divergence = {
                    "index": i,
                    "seq_a": ba["seq"],
                    "seq_b": bb["seq"],
                    "only_in_a": sorted(set(ids_a) - set(ids_b)),
                    "only_in_b": sorted(set(ids_b) - set(ids_a)),
                }

    def first_scheduled(batches: list[dict[str, Any]]) -> dict[str, int]:
        order: dict[str, int] = {}
        for i, b in enumerate(batches):
            for req in b["reqs"]:
                order.setdefault(req["id"], i)
        return order

    order_a, order_b = first_scheduled(batches_a), first_scheduled(batches_b)
    admission_deltas = {
        rid: order_b[rid] - order_a[rid]
        for rid in sorted(set(order_a) & set(order_b))
        if order_a[rid] != order_b[rid]
    }

    summary_a = summarize(records_a)
    summary_b = summarize(records_b)
    return {
        "num_batches_a": len(batches_a),
        "num_batches_b": len(batches_b),
        "num_differing_batches": num_differing,
        "first_divergence": first_divergence,
        "num_requests_admitted_in_different_order": len(admission_deltas),
        "admission_order_deltas": dict(list(admission_deltas.items())[:20]),
        "summary_a": summary_a,
        "summary_b": summary_b,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.v1.core.batch_trace",
        description="Inspect and compare vLLM batch composition traces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize", help="Aggregate one trace.")
    p_sum.add_argument("path", help="Trace file or directory.")
    p_sum.add_argument("--start", type=float, default=None, help="Start UNIX ts.")
    p_sum.add_argument("--end", type=float, default=None, help="End UNIX ts.")

    p_diff = sub.add_parser("diff", help="Compare two runs' batch composition.")
    p_diff.add_argument("run_a", help="Trace file or directory of run A.")
    p_diff.add_argument("run_b", help="Trace file or directory of run B.")

    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = summarize(load_records(args.path), args.start, args.end)
    else:
        result = diff_traces(load_records(args.run_a), load_records(args.run_b))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
